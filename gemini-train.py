import argparse
import logging
import math
import os

import torch
from datasets import load_from_disk
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from muon_optimizer import create_muon_optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Llama 모델을 처음부터 사전학습합니다.")
    parser.add_argument("--dataset_path", type=str, required=True, help="훈련 데이터셋 경로 (텍스트 파일들이 있는 디렉토리).")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="사전 훈련된 사용자 정의 토크나이저 경로.")
    parser.add_argument("--output_dir", type=str, default="./pretrain_output", help="모델 체크포인트와 결과를 저장할 디렉토리.")
    parser.add_argument("--model_config_name", type=str, default="small", help="사용할 모델 크기 설정 (small, medium, large).")

    parser.add_argument("--num_train_epochs", type=int, default=1, help="총 훈련 에포크 수.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="학습률.")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Flash Attention 2 사용 여부.")
    parser.add_argument("--packing", action="store_true", help="시퀀스 패킹 사용 여부.")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="모델의 최대 시퀀스 길이 (block size).")
    # weight_decay
    parser.add_argument("--weight_decay", type=float, default=0.1, help="옵티마이저의 weight decay 값.")
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"], help="사용할 옵티마이저 종류.")
    # upload hf model id
    parser.add_argument("--hf_model_id", type=str, default=None, help="Hugging Face 모델 ID.")

    args = parser.parse_args()

    logger.info(f"토크나이저 로딩: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('tokenizer.vocab_size:', tokenizer.vocab_size)
    print('len(tokenizer):', len(tokenizer))

    # 2. 모델 구성 정의
    model_configs = {
        "small": LlamaConfig(initializer_range=(1/ math.sqrt(768)), hidden_size=768, num_hidden_layers=25, intermediate_size=1920, tie_word_embeddings=True, num_attention_heads=12, num_key_value_heads=4),
        "smollm": LlamaConfig(hidden_size=576, num_hidden_layers=30, intermediate_size=1536, tie_word_embeddings=True, num_attention_heads=9, num_key_value_heads=3),
        # "medium": LlamaConfig(hidden_size=768, num_hidden_layers=29, intermediate_size=1920, tie_word_embeddings=True, num_attention_heads=12, num_key_value_heads=4),
        # "large": LlamaConfig(hidden_size=2048, num_hidden_layers=24, num_attention_heads=16, intermediate_size=5504),
    }
    config = model_configs.get(args.model_config_name, model_configs["small"])
    config.torch_dtype = torch.bfloat16
    config.vocab_size = len(tokenizer)
    config.use_cache = False

    config.max_position_embeddings = args.max_seq_length

    # rope_theta 설정
    if config.max_position_embeddings >= 8192:
        config.rope_theta = 1_000_000.0  # 또는 500_000.0로 변경 가능
    else:
        config.rope_theta = 10_000.0  # 기본값

    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.eos_token_id # Qwen 스타일로, 모델 설정의 BOS만 이렇게 설정, 실제로는 사용 X
    config.eos_token_id = tokenizer.eos_token_id
    if args.use_flash_attention_2:
        config._attn_implementation = "flash_attention_2"

    attn_implementation = "flash_attention_2" if args.use_flash_attention_2 else "eager"
    logger.info(f"'{args.model_config_name}' 설정으로 모델 초기화 중... (Attention: {attn_implementation})")
    model = LlamaForCausalLM(config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**3:.2f}B parameters")
    print(f"Model size: {model_size/1000**2:.2f}M parameters")
    print(f"Model size: {model_size/1000:.1f}K parameters")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(torch.bfloat16)
    model.to(device)
    
    if args.optimizer == "muon":
        logger.info("Muon 옵티마이저를 생성합니다.")
        optimizer = create_muon_optimizer(
            model, 
            lr=args.learning_rate, 
            wd=args.weight_decay,
        )
    else:
        logger.info("AdamW 옵티마이저를 생성합니다.")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
    
    logger.info("Preprocessing 데이터셋...")
    lm_datasets = load_from_disk(args.dataset_path)
    
    print("\n로딩 완료된 데이터셋 구조:")
    print(lm_datasets)
    print(f"훈련 샘플 수: {len(lm_datasets['train'])}")
    print(f"테스트 샘플 수: {len(lm_datasets['test'])}")
    print(f"샘플 0의 토큰 수: {len(lm_datasets['train'][0]['input_ids'])}")

    # 5. 데이터 콜레이터 설정
    if args.packing and args.use_flash_attention_2:
        logger.info("Flash Attention 2와 호환되는 패딩 없는(pad-free) 데이터 콜레이터를 사용합니다.")
        from transformers import DataCollatorWithFlattening
        data_collator = DataCollatorWithFlattening()
    else:
        logger.info("표준 언어 모델링 데이터 콜레이터를 사용합니다.")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    print(lm_datasets["train"])
    print("Collated batch example:")
    sample_batch = [lm_datasets["train"][i] for i in range(min(2, len(lm_datasets["train"])))]
    batch = data_collator(sample_batch)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
            print(value)
        else:
            print(f"{key}: {value}")

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    if args.hf_model_id:
        logger.info(f"Hugging Face 모델 ID '{args.hf_model_id}'에 토크나이저를 푸시합니다.")
        tokenizer.push_to_hub(args.hf_model_id)

    # 6. Trainer 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        auto_find_batch_size=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        # --- Muon이 weight decay를 자체 처리하므로 Trainer에서는 0으로 설정 ---
        weight_decay=0.0,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=1,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    trainer.train()

if __name__ == "__main__":
    main()