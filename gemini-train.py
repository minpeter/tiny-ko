import argparse
import logging
import os
from itertools import chain

import torch
from datasets import load_dataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Llama 모델을 처음부터 사전학습합니다.")
    parser.add_argument("--dataset_path", type=str, required=True, help="훈련 데이터셋 경로 (텍스트 파일들이 있는 디렉토리).")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="사전 훈련된 사용자 정의 토크나이저 경로.")
    parser.add_argument("--output_dir", type=str, default="./pretrain_output", help="모델 체크포인트와 결과를 저장할 디렉토리.")
    parser.add_argument("--model_config_name", type=str, default="small", help="사용할 모델 크기 설정 (small, medium, large).")
    # 학습 관련 인자 추가
    parser.add_argument("--batch_size", type=int, default=8, help="디바이스당 훈련 배치 크기.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="총 훈련 에포크 수.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="학습률.")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Flash Attention 2 사용 여부.")
    parser.add_argument("--packing", action="store_true", help="시퀀스 패킹 사용 여부.")

    args = parser.parse_args()

    # 1. 토크나이저 로드
    logger.info(f"토크나이저 로딩: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 모델 구성 정의
    model_configs = {
        "small": LlamaConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=4096),
        "medium": LlamaConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, max_position_embeddings=4096),
        "large": LlamaConfig(hidden_size=2048, num_hidden_layers=24, num_attention_heads=16, intermediate_size=5504, max_position_embeddings=4096),
    }
    config = model_configs.get(args.model_config_name, model_configs["small"])
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    if args.use_flash_attention_2:
        config.attn_implementation = "flash_attention_2"

    # 3. 모델 초기화 (무작위 가중치)
    attn_implementation = "flash_attention_2" if args.use_flash_attention_2 else "eager"
    logger.info(f"'{args.model_config_name}' 설정으로 모델 초기화 중... (Attention: {attn_implementation})")
    model = LlamaForCausalLM(config)
    
    # 4. 데이터셋 준비 및 처리
    logger.info("데이터셋 로딩 및 처리 중...")
    # raw_datasets = load_dataset("text", data_files={"train": os.path.join(args.dataset_path, "*.txt")})
    raw_datasets = load_dataset(args.dataset_path, split="train")
    
    # 토크나이즈 함수
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
    )

    # block_size = tokenizer.model_max_length
    block_size = 4096  # 모델의 최대 시퀀스 길이로 설정

    # 그룹화 함수 (시퀀스 패킹의 기초)
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
    )

    # 5. 데이터 콜레이터 설정
    if args.packing and args.use_flash_attention_2:
        logger.info("Flash Attention 2와 호환되는 패딩 없는(pad-free) 데이터 콜레이터를 사용합니다.")
        from transformers import DataCollatorWithFlattening
        data_collator = DataCollatorWithFlattening()
    else:
        logger.info("표준 언어 모델링 데이터 콜레이터를 사용합니다.")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Trainer 설정
    training_args = TrainingArguments(
        # bs 16
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True, # A100/H100의 경우 bf16=True 사용
        torch_compile=True,
        # report_to="tensorboard",
        use_cpu=not torch.cuda.is_available(), # CUDA가 없을 경우 CPU 사용
        # attn_implementation=attn_implementation,
    )

    print(lm_datasets)
    print(lm_datasets[0])
    print("Collated batch example:")
    print(data_collator(lm_datasets[0:2]))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 7. 훈련 시작
    logger.info("사전학습을 시작합니다.")
    trainer.train()

    # 8. 최종 모델 저장
    logger.info("훈련 완료. 최종 모델을 저장합니다.")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()