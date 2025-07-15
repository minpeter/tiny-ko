import os
import torch
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    # DataCollatorForLanguageModeling를 임포트합니다.
    DataCollatorWithFlattening,
)
from tqdm import tqdm
from muon_optimizer import Muon

# MoonDataset 클래스는 더 이상 필요 없으므로 삭제합니다.

# get_model_and_dataloader 함수를 DataCollator 방식으로 수정합니다.
def get_model_and_dataloader(model_name, dataset_name, max_seq_length):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        "./tknz/tiny-ko-tokenizer"
    )

    # 데이터셋 로드
    raw_dataset = load_dataset(name2path[dataset_name], split="train")

    # 토큰화 함수 정의
    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples['text'])

        if tokenizer.eos_token_id is not None:
            for i in range(len(tokenized_inputs["input_ids"])):
                tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
                tokenized_inputs["attention_mask"][i].append(1)
                if "token_type_ids" in tokenized_inputs:
                    tokenized_inputs["token_type_ids"][i].append(0)

        return tokenized_inputs

    # .map()을 사용하여 데이터셋 전체를 효율적으로 토큰화합니다.
    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=os.cpu_count(), # 사용 가능한 모든 CPU 코어를 활용해 병렬 처리
        remove_columns=raw_dataset.column_names, # 기존 text 컬럼은 제거
    )
    
    # Causal LM을 위한 데이터 콜레이터를 생성합니다. mlm=False가 핵심입니다.
    # 이 콜레이터가 여러 샘플을 max_seq_length 길이의 배치로 묶어줍니다.
    data_collator = DataCollatorWithFlattening()

    # 수정된 데이터셋과 콜레이터로 DataLoader를 생성합니다.
    train_loader = DataLoader(
        tokenized_dataset, batch_size=8, shuffle=True, collate_fn=data_collator
    )

    # --- 모델 설정 부분은 기존과 동일 ---
    if model_name == "qwen":
        config = Qwen2Config(
            attn_implementation="flash_attention_2",
            hidden_size=1024,
            intermediate_size=4864,
            max_position_embeddings=max_seq_length,
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            tie_word_embeddings=True,
            torch_dtype=torch.bfloat16,
            vocab_size=len(tokenizer),
            # ... 기타 Qwen 설정 ...
        )
        model = Qwen2ForCausalLM(config)
    elif model_name == "llama":
        config = LlamaConfig(
            hidden_size=480,
            num_hidden_layers=32,
            intermediate_size=1920,
            tie_word_embeddings=True,
            num_attention_heads=6,
            num_key_value_heads=2,
            vocab_size=len(tokenizer),
            max_position_embeddings=max_seq_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = LlamaForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"

    return model, train_loader, tokenizer


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the trained model")
    parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum sequence length for tokenization and model.")
    args = parser.parse_args()
    
    log_file_name = f"train_{args.model}_{args.optimizer}_lr{args.lr}.log"
    logger.add(os.path.join("logs", log_file_name))

    output_path = os.path.join(args.output_dir, f"{args.model}_{args.optimizer}_lr{args.lr}")
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Model will be saved to: {output_path}")

    # get_model_and_dataloader 호출 시 max_seq_length를 전달합니다.
    # DataCollator가 내부적으로 이 값을 사용하지는 않지만, 모델 설정에 필요합니다.
    model, train_loader, tokenizer = get_model_and_dataloader(
        args.model, args.dataset, args.max_seq_length
    )
    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr, wd=args.wd
    )

    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(torch.bfloat16)
    model.to(device)
    # model = torch.compile(model)

    model.train()
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    for epoch in range(epoch):
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # DataCollator가 생성한 배치는 'input_ids', 'attention_mask', 'labels' 키를 가집니다.
            # 'labels'는 자동으로 생성되므로 모델에 그대로 전달하면 됩니다.
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if step % 50 == 0:
                logger.info(
                    f"Epoch: {epoch} Step: {step} LR: {optimizer.param_groups[0]['lr']:.6f} Training loss: {loss.item():.4f}"
                )
    logger.info("Training finished. Saving model and tokenizer...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"Model and tokenizer saved successfully to {output_path}")