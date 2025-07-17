import os
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    DataCollatorWithFlattening,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# --- 1. 설정 및 상수 ---
MODEL_NAME = "./artifacts/tknz/tiny-ko-tokenizer"
DATASET_NAME = "Elriggs/openwebtext-100k"
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "./outputs/test_improved"

def main():
    # --- 2. 토크나이저 및 모델 설정 ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 모델 설정을 LlamaConfig를 사용하여 명시적으로 정의합니다.
    config = LlamaConfig(
        hidden_size=480,
        num_hidden_layers=32,
        intermediate_size=1920,
        tie_word_embeddings=True,
        num_attention_heads=6,
        num_key_value_heads=2,
        vocab_size=len(tokenizer),
        max_position_embeddings=MAX_SEQ_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        rope_theta=10000.0,
        use_cache=False,  # 학습 중에는 캐시를 사용하지 않는 것이 좋습니다.
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = LlamaForCausalLM(config)
    model.to(torch.bfloat16)

    # --- 3. 데이터셋 전처리 (Padding-Free 방식) ---
    # 먼저 모든 텍스트를 토큰화하고 하나의 긴 시퀀스로 결합합니다.
    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples['text'])

        if tokenizer.eos_token_id is not None:
            for i in range(len(tokenized_inputs["input_ids"])):
                tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
                tokenized_inputs["attention_mask"][i].append(1)
                if "token_type_ids" in tokenized_inputs:
                    tokenized_inputs["token_type_ids"][i].append(0)

        return tokenized_inputs

    raw_dataset = load_dataset(DATASET_NAME, split="train")
    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=raw_dataset.column_names
    )

    data_collator = DataCollatorWithFlattening()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,  # 사용 가능한 VRAM에 맞춰 조정하세요.
        gradient_accumulation_steps=4,  # 배치 크기를 효과적으로 늘립니다. (4 * 4 = 16)
        learning_rate=1e-4,
        weight_decay=0.1,
        bf16=True,
        torch_compile=True,
        logging_steps=50,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=True,
        dataloader_num_workers=os.cpu_count(),
        dataloader_prefetch_factor=2,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        eval_steps=500,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # --- 5. 학습 시작 ---
    trainer.train()

if __name__ == "__main__":
    main()