import os
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# --- 1. 설정 및 상수 ---
MODEL_NAME = "./tknz/tiny-ko-tokenizer"
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

    # --- 3. 데이터셋 전처리 (Padding-Free 방식) ---
    # 먼저 모든 텍스트를 토큰화하고 하나의 긴 시퀀스로 결합합니다.
    def tokenize_function(examples):
        return tokenizer(examples['text'])

    raw_dataset = load_dataset(DATASET_NAME, split="train")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=raw_dataset.column_names
    )

    # 토큰화된 모든 텍스트를 하나로 합칩니다.
    main_column = "input_ids"
    concatenated_ids = [id for ids in tokenized_dataset[main_column] for id in ids]
    total_length = len(concatenated_ids)
    # EOS 토큰을 추가할 필요가 없습니다. 청크가 자연스럽게 문장의 끝을 학습합니다.
    
    # 긴 시퀀스를 max_seq_length 길이의 청크로 나눕니다.
    total_length = (total_length // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
    # 마지막에 남는 짧은 시퀀스는 버립니다.
    result = {
        "input_ids": [concatenated_ids[i: i + MAX_SEQ_LENGTH] for i in range(0, total_length, MAX_SEQ_LENGTH)],
        "labels": [concatenated_ids[i: i + MAX_SEQ_LENGTH] for i in range(0, total_length, MAX_SEQ_LENGTH)],
    }
    
    from datasets import Dataset
    processed_dataset = Dataset.from_dict(result)

    # --- 4. Trainer 설정 ---
    # DataCollatorForLanguageModeling은 Causal LM을 위해 자동으로 라벨을 생성합니다.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
        # 옵티마이저와 스케줄러는 TrainingArguments에 따라 자동으로 생성됩니다.
    )

    # --- 5. 학습 시작 ---
    trainer.train()

if __name__ == "__main__":
    main()