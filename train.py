# RUN COMMAND: time uv run accelerate launch train.py

import os
import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk

# --- 설정 ---
# preprocess.py에서 저장한 데이터셋 경로
PROCESSED_DATA_PATH = "./processed_data"
TOKENIZER_PATH = "./tknz/tiny-ko-tokenizer"
CONTEXT_LENGTH = 2048
HF_MODEL_ID = "minpeter/tiny-ko-124m-base"
LOCAL_MODEL_PATH = "model/tiny-ko-124m-base"
# ----------------

# 1. 전처리 완료된 데이터셋을 디스크에서 바로 로드
print(f"사전 처리된 데이터셋을 '{PROCESSED_DATA_PATH}'에서 로드합니다.")
tokenized_dataset = load_from_disk(PROCESSED_DATA_PATH)

print("\n로딩 완료된 데이터셋 구조:")
print(tokenized_dataset)
print(f"훈련 샘플 수: {len(tokenized_dataset['train'])}")
print(f"테스트 샘플 수: {len(tokenized_dataset['test'])}")
print(f"샘플 0의 토큰 수: {len(tokenized_dataset['train'][0]['input_ids'])}")

# 2. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
# tokenizer.model_max_length = CONTEXT_LENGTH

try:
    print(f"\n사용될 EOS 토큰: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print(f"사용될 PAD 토큰: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    print(f"사용될 BOS 토큰: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
except AttributeError as e:
    print(e)

# 3. 모델 구성 (Config)
config = LlamaConfig(
    hidden_size=480,
    num_hidden_layers=32,
    intermediate_size=1920,
    tie_word_embeddings=True,
    num_attention_heads=6,
    num_key_value_heads=2,
    vocab_size=len(tokenizer),
    max_position_embeddings=CONTEXT_LENGTH,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    rope_theta=10000.0,
    use_cache=False,
    attn_implementation="flash_attention_2",
)

# 4. 모델 초기화
config._attn_implementation = "flash_attention_2"
model = LlamaForCausalLM(config)
model = model.to(torch.bfloat16)

model_size = sum(t.numel() for t in model.parameters())
print(f"\n모델 크기: {model_size/1000**3:.2f}B parameters")

# 5. 데이터 콜레이터, 토크나이저 저장 및 푸시
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
tokenizer.save_pretrained(LOCAL_MODEL_PATH)
tokenizer.push_to_hub(HF_MODEL_ID)

# 6. 학습 인자 (TrainingArguments) 설정
max_cpu_count = int(os.cpu_count() / 3) or 1
args = TrainingArguments(
    output_dir=LOCAL_MODEL_PATH,
    push_to_hub=True, # 필요시 주석 해제
    hub_model_id=HF_MODEL_ID,
    hub_strategy="every_save",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1_000,
    save_steps=1_000,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=39,
    per_device_eval_batch_size=39,
    logging_steps=25,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    learning_rate=6e-4,
    optim="adamw_torch_fused",
    dataloader_pin_memory=True,
    bf16=True,
    torch_compile=True,
    dataloader_num_workers=max_cpu_count,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 7. 트레이너(Trainer) 초기화 및 학습 시작
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()