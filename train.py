import os
import math
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, DatasetDict

# 🚀 데이터 로드
print("데이터셋 로드 중 ('minpeter/pretrain-korean-dedup')...")
full_ds = load_dataset('minpeter/pretrain-korean-dedup', split='train')
# 실제 학습 시에는 test_size를 적절히 조절 (예: 0.001 ~ 0.01)
ds_split = full_ds.train_test_split(test_size=0.01, shuffle=True, seed=5768112)
ds = DatasetDict({
    'train': ds_split['train'],
    'test': ds_split['test']
})
print("원본 데이터셋 구조:")
print(ds)
print(f"학습 데이터 샘플 수: {len(ds['train'])}")
print(f"테스트 데이터 샘플 수: {len(ds['test'])}")

max_cpu_core = int(os.cpu_count() / 4) if os.cpu_count() else 1 # cpu_count가 None일 경우 대비

# 🚀 토크나이저 로드 및 EOS/PAD 토큰 설정
context_length = 2048
tokenizer_name = "kakaocorp/kanana-nano-2.1b-base"
print(f"토크나이저 로드 중 ('{tokenizer_name}')...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print(f"원본 토크나이저 어휘 크기 (vocab_size): {len(tokenizer)}")

if tokenizer.eos_token is None:
    print("⚠️ 경고: 토크나이저에 기본 EOS 토큰이 없습니다. '</s>'로 설정합니다.")
    tokenizer.add_special_tokens({'eos_token': '</s>'})

if tokenizer.pad_token is None:
    print("토크나이저에 PAD 토큰이 정의되어 있지 않아 EOS 토큰을 PAD 토큰으로 사용합니다.")
    tokenizer.pad_token = tokenizer.eos_token

print(f"사용될 EOS 토큰: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
print(f"사용될 BOS 토큰: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
print(f"사용될 PAD 토큰: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
print(f"최종 토크나이저 어휘 크기 (vocab_size): {len(tokenizer)}")


# 🚀 각 문서 끝에 EOS 토큰 추가
def append_eos_to_text(examples):
    processed_texts = [text + tokenizer.eos_token for text in examples['text']]
    examples['text'] = processed_texts
    return examples

print("\n각 문서에 EOS 토큰 추가 중...")
ds_with_eos = ds.map(
    append_eos_to_text,
    batched=True,
    batch_size=5_000,
    num_proc=max_cpu_core
)
print("EOS 추가 후 데이터셋 샘플 (text 필드만):")
if len(ds_with_eos["train"]) > 0:
    print(ds_with_eos["train"][0]['text'][-100:])
else:
    print("학습 데이터셋이 비어있어 EOS 추가 샘플을 확인할 수 없습니다.")


# 🚀 초기 토큰화 (텍스트 -> 토큰 ID 리스트)
def tokenize_for_grouping(element):
    return tokenizer(element["text"], truncation=False, return_attention_mask=False)

print("\n초기 토큰화 진행 중...")
tokenized_dataset_for_grouping = ds_with_eos.map(
    tokenize_for_grouping,
    batched=True,
    batch_size=5_000,
    num_proc=max_cpu_core,
    remove_columns=ds_with_eos["train"].column_names
)

# 🚀 텍스트 그룹화 (Concatenate and Chunk)
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= context_length:
        total_length = (total_length // context_length) * context_length
    
    result = {
        k: [t[i : i + context_length] for i in range(0, total_length, context_length)]
        for k, t in concatenated_examples.items()
    }
    return result

print("\n토큰화된 텍스트 그룹화 중...")
lm_datasets = tokenized_dataset_for_grouping.map(
    group_texts,
    batched=True,
    batch_size=500, # 그룹화 시 배치 크기 (메모리 사용량에 따라 조절)
    num_proc=max_cpu_core,
)
print("그룹화된 데이터셋 구조:")
print(lm_datasets)
if len(lm_datasets["train"]) > 0:
    print("그룹화된 학습 데이터셋 첫 샘플 input_ids 길이:", len(lm_datasets["train"][0]['input_ids']))
    print(f"그룹화 후 학습 샘플 수: {len(lm_datasets['train'])}")
    print(f"그룹화 후 테스트 샘플 수: {len(lm_datasets['test'])}")
else:
    print("그룹화된 학습 데이터셋이 비어 있습니다. context_length 또는 데이터 양을 확인하세요.")


# 🚀 모델 초기화
model_config_name = "HuggingFaceTB/SmolLM2-135M"
print(f"모델 설정 로드 중 ('{model_config_name}')...")
config = AutoConfig.from_pretrained(
    model_config_name,
    vocab_size=len(tokenizer),
    max_position_embeddings=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

print(f"LlamaForCausalLM 모델 초기화 중 (vocab_size={config.vocab_size}, max_pos_emb={config.max_position_embeddings})...")
model = LlamaForCausalLM(config)

model_size = sum(t.numel() for t in model.parameters())
print(f"\n모델 크기: {model_size/1000**2:.1f}M parameters")


# 🚀 평가 지표 준비
accuracy_metric = evaluate.load("accuracy")

def preprocess_logits_for_metrics(logits, _):
    if isinstance(logits, tuple): # 모델 출력에서 logits 추출
        return logits[0]
    return logits

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    
    # Perplexity 계산
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # DataCollatorForLanguageModeling은 레이블의 패딩 부분을 -100으로 채우므로, ignore_index를 -100으로 설정
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = math.exp(loss.item())

    # Accuracy 계산
    predictions = torch.argmax(logits, axis=-1)
    labels_for_acc = labels[:, 1:].reshape(-1) # labels에서 첫번째 토큰 제외
    preds_for_acc = predictions[:, :-1].reshape(-1) # predictions에서 마지막 토큰 제외
    
    # DataCollatorForLanguageModeling이 패딩을 -100으로 처리하므로, 해당 값을 마스크 처리
    mask = labels_for_acc != -100

    valid_labels = labels_for_acc[mask]
    valid_preds = preds_for_acc[mask]
    accuracy = accuracy_metric.compute(predictions=valid_preds, references=valid_labels)["accuracy"]
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "eval_loss": loss.item() # eval_loss를 추가하여 Trainer 로그와 일관성 유지
    }


# 🚀 데이터 콜레이터
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
print(f"데이터 콜레이터의 pad_token_id는 {tokenizer.pad_token_id}이지만, 레이블에서는 -100으로 패딩 처리됩니다.")


# 🚀 TrainingArguments
output_dir_name = "model/pretrained-tiny-ko-eos"
hub_id = "minpeter/pretrain-tiny-ko-eos" # Hugging Face Hub ID

# 학습 전 토크나이저 저장 및 Hub 업로드
tokenizer.save_pretrained(output_dir_name)
try:
    tokenizer.push_to_hub(hub_id)
    print(f"토크나이저가 '{hub_id}' Hub에 업로드되었습니다.")
except Exception as e:
    print(f"토크나이저 Hub 업로드 실패: {e}. 로컬에만 저장됩니다: '{output_dir_name}'")


args = TrainingArguments(
    output_dir=output_dir_name,

    push_to_hub=True,
    hub_model_id=hub_id,
    hub_strategy="every_save",

    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    save_steps=1000,

    gradient_accumulation_steps=4,
    per_device_train_batch_size=8, # GPU 메모리에 맞춰 조정
    per_device_eval_batch_size=2,  # GPU 메모리에 맞춰 조정

    logging_steps=50, # 로그 빈도 (기존 5에서 늘림, 너무 잦은 로그 방지)
    num_train_epochs=2,
    weight_decay=0.1,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    
    # BF16/FP16 설정: 사용 가능한 경우 BF16 우선, 아니면 FP16
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    fp16=torch.cuda.is_available() and not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),

    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_perplexity", # perplexity가 낮은 모델을 최선으로 선택
    greater_is_better=False,

    # torch_compile=True, # PyTorch 2.0+ 에서 성능 향상을 위해 사용 가능 (첫 스텝 느릴 수 있음)
    report_to="wandb", # "none", "wandb", "tensorboard" 등
    # deepspeed="ds_config.json", # DeepSpeed 사용 시 설정 파일 경로
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=lm_datasets["train"] if lm_datasets and len(lm_datasets["train"]) > 0 else None,
    eval_dataset=lm_datasets["test"] if lm_datasets and len(lm_datasets["test"]) > 0 else None,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

print("\n훈련 시작 준비 완료!")

if trainer.train_dataset and trainer.eval_dataset:
    num_devices = max(1, args.world_size) # 분산 학습 고려
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * num_devices
    total_train_samples = len(lm_datasets['train'])
    steps_per_epoch = math.ceil(total_train_samples / effective_batch_size)
    total_steps = steps_per_epoch * args.num_train_epochs
    print(f"총 학습 스텝 수 예상: {total_steps} (에포크 당 {steps_per_epoch} 스텝)")
    
    print("학습을 시작합니다...")
    trainer.train()
    print("학습 완료!")

    # 학습 완료 후 모델과 토크나이저를 Hub에 푸시 (load_best_model_at_end=True 이므로 최적 모델)
    try:
        trainer.push_to_hub()
        print("최종 모델 및 토크나이저가 Hub에 성공적으로 업로드되었습니다.")
    except Exception as e:
        print(f"최종 모델 Hub 업로드 실패: {e}")

else:
    print("학습 또는 평가 데이터셋이 준비되지 않아 훈련을 시작할 수 없습니다.")
    print("데이터셋 로드, 토큰화, 그룹화 과정을 확인해주세요.")
    if lm_datasets:
        print(f"학습 데이터 샘플 수: {len(lm_datasets.get('train', []))}")
        print(f"평가 데이터 샘플 수: {len(lm_datasets.get('test', []))}")