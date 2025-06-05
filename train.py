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
print("데이터셋 로드 중 ('minpeter/pretrain-korean-dedup')... 시간이 걸릴 수 있습니다.")
full_ds = load_dataset('minpeter/pretrain-korean-dedup', split='train')
# train_test_split을 사용하여 데이터셋 분할 (예: 99% train, 1% test)
# 실제 학습 시에는 test_size를 적절히 조절하세요 (예: 0.001 ~ 0.01)
ds_split = full_ds.train_test_split(test_size=0.01, shuffle=True, seed=5768112) # 0.5% for test
ds = DatasetDict({
    'train': ds_split['train'],
    'test': ds_split['test']
})
print("원본 데이터셋 구조:")
print(ds)
print(f"학습 데이터 샘플 수: {len(ds['train'])}")
print(f"테스트 데이터 샘플 수: {len(ds['test'])}")


# 🚀 토크나이저 로드 및 EOS/PAD 토큰 확인 및 설정
context_length = 2048 # 원래 context_length로 복원
tokenizer_name = "kakaocorp/kanana-nano-2.1b-base" # 원래 토크나이저로 복원
print(f"토크나이저 로드 중 ('{tokenizer_name}')...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.chat_template = None # reset chat template

print(f"원본 토크나이저 어휘 크기 (vocab_size): {len(tokenizer)}")

# 1. EOS 토큰 설정
if tokenizer.eos_token is None:
    print("⚠️ 경고: 토크나이저에 기본 EOS 토큰이 없습니다. '</s>'로 설정합니다.")
    tokenizer.add_special_tokens({'eos_token': '</s>'})

# 2. PAD 토큰 설정
if tokenizer.pad_token is None:
    print("토크나이저에 PAD 토큰이 정의되어 있지 않아 EOS 토큰을 PAD 토큰으로 사용합니다.")
    tokenizer.pad_token = tokenizer.eos_token

print(f"사용될 EOS 토큰: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
print(f"사용될 BOS 토큰: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
print(f"사용될 PAD 토큰: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
print(f"최종 토크나이저 어휘 크기 (vocab_size): {len(tokenizer)}")


# 🚀 각 문서 끝에 EOS 토큰 추가하는 함수
def append_eos_to_text(examples):
    processed_texts = [text + tokenizer.eos_token for text in examples['text']]
    examples['text'] = processed_texts
    return examples

print("\n각 문서에 EOS 토큰 추가 중...")
ds_with_eos = ds.map(
    append_eos_to_text,
    batched=True,
    batch_size=5_000, # 실제 데이터셋에 맞는 배치 크기
    num_proc=os.cpu_count()
)
print("EOS 추가 후 데이터셋 샘플 (text 필드만):")
print(ds_with_eos["train"][0]['text'][-100:])


# 🚀 개선된 토큰화 함수 (초기 토큰화)
def tokenize_for_grouping(element):
    return tokenizer(element["text"], truncation=False, return_attention_mask=False)

print("\n초기 토큰화 진행 중 (텍스트 -> 토큰 ID 리스트)...")
tokenized_dataset_for_grouping = ds_with_eos.map(
    tokenize_for_grouping,
    batched=True,
    batch_size=5_000, # 실제 데이터셋에 맞는 배치 크기
    num_proc=os.cpu_count(),
    remove_columns=ds_with_eos["train"].column_names
)

# 🚀 텍스트 그룹화 함수 (Concatenate and Chunk)
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

print("\n토큰화된 텍스트 그룹화 중 (Concatenate and Chunk)...")
lm_datasets = tokenized_dataset_for_grouping.map(
    group_texts,
    batched=True,
    batch_size=500, # 그룹화 시 배치 크기 (메모리 사용량에 따라 조절)
    num_proc=os.cpu_count(),
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
model_config_name = "HuggingFaceTB/SmolLM2-135M" # 원래 모델 config로 복원
print(f"모델 설정 로드 중 ('{model_config_name}')...")
config = AutoConfig.from_pretrained(
    model_config_name,
    vocab_size=len(tokenizer),
    max_position_embeddings=context_length, # Llama 계열은 이 필드 사용
    # n_positions=context_length, # 일부 모델은 이 필드 사용
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    # LlamaForCausalLM은 config의 다른 아키텍처 파라미터 (예: hidden_size, num_layers)를
    # SmolLM2-135M의 값으로 사용하거나, 없다면 기본값을 사용할 수 있습니다.
    # 만약 아키텍처 불일치 오류 발생 시, 해당 파라미터들을 수동으로 config에 명시해야 할 수 있습니다.
)

print(f"LlamaForCausalLM 모델 초기화 중 (vocab_size={config.vocab_size}, max_pos_emb={config.max_position_embeddings})...")
model = LlamaForCausalLM(config)

model_size = sum(t.numel() for t in model.parameters())
print(f"\n모델 크기: {model_size/1000**2:.1f}M parameters")


# 🚀 평가 지표 준비
accuracy_metric = evaluate.load("accuracy")

def preprocess_logits_for_metrics(logits, _):
    # 'labels' 파라미터가 사용되지 않으므로 '_'로 변경
    if isinstance(logits, tuple): # 모델 출력에서 logits 추출
        return logits[0]
    return logits

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    
    # Perplexity 계산
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = math.exp(loss.item())

    # Accuracy 계산
    predictions = torch.argmax(logits, axis=-1)
    labels_for_acc = labels[:, 1:].reshape(-1)
    preds_for_acc = predictions[:, :-1].reshape(-1)
    mask = labels_for_acc != tokenizer.pad_token_id # DataCollator가 pad_token_id를 -100으로 바꿀수도 있음. 일관성 확인.
                                                    # Trainer는 기본적으로 label의 -100을 무시합니다.
                                                    # DataCollatorForLanguageModeling의 기본 ignore_index가 -100 이므로,
                                                    # tokenizer.pad_token_id 대신 -100을 사용하는 것이 더 일반적일 수 있습니다.
                                                    # 여기서는 tokenizer.pad_token_id로 설정된 값을 기준으로 합니다.
                                                    # 만약 pad_token_id가 -100이 아니라면, loss_fct의 ignore_index도 맞춰야합니다.

    # DataCollatorForLanguageModeling은 기본적으로 labels의 padding 부분을 -100으로 채움.
    # 따라서 mask는 labels_for_acc != -100 이 더 안전할 수 있음.
    actual_mask_value = -100 # HF Trainer/DataCollator 표준
    mask = labels_for_acc != actual_mask_value

    valid_labels = labels_for_acc[mask]
    valid_preds = preds_for_acc[mask]
    accuracy = accuracy_metric.compute(predictions=valid_preds, references=valid_labels)["accuracy"]
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "eval_loss": loss.item()
    }


# 🚀 데이터 콜레이터
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
print(f"데이터 콜레이터 pad_token_id: {tokenizer.pad_token_id} (실제로는 -100으로 채워질 수 있음)")


# 🚀 TrainingArguments
output_dir_name = "model/pretrained-tiny-ko-eos" # 실제 프로젝트에 맞게 수정
hub_id = "minpeter/pretrain-tiny-ko-eos" # 실제 Hugging Face Hub ID로 변경

# 학습 전 토크나이저 저장
tokenizer.save_pretrained(output_dir_name)
tokenizer.push_to_hub(hub_id) # 실제 Hub에 업로드 시 활성화
print(f"토크나이저가 '{output_dir_name}'에 저장되었습니다.")

args = TrainingArguments(
    output_dir=output_dir_name,

    push_to_hub=True, # 실제 Hub에 업로드 시 활성화
    hub_model_id=hub_id, # 실제 Hub에 업로드 시 활성화
    hub_strategy="every_save",

    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,       # 원래 값으로 복원
    save_steps=1000,       # 원래 값으로 복원

    gradient_accumulation_steps=4, # 원래 값으로 복원
    per_device_train_batch_size=56, # GPU 메모리에 맞춰 조정 (원래 56은 매우 클 수 있음, A100 40GB 기준 8~16 적절)
    per_device_eval_batch_size=56,  # GPU 메모리에 맞춰 조정

    logging_steps=5, # 로그 빈도 조정
    num_train_epochs=2,    # 원래 값으로 복원 (또는 필요에 따라 조절)
    weight_decay=0.1,
    warmup_steps=200,     # 원래 값으로 복원
    lr_scheduler_type="cosine",
    learning_rate=1e-4,    # 원래 값으로 복원
    
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, # Ampere 이상에서 BF16 사용
    fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) and torch.cuda.is_available(), # BF16 사용 불가 시 FP16

    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_perplexity",
    greater_is_better=False,

    # torch_compile=True, # PyTorch 2.0+ 및 호환 환경에서 사용 시 주석 해제 (첫 스텝이 느릴 수 있음)
    # report_to="tensorboard",
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
    print(f"총 학습 스텝 수 예상: { (len(lm_datasets['train']) // (args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, args.world_size))) * args.num_train_epochs }")
    # print("실제 학습을 시작하려면 아래 trainer.train() 주석을 해제하세요.")
    trainer.train()
    # print("실제 학습은 주석 처리되어 있습니다.")
else:
    print("학습 또는 평가 데이터셋이 준비되지 않아 훈련을 시작할 수 없습니다.")
    print("데이터셋 로드, 토큰화, 그룹화 과정을 확인해주세요.")
    if lm_datasets:
        print(f"학습 데이터 샘플 수: {len(lm_datasets.get('train', []))}")
        print(f"평가 데이터 샘플 수: {len(lm_datasets.get('test', []))}")