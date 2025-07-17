from itertools import chain
import os
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithFlattening
from datasets import Dataset
import pprint


max_seq_length = 10


def get_attention_mask_for_packed_sequence(x, token_id, eos: bool = True):
    """EOS 토큰을 기반으로 sequence boundary를 고려한 attention mask 생성"""
    B, T = x.shape
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = torch.cat([eos_idx, torch.arange(0,B*T+1,T)]).unique().sort()[0]
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)
    repeated_idx = torch.repeat_interleave(normalized_idx[1:], reps).view(B,1,T).expand(-1,T,-1)
    mask_indices = torch.arange(T).view(1,-1,1).expand(B, -1, T)
    mask = torch.ones(T, T, dtype=torch.bool).tril().expand(B, -1, -1)
    mask = mask.masked_fill(mask_indices >= repeated_idx, False)
    return mask


def get_position_ids_for_packed_sequence(x, token_id, eos: bool = True):
    """EOS 토큰을 기반으로 sequence boundary를 고려한 position_ids 생성"""
    B, T = x.shape
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = torch.cat([eos_idx, torch.arange(0,B*T+1,T)]).unique().sort()[0]
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)
    pos_ids = (torch.arange(B*T) - torch.repeat_interleave(eos_idx_expanded[:-1], reps)).view(B,T)
    return pos_ids


class CustomDataCollator:
    def __init__(self, tokenizer, eos_token_id):
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
    
    def __call__(self, features):
        batch = {}
        
        # input_ids와 labels 처리
        input_ids = torch.tensor([f["input_ids"] for f in features])
        labels = torch.tensor([f["labels"] for f in features])
        
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        
        # EOS 토큰 기반으로 attention mask 생성
        attention_mask = get_attention_mask_for_packed_sequence(input_ids, self.eos_token_id, eos=True)
        batch["attention_mask"] = attention_mask
        
        # EOS 토큰 기반으로 position_ids 생성
        position_ids = get_position_ids_for_packed_sequence(input_ids, self.eos_token_id, eos=True)
        batch["position_ids"] = position_ids
        
        return batch


def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding=False, truncation=False)

    if tokenizer.eos_token_id is not None:
        # batched=False일 때는 단일 샘플이므로 직접 처리
        if isinstance(tokenized_inputs["input_ids"][0], list):
            # batched=True인 경우
            for i in range(len(tokenized_inputs["input_ids"])):
                tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
                tokenized_inputs["attention_mask"][i].append(1)
                if "token_type_ids" in tokenized_inputs:
                    tokenized_inputs["token_type_ids"][i].append(0)
        else:
            # batched=False인 경우 (단일 샘플)
            tokenized_inputs["input_ids"].append(tokenizer.eos_token_id)
            tokenized_inputs["attention_mask"].append(1)
            if "token_type_ids" in tokenized_inputs:
                tokenized_inputs["token_type_ids"].append(0)

    return tokenized_inputs

def group_texts(examples):
    block_size=max_seq_length

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


tokenizer = AutoTokenizer.from_pretrained(
    "./tknz/tiny-ko-tokenizer"
)

raw_datasets = Dataset.from_dict({
    'text': ["This is a test", "Here is another one.Here is another one.", "And yet another test sentence.And yet another test sentence."]
})

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    # batched=True,
    # num_proc=os.cpu_count(),
    remove_columns=["text"],
)

print("\n토큰화된 데이터셋:")
for i, sample in enumerate(tokenized_datasets):
    print(f"샘플 {i}: {len(sample['input_ids'])} 토큰 - {sample['input_ids']}")

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    # num_proc=os.cpu_count(),
)

print("\n로딩 완료된 데이터셋 구조:")
print(lm_datasets)
print(f"훈련 샘플 수: {len(lm_datasets)}")
print(f"샘플 0의 토큰 수: {len(lm_datasets[0]['input_ids'])}")
print(lm_datasets[0])



data_collator = CustomDataCollator(tokenizer, tokenizer.eos_token_id)
# data_collator = DataCollatorWithFlattening()
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Collated batch example:")
eos_id = tokenizer.eos_token_id
found = False
for idx, sample in enumerate(lm_datasets):
    print(f"\n=== 샘플 {idx} ===")
    print("원본 샘플:", sample)
    batch = data_collator([sample])
    print("Collated batch:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
            print(value)
        else:
            print(f"{key}: {value}")
    
    if eos_id in sample["input_ids"]:
        eos_positions = [i for i, token_id in enumerate(sample["input_ids"]) if token_id == eos_id]
        print(f"EOS 토큰(ID: {eos_id}) 위치: {eos_positions}")
        
        # attention mask 시각화
        if "attention_mask" in batch and len(batch["attention_mask"].shape) == 3:
            print("Attention Mask:")
            mask = batch["attention_mask"][0]
            print(mask.int())
        
        found = True
        
if not found:
    print("EOS 토큰이 포함된 샘플을 찾지 못했습니다.")