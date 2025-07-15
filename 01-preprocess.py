import os
import time
import itertools
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer

# --- 설정 ---
CONTEXT_LENGTH = 2048
TOKENIZER_PATH = "./tknz/tiny-ko-tokenizer"
SAVE_PATH = "./processed_data"
# ----------------

class ConstantLengthPacker:
    """
    datasets.map을 사용하여 상태를 유지하며 초고속으로 데이터셋을 패킹하는 클래스.
    """
    def __init__(self, context_length=2048):
        self.context_length = context_length
        # 이전 배치에서 처리하고 남은 토큰들을 저장하는 버퍼
        self.buffer = []

    def __call__(self, batch):
        # 1. 현재 배치의 모든 input_ids를 하나의 리스트로 평탄화하고, 이전 버퍼와 합칩니다.
        # itertools.chain.from_iterable은 중첩 리스트를 빠르게 평탄화합니다.
        concatenated_ids = self.buffer + list(itertools.chain.from_iterable(batch['input_ids']))

        # 2. 합쳐진 리스트에서 만들 수 있는 전체 청크(chunk)의 개수를 계산합니다.
        total_length = len(concatenated_ids)
        num_chunks = total_length // self.context_length

        # 3. 온전한 청크들을 잘라냅니다.
        # NumPy를 사용하여 매우 빠르게 배열을 만들고 형태를 바꿉니다.
        chunked_ids = np.array(
            concatenated_ids[:num_chunks * self.context_length]
        ).reshape(-1, self.context_length)

        # 4. 다음 배치를 위해 처리하지 못하고 남은 토큰들을 버퍼에 저장합니다.
        self.buffer = concatenated_ids[num_chunks * self.context_length:]

        # 5. map 함수가 요구하는 딕셔너리 형태로 반환합니다.
        return {"input_ids": chunked_ids.tolist()}

def pack_and_save_dataset_fast(tokenized_ds, context_length, save_path):
    """
    ConstantLengthPacker와 map을 사용하여 데이터셋을 초고속으로 패킹하고 저장합니다.
    """
    print("\n데이터셋 패킹 시작 (map 기반 고속 처리)...")
    
    # train 데이터셋 패킹
    train_packer = ConstantLengthPacker(context_length)
    train_packed = tokenized_ds["train"].map(
        train_packer,
        batched=True,
        batch_size=100_000, # 한 번에 처리할 문서의 수. 메모리 상황에 맞게 조절 가능.
        num_proc=1, # 상태를 유지해야 하므로 단일 프로세스로 실행. 그럼에도 훨씬 빠릅니다.
        remove_columns=tokenized_ds["train"].column_names
    )
    # 패킹 후 남은 버퍼는 폐기됩니다.

    # test 데이터셋 패킹
    test_packer = ConstantLengthPacker(context_length)
    test_packed = tokenized_ds["test"].map(
        test_packer,
        batched=True,
        batch_size=100_000,
        num_proc=1,
        remove_columns=tokenized_ds["test"].column_names
    )
    
    packed_dataset_dict = DatasetDict({"train": train_packed, "test": test_packed})
    
    print("\n패킹 완료된 데이터셋 구조:")
    print(packed_dataset_dict)
    print(f"훈련 샘플 수: {len(packed_dataset_dict['train'])}")
    print(f"테스트 샘플 수: {len(packed_dataset_dict['test'])}")
    
    print(f"\n처리된 데이터셋을 '{save_path}' 경로에 저장합니다...")
    packed_dataset_dict.save_to_disk(save_path)
    print("저장 완료!")


# ======================================================================
# 아래 main 함수 부분은 이전과 동일하게 두되, 호출하는 함수만 변경합니다.
# ======================================================================

def setup_directories():
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    os.makedirs(SAVE_PATH, exist_ok=True)

def load_raw_datasets():
    print("원본 데이터셋 로딩 중...")
    ds_kr = load_dataset("minpeter/tiny-ko-corpus", split="train[:2000]")
    cosmopedia = load_dataset("HuggingFaceTB/smollm-corpus", data_files=[f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(21)], split="train[:2000]")
    fineweb = load_dataset("HuggingFaceTB/smollm-corpus", data_files=[f"fineweb-edu-dedup/train-{i:05d}-of-00234.parquet" for i in range(21)], split="train[:2000]")
    cosmopedia_text = cosmopedia.remove_columns([col for col in cosmopedia.column_names if col != "text"])
    fineweb_text = fineweb.remove_columns([col for col in fineweb.column_names if col != "text"])
    ds_en = concatenate_datasets([cosmopedia_text, fineweb_text])
    ds = concatenate_datasets([ds_kr, ds_en])
    return ds.train_test_split(test_size=0.001, shuffle=True, seed=5768112)

def tokenize_dataset(ds, tokenizer):
    print("\n토큰화 진행 중...")
    num_proc = max(1, os.cpu_count() - 2)
    def tokenize_with_eos(examples):
        return tokenizer([text + tokenizer.eos_token for text in examples["text"]], truncation=False, padding=False)
    return ds.map(tokenize_with_eos, batched=True, num_proc=num_proc, remove_columns=ds["train"].column_names)


if __name__ == "__main__":
    total_start_time = time.time()
    setup_directories()

    # 1. 로딩
    raw_ds = load_raw_datasets()
    
    # 2. 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # 3. 토큰화
    tokenized_ds = tokenize_dataset(raw_ds, tokenizer)
    
    # 4. 패킹 및 저장 (새로운 고속 함수 호출)
    pack_and_save_dataset_fast(tokenized_ds, CONTEXT_LENGTH, SAVE_PATH)
    
    print(f"\n총 소요 시간: {(time.time() - total_start_time)/60:.2f} 분")