import torch
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from datasets import Dataset
from trl import pack_dataset
import pprint

# 1. 토크나이저 및 데이터 준비
# 사용자의 토크나이저 경로를 사용합니다.
tokenizer = AutoTokenizer.from_pretrained("./tknz/tiny-ko-tokenizer")
# pad_token이 없을 경우 eos_token으로 설정 (CLM에서 일반적인 처리)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 실제 한국어 텍스트 데이터
text_examples = [
    "안녕하세요, 만나서 반갑습니다, 저는",
    "이것은 TRL의 pack_dataset 함수 테스트 예제입니다.",
    "짧은 문장.",
    "그리고 조금 더 긴 문장입니다."
]


print("--- [단계 1] 원본 텍스트 데이터 ---")
pprint.pprint(text_examples)

# 2. 데이터 토큰화 및 Dataset 생성
tokenized_dataset = Dataset.from_dict(tokenizer(text_examples))

# 원본 샘플별 길이 출력
print("\n--- [단계 1] 토큰화된 데이터셋 결과 ---")
pprint.pprint(tokenized_dataset[:])

# 각 샘플의 길이 확인
print("\n--- [단계 1] 각 샘플의 길이 ---")
lengths = [len(sample['input_ids']) for sample in tokenized_dataset]
print(lengths)



# 3. 데이터셋 패킹 (seq_length=20으로 설정)
# 짧은 문장들을 합쳐서 길이 20의 시퀀스로 만듭니다.
packed_dataset = pack_dataset(tokenized_dataset, seq_length=5, strategy="wrapped")

print("\n--- [단계 2] 패킹된 데이터셋 결과 ---")
# pack_dataset은 Dataset을 반환하며, 각 샘플의 길이는 seq_length와 같거나 더 짧습니다.
pprint.pprint(packed_dataset[:])

# 4. 데이터 콜레이터 준비 및 적용
# mlm=False는 Causal Language Modeling (다음 단어 예측)을 의미합니다.
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 패킹된 데이터셋 전체를 하나의 배치로 만듭니다.
# 실제 훈련 시에는 DataLoader가 이 역할을 수행합니다.
batch = [sample for sample in packed_dataset]

# 콜레이터를 배치에 적용합니다.
collated_batch = collator(batch)

print("\n--- [단계 3] 최종 콜레이팅된 배치 결과 ---")
# PyTorch 텐서를 확인하기 쉽게 리스트로 변환하여 출력합니다.
final_result = {k: v.tolist() for k, v in collated_batch.items()}
pprint.pprint(final_result)

print("\n--- 'labels' 상세 비교 ---")
print(f"input_ids[0]: {final_result['input_ids'][0]}")
print(f"labels[0]   : {final_result['labels'][0]}")
print("\n'labels'는 'input_ids'를 왼쪽으로 한 칸씩 민(shift) 값이며, 예측할 필요가 없는 위치는 -100으로 마스킹됩니다.")