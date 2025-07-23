import os
import torch
from typing import Dict, Any, List
from functools import partial # functools.partial 임포트

from datatrove.data import Document, DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from datatrove.utils.batching import batched
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# torch memory cleanup
torch.cuda.empty_cache()

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print("TF32 사용 가능. 행렬 곱 정밀도 설정 중...")
    torch.set_float32_matmul_precision('high')

# 1. 분류기 블록 (기존과 동일)
class EduScoreClassifier(PipelineStep):
    """
    'devngho/ko_edu_classifier_v2_nlpai-lab_KoE5' 모델을 사용하여
    텍스트의 교육적 가치 점수를 계산합니다.
    모델의 원시 logit 출력을 직접 사용하여 정확한 회귀 점수를 얻습니다.
    """
    type = "🧑‍🏫 - CLASSIFIER"
    name = "Ko-EDU Score Classifier"

    def __init__(
        self,
        model_name: str = "devngho/ko_edu_classifier_v2_nlpai-lab_KoE5",
        batch_size: int = 32,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = None

    @property
    def tokenizer(self) -> AutoTokenizer:
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self) -> AutoModelForSequenceClassification:
        if not self._model:
            print(f"'{self.model_name}' 모델을 로딩합니다. (Device: {self.device.upper()})")


            self.torch_dtype = torch.float32 # 기본값
            if "cuda" in self.device:
                # Ampere 아키텍처 이상 (A100, RTX 30xx/40xx 등)인지 확인
                if torch.cuda.get_device_capability(self.device)[0] >= 8:
                    self.torch_dtype = torch.bfloat16
                    print("bfloat16 지원 확인. 모델을 bfloat16으로 로드합니다.")
                else:
                    self.torch_dtype = torch.float16
                    print("bfloat16 미지원. 모델을 float16 (AMP)으로 로드합니다.")

            print(f"'{self.model_name}' 모델을 로딩합니다. (Device: {self.device.upper()}, DType: {self.torch_dtype})")


            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype
            ).to(self.device)

            print(f"'{self.model_name}' 모델 컴파일 시작...")
            self._model = torch.compile(self._model)
            print(f"'{self.model_name}' 모델 컴파일 완료.")

        return self._model

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for batch in batched(data, self.batch_size):
            # 모델 입력 형식에 맞게 "passage: " 접두어 추가
            texts = ["passage: " + doc.text for doc in batch]

            try:
                with self.track_time("batch"), torch.no_grad():
                    # 텍스트를 토크나이징하고 모델이 있는 디바이스로 이동
                    inputs = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    # 모델 추론 실행
                    outputs = self.model(**inputs)

                    # Logits에서 직접 점수 추출 (소프트맥스 미적용)
                    # .squeeze(-1)을 통해 [batch_size, 1] -> [batch_size] 형태로 변환
                    scores = outputs.logits.squeeze(-1).float().cpu().numpy()

                for i, doc in enumerate(batch):
                    raw_score = scores[i].item() # numpy float -> python float 변환

                    doc.metadata["edu_score_raw"] = round(raw_score, 4)
                    doc.metadata["edu_score"] = int(raw_score)

                    self.stat_update(f"score_{int(raw_score)}")

                yield from batch

            except Exception as e:
                self.stat_update("classification_error_batch")
                print(f"배치 분류 중 오류 발생: {e}")
                for doc in batch:
                    doc.metadata["edu_score"] = -1
                    doc.metadata["edu_score_raw"] = -1.0
                    yield doc

# 2. 여러 데이터 소스를 로드하기 위한 제너레이터 함수 (수정됨)
def load_multiple_sources(data: DocumentsPipeline, rank: int, world_size: int, paths: List[str]):
    """
    주어진 경로 리스트에서 데이터를 순차적으로 로드하는 제너레이터입니다.
    datatrove 파이프라인의 표준 인자를 받습니다.
    """
    # 이전 단계의 데이터가 있다면 먼저 전달합니다. (이 경우 없음)
    if data:
        yield from data

    for path in paths:
        print(f"--- Task {rank}/{world_size}: '{path}'에서 데이터 로딩 시작 ---")
        reader = ParquetReader(path, limit=4096) # 테스트를 위해 limit 유지
        # 각 리더에 rank와 world_size를 전달하여 샤딩(sharding)이 올바르게 동작하도록 합니다.
        yield from reader.run(rank=rank, world_size=world_size)


# -- 파이프라인 설정 --

# huggingface-cli login 또는 login(token="...") 필요
# from huggingface_hub import login
# login("YOUR_HF_TOKEN")

# 상수 정의
TRAIN_DATASET_PATH = "hf://datasets/HuggingFaceFW/fineweb-2/data/kor_Hang/train"
TEST_DATASET_PATH = "hf://datasets/HuggingFaceFW/fineweb-2/data/kor_Hang/test"
TARGET_DATASET_NAME = "minpeter/fineweb-2-edu-korean-scored"
LOCAL_TEMP_DIR = "output/fineweb_korean_edu_scored_temp"

os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

# 처리할 데이터 소스 경로 리스트
all_paths = [TRAIN_DATASET_PATH, TEST_DATASET_PATH]

# 3. functools.partial을 사용하여 함수와 인자를 묶어줍니다.
loader = partial(load_multiple_sources, paths=all_paths)


if torch.cuda.is_available():
    num_tasks = torch.cuda.device_count()
    print(f"GPU 감지됨. {num_tasks}개의 GPU를 사용하여 병렬 처리합니다.")
else:
    num_tasks = os.cpu_count() or 1
    print(f"CPU만 사용. {num_tasks}개의 코어를 사용하여 병렬 처리합니다.")

executor = LocalPipelineExecutor(
    pipeline=[
        # 4. 파이프라인에는 함수 호출 결과가 아닌, partial 객체를 전달합니다.
        loader,
        EduScoreClassifier(batch_size=258),
        HuggingFaceDatasetWriter(
            dataset=TARGET_DATASET_NAME,
            local_working_dir=LOCAL_TEMP_DIR,
            private=False,
            output_filename="data/train-${rank}.parquet",
            cleanup=True,
        )
    ],
    tasks=num_tasks, # 병렬 처리를 원한다면 tasks > 1 로 설정할 수 있습니다.
)

if __name__ == '__main__':
    print("--- train 및 test 데이터 통합 처리 시작 ---")
    executor.run()
    print("--- 모든 데이터 처리 완료 ---")