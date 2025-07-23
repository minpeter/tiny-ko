import ray
import torch
import os
import pandas as pd
from itertools import islice
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ray.util.actor_pool import ActorPool

# torch memory cleanup
torch.cuda.empty_cache()

# TF32 설정 (Ampere 아키텍처 이상에서 성능 향상)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print("TF32 사용 가능. 행렬 곱 정밀도 설정 중...")
    torch.set_float32_matmul_precision('high')

# 1. 각 GPU에서 모델을 로드하고 스코어링을 수행할 Ray Actor를 정의합니다.
# num_gpus=1 옵션은 각 액터가 GPU 하나를 점유하도록 보장합니다.
@ray.remote(num_gpus=1)
class ScoringActor:
    def __init__(self, model_name):
        # __init__는 액터가 처음 생성될 때 한 번만 호출됩니다.
        # 여기에 무거운 모델을 로드하여 반복적인 로딩을 피합니다.
        gpu_id = ray.get_gpu_ids()
        print(f"Actor on GPU {gpu_id} is loading the model...")
        
        # GPU capability 기반 dtype 결정
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32  # 기본값
        
        if "cuda" in self.device:
            gpu_capability = torch.cuda.get_device_capability(self.device)
            print(f"GPU capability detected: {gpu_capability}")
            
            # Ampere 아키텍처 이상 (A100, RTX 30xx/40xx 등)인지 확인
            if gpu_capability[0] >= 8:
                self.torch_dtype = torch.bfloat16
                print("bfloat16 지원 확인. 모델을 bfloat16으로 로드합니다.")
            else:
                self.torch_dtype = torch.float16
                print("bfloat16 미지원. 모델을 float16으로 로드합니다.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()
        
        # torch.compile 적용 (GPU에서만)
        if torch.cuda.is_available():
            print(f"모델 컴파일 시작...")
            self.model = torch.compile(self.model)
            print(f"모델 컴파일 완료.")
        
        print(f"Actor on GPU {ray.get_gpu_ids()} finished loading.")

    def score_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임으로 된 텍스트 배치를 입력받아 교육 점수를 계산하고,
        결과를 새로운 열에 추가하여 반환합니다.
        """
        # 모델 입력 형식에 맞게 "passage: " 접두어 추가
        texts = ["passage: " + text for text in batch_df["text"].tolist()]
        
        # 모델 추론
        with torch.no_grad():
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Logits에서 직접 점수 추출 (소프트맥스 미적용)
            # .squeeze(-1)을 통해 [batch_size, 1] -> [batch_size] 형태로 변환
            raw_scores = outputs.logits.squeeze(-1).float().cpu().numpy()
        
        # 결과를 데이터프레임에 추가
        batch_df['edu_score_raw'] = [round(float(score), 4) for score in raw_scores]
        batch_df['edu_score'] = [int(score) for score in raw_scores]
        
        return batch_df

def main():
    # --- 설정 ---
    # 사용하실 모델과 데이터셋 정보를 여기에 입력하세요.
    MODEL_NAME = "devngho/ko_edu_classifier_v2_nlpai-lab_KoE5" # 한국어 교육 분류 모델
    DATASET_NAME = "HuggingFaceFW/fineweb-2"
    DATASET_CONFIG = "kor_Hang"
    DATASET_SPLIT = "test"  # test 스플릿은 매우 작으므로 train으로 예시를 보여드립니다.
    
    BATCH_SIZE = 32  # GPU 메모리에 맞춰 조정하세요.
    OUTPUT_DIR = "./fineweb_scored_output"
    
    # --- Ray 초기화 ---
    if ray.is_initialized():
        ray.shutdown()
    ray.init(_temp_dir="/data/temp/ray")  # Ray 임시 디렉토리 설정
    
    if not torch.cuda.is_available():
        print("PyTorch가 CUDA를 사용할 수 없습니다. GPU 드라이버나 PyTorch 설치를 확인해주세요.")
        ray.shutdown()
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("사용 가능한 GPU를 찾을 수 없습니다. 스크립트를 종료합니다.")
        ray.shutdown()
        return

    print(f"사용 가능한 GPU {num_gpus}개를 사용하여 병렬 처리를 시작합니다.")

    # 2. 사용 가능한 GPU 수만큼 액터 풀(Actor Pool)을 생성합니다.
    actors = [ScoringActor.remote(MODEL_NAME) for _ in range(num_gpus)]
    pool = ActorPool(actors)

    # 3. Hugging Face 스트리밍 데이터셋을 로드합니다.
    hf_stream_ds = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True
    )
    
    # 스트리밍 데이터셋을 배치 단위로 나누기 위한 제너레이터 함수
    def batch_generator(iterator, batch_size):
        while True:
            chunk = list(islice(iterator, batch_size))
            if not chunk:
                return
            yield pd.DataFrame(chunk)

    # 4. 데이터셋을 순회하며 작업을 액터 풀에 분배합니다.
    print("데이터 스트림 처리를 시작합니다...")
    iterator = iter(hf_stream_ds)
    
    for batch_df in batch_generator(iterator, BATCH_SIZE):
        # submit()은 작업을 풀에 제출하고 즉시 다음 코드로 넘어갑니다. (비동기)
        pool.submit(lambda actor, df: actor.score_batch.remote(df), batch_df)

    # 5. 처리된 결과를 순서대로 받아 파일로 저장합니다.
    print("모든 작업을 제출했으며, 이제 결과를 수집하고 저장합니다.")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # tqdm을 사용하여 진행 상황을 시각화합니다.
    # pool.get_next_ordered()는 제출된 작업의 결과를 순서대로 기다려 반환합니다.
    # total 값은 추정치이며, 실제 데이터셋 크기에 맞게 조절하면 더 정확해집니다.
    progress_bar = tqdm(total=pool.has_next(), desc="Saving results")
    
    file_index = 0
    while pool.has_next():
        try:
            processed_batch_df = pool.get_next_ordered()
            output_path = os.path.join(OUTPUT_DIR, f"part-{file_index:05d}.parquet")
            processed_batch_df.to_parquet(output_path)
            file_index += 1
            progress_bar.update(1)
        except Exception as e:
            print(f"배치 처리 중 오류 발생: {e}")
            
    progress_bar.close()
    print(f"\n모든 처리가 완료되었습니다. 결과는 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")
    
    ray.shutdown()

if __name__ == "__main__":
    main()