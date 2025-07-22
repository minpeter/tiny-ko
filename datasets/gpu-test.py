import os
import shutil
import tempfile
import time
import json # 더미 데이터 생성 시 필요

import ray
import torch
from loguru import logger

from datatrove.executor import RayPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document, DocumentsPipeline
from datatrove.utils.batching import batched
from datatrove.utils.typeshelper import StatHints

# --- 1. 사용자 정의 파이프라인 단계 정의 ---
class SimpleGpuProcessingStep(PipelineStep):
    name = "간단한 GPU 처리"
    type = "처리"

    def __init__(self, batch_size: int = 4, target_case: str = "upper", **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.target_case = target_case
        self.device = None

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if self.device is None:
            self.device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
            logger.info(f"태스크 {rank}: {self.device} 사용 중.")

        # GPU 작업 시뮬레이션
        dummy_tensor = torch.zeros(1, device=self.device)
        _ = dummy_tensor + 1

        for batch_docs in batched(data, self.batch_size):
            self.stat_update("배치_수")

            with self.track_time(unit="batch"):
                processed_texts = []
                for doc in batch_docs:
                    if self.target_case == "upper":
                        processed_texts.append(doc.text.upper())
                    else:
                        processed_texts.append(doc.text.lower())
                
                time.sleep(0.01 * len(batch_docs)) # 시뮬레이션: GPU 작업 시간

            for i, doc in enumerate(batch_docs):
                doc.text = processed_texts[i]
                doc.metadata["processed_by_gpu"] = self.device
                self.stat_update(StatHints.forwarded)
                self.update_doc_stats(doc)
                yield doc

# --- 2. 데이터 준비 (가상의 입력 파일) ---
def create_dummy_data(folder_path, num_files=4, docs_per_file=10):
    os.makedirs(folder_path, exist_ok=True)
    for i in range(num_files):
        file_name = os.path.join(folder_path, f"data_part_{i:02d}.jsonl")
        with open(file_name, 'w', encoding='utf-8') as f:
            for j in range(docs_per_file):
                doc_id = f"{i}_{j}"
                text = f"This is document {doc_id} for processing."
                json.dump({"id": doc_id, "text": text}, f)
                f.write('\n')
    logger.info(f"더미 데이터 {num_files}개 파일 생성 완료: {folder_path}")

# --- 3. 파이프라인 구성 및 실행 ---
def run_pipeline():
    temp_dir = tempfile.mkdtemp()
    input_data_dir = os.path.join(temp_dir, "input_data")
    output_data_dir = os.path.join(temp_dir, "output_data")
    logging_dir = os.path.join(temp_dir, "logs")

    create_dummy_data(input_data_dir, num_files=4, docs_per_file=10)

    try:
        ray.init(num_gpus=2) # 2개의 GPU를 사용 (실제 GPU 수에 맞게 조절)
    except RuntimeError as e:
        if "Ray has already been started" in str(e):
            logger.info("Ray가 이미 초기화되어 있습니다.")
        else:
            raise

    pipeline_steps = [
        JsonlReader(data_folder=input_data_dir, glob_pattern="*.jsonl"),
        SimpleGpuProcessingStep(batch_size=4, target_case="upper"),
        JsonlWriter(
            output_folder=output_data_dir,
            output_filename="${rank}.jsonl",
            expand_metadata=True,
        )
    ]

    executor = RayPipelineExecutor(
        pipeline=pipeline_steps,
        tasks=4, # 4개의 논리적 태스크 (입력 파일 수와 일치)
        workers=2, # 동시에 2개의 태스크 실행 (2개의 GPU에 대응)
        cpus_per_task=1,
        ray_remote_kwargs={"num_gpus": 1}, # 각 Ray 태스크에 1개의 GPU 요청
        logging_dir=logging_dir,
        skip_completed=True, # 이어서 처리하는 기능 활성화
    )

    logger.info("--- 첫 번째 실행 시작 (중간에 강제 종료 시뮬레이션 가능) ---")
    try:
        executor.run()
    except Exception as e:
        logger.warning(f"첫 번째 실행 중단: {e}. (의도된 시뮬레이션일 수 있습니다.)")

    logger.info("\n--- 두 번째 실행 시작 (이어서 처리되는지 확인) ---")
    executor.run() # 동일한 Executor 인스턴스를 다시 실행

    logger.info("\n--- 파이프라인 실행 완료 및 결과 확인 ---")
    logger.info(f"생성된 출력 파일: {os.listdir(output_data_dir)}")
    logger.info(f"로그 파일: {os.listdir(os.path.join(logging_dir, 'logs'))}")

    # 임시 디렉토리 정리 (선택 사항)
    # shutil.rmtree(temp_dir)
    # logger.info(f"임시 디렉토리 {temp_dir} 삭제.")

if __name__ == "__main__":
    run_pipeline()