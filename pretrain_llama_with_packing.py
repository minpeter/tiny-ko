import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithFlattening,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
# ConstantLengthDataset 대신 pack_dataset을 직접 임포트합니다.
from trl.data_utils import pack_dataset

# 로거 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- 인자 정의 ---
@dataclass
class ModelArguments:
    """모델 관련 인자"""
    model_name_or_path: str = field(
        metadata={"help": "사전 학습을 시작할 모델의 경로 또는 Hugging Face Hub 이름"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "사용자 정의 모델 코드를 신뢰할지 여부"},
    )

@dataclass
class DataArguments:
    """데이터 관련 인자"""
    dataset_name: str = field(
        metadata={"help": "데이터셋의 경로 또는 Hugging Face Hub 이름"}
    )
    dataset_text_field: str = field(default="text", metadata={"help": "데이터셋에서 텍스트가 포함된 필드 이름"})
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "토크나이징 후 시퀀스의 최대 길이"},
    )
    packing: bool = field(
        default=True,
        metadata={"help": "데이터셋 패킹을 사용하여 학습 효율을 높일지 여부"},
    )

def main():
    # --- 인자 파싱 ---
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 로깅 및 시드 설정 ---
    logging.basicConfig(level=training_args.get_process_log_level())
    logger.setLevel(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    set_seed(training_args.seed)

    logger.info(f"Training/evaluation parameters {training_args}")

    # --- 토크나이저 로드 ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 데이터셋 로드 및 처리 ---
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    dataset = load_dataset(data_args.dataset_name, split="train")

    # 항상 먼저 토크나이징을 수행합니다.
    logger.info("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(examples[data_args.dataset_text_field])

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=training_args.dataloader_num_workers,
        remove_columns=dataset.column_names,
    )

    # **<<<<< 수정된 부분 시작 >>>>>**
    if data_args.packing:
        logger.info(f"Packing dataset into sequences of length {data_args.max_seq_length}")
        # pack_dataset 함수를 직접 사용하여 이미 토크나이징된 데이터를 패킹합니다.
        # ConstantLengthDataset를 사용하지 않아 직렬화 오류를 피합니다.
        train_dataset = pack_dataset(
            tokenized_dataset,
            seq_length=data_args.max_seq_length,
        )
    else:
        logger.info("Skipping dataset packing.")
        train_dataset = tokenized_dataset
    # **<<<<< 수정된 부분 끝 >>>>>**


    # --- 데이터 콜레이터 설정 ---
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_collator = DataCollatorWithFlattening()

    # --- 모델 로드 ---
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # 수정된 부분
    )
    
    # --- 트레이너 설정 및 학습 시작 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # --- 모델 저장 ---
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training complete!")

if __name__ == "__main__":
    main()