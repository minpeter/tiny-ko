import argparse
import os
import time
from itertools import chain
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


parser = argparse.ArgumentParser(description="Preprocess datasets for tiny-ko")
parser.add_argument("--context_length", type=int, default=8192, help="Context length for grouping texts")
parser.add_argument("--tokenizer_path", type=str, default="./tknz/tiny-ko-tokenizer", help="Path to tokenizer")
parser.add_argument("--save_path", type=str, default="./processed_data", help="Path to save processed data")

args = parser.parse_args()


def setup_directories():
    os.makedirs(os.path.dirname(args.tokenizer_path), exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

def load_raw_datasets():
    print("Loading raw datasets...")
    ds_kr = load_dataset("minpeter/tiny-ko-corpus", split="train[:10000]")
    cosmopedia = load_dataset("HuggingFaceTB/smollm-corpus", data_files=[f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(21)], split="train[:10000]")
    fineweb = load_dataset("HuggingFaceTB/smollm-corpus", data_files=[f"fineweb-edu-dedup/train-{i:05d}-of-00234.parquet" for i in range(21)], split="train[:10000]")
    cosmopedia_text = cosmopedia.remove_columns([col for col in cosmopedia.column_names if col != "text"])
    fineweb_text = fineweb.remove_columns([col for col in fineweb.column_names if col != "text"])
    ds_en = concatenate_datasets([cosmopedia_text, fineweb_text])
    ds = concatenate_datasets([ds_kr, ds_en])
    return ds.train_test_split(test_size=0.001, shuffle=True, seed=5768112)

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding=False, truncation=False)

    if tokenizer.eos_token_id is not None:
        for i in range(len(tokenized_inputs["input_ids"])):
            tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
            tokenized_inputs["attention_mask"][i].append(1)
            if "token_type_ids" in tokenized_inputs:
                tokenized_inputs["token_type_ids"][i].append(0)

    return tokenized_inputs

def group_texts(examples):
    block_size=args.context_length

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

if __name__ == "__main__":
    total_start_time = time.time()
    setup_directories()

    raw_datasets = load_raw_datasets()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
    )
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
    )

    lm_datasets.save_to_disk(args.save_path)

    print(f"\nTotal elapsed time: {(time.time() - total_start_time)/60:.2f} minutes")