# uv run 01-preprocess.py --context_length 8192 --tokenizer_id minpeter/tiny-ko-tokenizer-32k-250725 --save_path ./artifacts/prepacked-8k-new

import argparse
import os
import time
import statistics  # for summary statistics
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import pack_dataset


parser = argparse.ArgumentParser(description="Preprocess datasets for tiny-ko")
parser.add_argument("--context_length", type=int, default=8192,
                    help="Context length for grouping texts")
parser.add_argument("--tokenizer_id", type=str,
                    default="minpeter/tiny-ko-tokenizer-32k-250725", help="Path to tokenizer")
parser.add_argument("--save_path", type=str,
                    default="./artifacts/prepacked", help="Path to save processed data")
parser.add_argument("--dataset_id", type=str, default="minpeter/tiny-corpus",
                    help="Dataset ID to load from Hugging Face Hub")

args = parser.parse_args()


def setup_directories():
    os.makedirs(args.save_path, exist_ok=True)


def load_raw_datasets():
    print("Loading raw datasets...")
    dataset = load_dataset(args.dataset_id, split="train[:100]")
    # return dataset.train_test_split(test_size=0.001, shuffle=True, seed=5768112)
    return dataset


def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"], padding=False, truncation=False)

    if tokenizer.eos_token_id is not None:
        for i in range(len(tokenized_inputs["input_ids"])):
            tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
            tokenized_inputs["attention_mask"][i].append(1)
            if "token_type_ids" in tokenized_inputs:
                tokenized_inputs["token_type_ids"][i].append(0)

    return tokenized_inputs


def ascii_histogram(data, bins=10, width=50):
    """
    Function to draw aligned Unicode histogram in terminal
    """

    # summary stats
    min_val, max_val = min(data), max(data)
    mean_val = statistics.mean(data)
    median_val = statistics.median(data)
    std_val = statistics.stdev(data) if len(data) > 1 else 0
    # bin edges and counts
    bin_size = (max_val - min_val) / bins if max_val > min_val else 1
    edges = [min_val + i * bin_size for i in range(bins + 1)]
    counts = [0] * bins
    for d in data:
        idx = int((d - min_val) / bin_size)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1
    max_count = max(counts) if counts else 1
    # formatted output
    range_width = 15
    count_width = 7
    hist_lines = []
    # header
    hist_lines.append(
        f"{'Range':<{range_width}} | {'Count':>{count_width}} | Histogram")
    hist_lines.append('-' * (range_width + 3 + count_width + 3 + width))
    # bins
    for i, count in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        range_str = f"{left:7.1f}-{right:7.1f}"
        count_str = f"{count:>{count_width}d}"
        bar = '▇' * int(count / max_count * width) if count > 0 else ''
        hist_lines.append(f"{range_str} | {count_str} | {bar}")
    # summary footer
    hist_lines.append(
        f"Min {min_val:.1f}, Max {max_val:.1f}, Mean {mean_val:.1f}, Median {median_val:.1f}, Std {std_val:.1f}")
    return hist_lines


if __name__ == "__main__":
    total_start_time = time.time()
    setup_directories()

    raw_datasets = load_raw_datasets()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    # num_processors = max(1, os.cpu_count() - 8)
    # During working hours, do not occupy all cluster resources.
    num_processors = 32
    print(
        f"Total CPUs: {os.cpu_count()}, Using {num_processors} processes for mapping.")

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_processors,
        remove_columns=raw_datasets.column_names,
    )

    packed_dataset = pack_dataset(
        tokenized_datasets, seq_length=args.context_length, strategy="wrapped")

    # Drop last incomplete chunk and report dropped tokens
    if len(packed_dataset) > 0:
        last_len = len(packed_dataset[-1]["input_ids"])
        if last_len < args.context_length:
            dropped_tokens = last_len
            print(f"\n\033[1;41;97mDropped tokens: {dropped_tokens}\033[0m")
            # remove the incomplete final sample
            packed_dataset = packed_dataset.select(
                list(range(len(packed_dataset) - 1)))

    # >>>>>> Debugging output >>>>>>
    SHOW_EXAMPLE_ROWS_LIMIT = 3

    for i in range(min(SHOW_EXAMPLE_ROWS_LIMIT, len(packed_dataset))):
        sample = packed_dataset[i]
        eos_id = tokenizer.eos_token_id

        colored_items = []
        for item in sample["input_ids"]:
            # get token string for display (decode ID to actual text)
            token_str = tokenizer.decode(
                [item], clean_up_tokenization_spaces=False)
            if item == eos_id:
                # red background for EOS tokens, white text
                colored_items.append(f'\033[41;97m{token_str}\033[0m({item})')
            else:
                # blue background for non-EOS tokens, white text
                colored_items.append(f'\033[44;97m{token_str}\033[0m({item})')
        print(
            f"\n\033[1;43;30m[PACKED SAMPLE {i}]\033[0m {' '.join(colored_items)}")
    # <<<<<< End of debugging output <<<<<<

    packed_dataset.save_to_disk(args.save_path)

    # Dataset statistics
    original_rows = len(raw_datasets)
    packed_rows = len(packed_dataset)
    print(f"\nOriginal dataset rows: {original_rows}")
    print(f"Packed dataset rows: {packed_rows}")

    # Original dataset token lengths
    original_lengths = [len(ids) for ids in tokenized_datasets["input_ids"]]
    print("\nOriginal dataset token length distribution:")
    for line in ascii_histogram(original_lengths):
        print(line)

    # Packed dataset token lengths
    packed_lengths = [len(sample["input_ids"]) for sample in packed_dataset]
    print("\nPacked dataset token length distribution:")
    for line in ascii_histogram(packed_lengths):
        print(line)

    print(
        f"\nTotal elapsed time: {(time.time() - total_start_time)/60:.2f} minutes")
