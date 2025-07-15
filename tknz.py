import os
from datasets import load_dataset, concatenate_datasets, Dataset
from tokenizers import (
    Tokenizer,
    AddedToken,
    Regex,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def train_and_save_huggingface_tokenizer(
    dataset: Dataset,
    output_dir: str = "./tknz/tiny-ko-tokenizer",
    target_vocab_size: int = 32000
):
    # Define additional tokens needed, excluding EOS, BOS, PAD, UNK, and instruct tokens
    additional_tokens = [
        # Tokens for tool calls and thinking
        AddedToken("<tool_call>", special=False, normalized=False),
        AddedToken("</tool_call>", special=False, normalized=False),
        AddedToken("<think>", special=False, normalized=False),
        AddedToken("</think>", special=False, normalized=False),
        # Reserved special tokens
        AddedToken("<|unused_special_token_0|>", special=True, normalized=False),
        AddedToken("<|unused_special_token_1|>", special=True, normalized=False),
        AddedToken("<|unused_special_token_2|>", special=True, normalized=False),
        AddedToken("<|unused_special_token_3|>", special=True, normalized=False),
    ]

    vocab_size = target_vocab_size - len(additional_tokens)

    def get_training_corpus():
        for i in range(len(dataset)):
            text = dataset[i]["text"]
            yield text if text is not None else ""

    tokenizer = Tokenizer(
        models.BPE(
            byte_fallback=True,
        )
    )
    tokenizer.normalizer = normalizers.NFKC()
    tiktoken_pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(tiktoken_pattern), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )

    tokenizer.decoder = decoders.ByteLevel()
    byte_level_alphabet = pre_tokenizers.ByteLevel.alphabet()
    chatml_reserved_words = ["system", "user", "assistant", "tool"]

    initial_alphabet = sorted(
        list(set(byte_level_alphabet + chatml_reserved_words))
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=initial_alphabet,
        min_frequency=2,
        max_token_length=30,
        special_tokens=[
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",

        ],
    )

    print("⏳ Training started...")
    tokenizer.train_from_iterator(
        get_training_corpus(), trainer=trainer, length=len(dataset)
    )
    print("✅ Training completed!")

    print("\n✅ Adding extra tokens to the trained tokenizer...")
    tokenizer.add_tokens(additional_tokens)
    print(f"✅ {len(additional_tokens)} tokens have been added.")

    print("\n✅ Saving tokenizer in AutoTokenizer compatible format...")
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        bos_token=None,
        add_bos_token=False,
        add_prefix_space=False,
        split_special_tokens=False,
        # model_max_length=8192,
    )

    os.makedirs(output_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(output_dir)

    print(f"✅ Tokenizer and config files have been saved to '{output_dir}'")
    print("Generated files:", os.listdir(output_dir))

    print("\n--- Final validation ---")
    try:
        loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print("✅ AutoTokenizer loaded successfully!")

        text_to_test = "This is a <tool_call> test."
        encoded_output = loaded_tokenizer(text_to_test)
        decoded_text = loaded_tokenizer.decode(encoded_output["input_ids"])

        print(f"\nTest sentence: {text_to_test}")
        print(f"Encoded IDs: {encoded_output['input_ids']}")
        print(f"Decoded sentence: {decoded_text}")

        if "<tool_call>" in decoded_text:
            print("✅ Success: '<tool_call>' token is preserved after decoding.")
        else:
            print("❌ Failure: '<tool_call>' token disappeared after decoding.")

    except Exception as e:
        print(f"❌ Failed to load AutoTokenizer: {e}")

    print("Tokenizer total vocab size:", len(loaded_tokenizer))

if __name__ == "__main__":
    ds_kr = load_dataset("minpeter/tiny-ko-corpus", split="train[:500]")
    cosmopedia = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        data_files=[f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(21)],
        split="train[:500]",
    )
    fineweb = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        data_files=[
            f"fineweb-edu-dedup/train-{i:05d}-of-00234.parquet" for i in range(21)
        ],
        split="train[:500]",
    )
    cosmopedia_text = cosmopedia.remove_columns(
        [col for col in cosmopedia.column_names if col != "text"]
    )
    fineweb_text = fineweb.remove_columns(
        [col for col in fineweb.column_names if col != "text"]
    )
    ds_en = concatenate_datasets([cosmopedia_text, fineweb_text])
    dataset = concatenate_datasets([ds_kr, ds_en])

    print("✅ Dataset loaded successfully")
    print(dataset)

    train_and_save_huggingface_tokenizer(
        dataset=dataset,
        output_dir="./tknz/tiny-ko-tokenizer-test",
        target_vocab_size=32000,
    )
