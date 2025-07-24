import argparse
import os
from datasets import load_dataset, Dataset
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


parser = argparse.ArgumentParser(
    description="Train and save a Hugging Face tokenizer.")
parser.add_argument("--tokenizer_id", type=str,
                    default="minpeter/tiny-ko-tokenizer",
                    required=True,
                    help="Huggingface tokenizer ID to save")
parser.add_argument("--vocab_size", type=int, default=32000,
                    help="Target vocabulary size for the tokenizer")

args = parser.parse_args()


def train_and_save_huggingface_tokenizer(
    dataset: Dataset,
    output_dir: str = "./artifacts/tknz",
    target_vocab_size: int = 32000,
    huggingface_hub_id: str = "minpeter/tiny-ko-tokenizer"
):
    # Define additional tokens needed, excluding EOS, BOS, PAD, UNK, and instruct tokens
    additional_tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        # Tokens for tool calls and thinking
        AddedToken("<tool_call>", special=False, normalized=False),
        AddedToken("</tool_call>", special=False, normalized=False),
        AddedToken("<think>", special=False, normalized=False),
        AddedToken("</think>", special=False, normalized=False),
        # Reserved special tokens
        AddedToken("<|unused_special_token_0|>",
                   special=True, normalized=False),
        AddedToken("<|unused_special_token_1|>",
                   special=True, normalized=False),
        AddedToken("<|unused_special_token_2|>",
                   special=True, normalized=False),
        AddedToken("<|unused_special_token_3|>",
                   special=True, normalized=False),
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
        # special_tokens=[

        # ],
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
    print(f"✅ Tokenizer saved to {output_dir}")
    fast_tokenizer.push_to_hub(huggingface_hub_id, private=False)
    print(f"✅ Tokenizer pushed to Hugging Face Hub: {huggingface_hub_id}")

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

    vocab_size = len(loaded_tokenizer)
    print("Tokenizer total vocab size:", vocab_size)
    if vocab_size % 64 != 0:
        print(
            f"\n⚠️ Warning: Tokenizer vocab size ({vocab_size}) is not divisible by 64.\n"
            "This may lead to slight performance degradation during model quantization or training."
        )


if __name__ == "__main__":
    dataset = load_dataset("minpeter/tiny-corpus", split="train")
    print("✅ Dataset loaded successfully")
    print(dataset)

    train_and_save_huggingface_tokenizer(
        dataset=dataset,
        output_dir=f"./artifacts/tknz/{args.tokenizer_id}",
        target_vocab_size=args.vocab_size,
        huggingface_hub_id=args.tokenizer_id
    )
