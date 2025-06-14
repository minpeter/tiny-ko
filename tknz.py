import os
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    AddedToken,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer

def train_and_save_huggingface_tokenizer():
    """
    Trains a BPE tokenizer using the 'minpeter/tiny-ko-corpus' dataset 
    and saves it in a format compatible with the Hugging Face Transformers library.
    """
    # 1. Load Dataset
    dataset = load_dataset("minpeter/tiny-ko-corpus", split="train[:5000]")

    print("✅ Dataset loaded successfully")
    print(dataset)

    # Generator function to iterate over the training data
    def get_training_corpus():
        for i in range(len(dataset)):
            text = dataset[i]["text"]
            yield text if text is not None else ""

    # 2. Initialize and configure the tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    vocab_size = 32000

    # Define special/additional tokens to add after training
    additional_tokens = [
        AddedToken("<|im_start|>", special=True, normalized=False),
        AddedToken("<|im_end|>", special=True, normalized=False),
        AddedToken("<|unk_token|>", special=True, normalized=False),
        AddedToken("<|pad_token|>", special=True, normalized=False),
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
        AddedToken("<|unused_special_token_4|>", special=True, normalized=False),
        AddedToken("<|unused_special_token_5|>", special=True, normalized=False),
        AddedToken("<|unused_special_token_6|>", special=True, normalized=False),
        AddedToken("<|unused_special_token_7|>", special=True, normalized=False),
        AddedToken("<|unused_special_token_8|>", special=True, normalized=False),
        AddedToken("<|unused_special_token_9|>", special=True, normalized=False),
    ]

    # 3. Set up the Initial Alphabet
    byte_level_alphabet = pre_tokenizers.ByteLevel.alphabet()
    
    # Korean Jamo characters
    chosung = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    jungsung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
    jongsung = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
    hangul_jamo = list(set(chosung + jungsung + jongsung))
    
    # Reserved words for ChatML format
    chatml_reserved_words = ["system", "user", "assistant", "tool"]

    # Create the initial alphabet including byte-level, Hangul Jamo, and reserved words
    initial_alphabet = sorted(list(set(byte_level_alphabet + hangul_jamo + chatml_reserved_words)))
 
    # 4. Train the Tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=initial_alphabet,
    )

    print("⏳ Training started...")
    tokenizer.train_from_iterator(
        get_training_corpus(), trainer=trainer, length=len(dataset)
    )
    print("✅ Training completed!")
 
    print("\n✅ Adding extra tokens to the trained tokenizer...")
    tokenizer.add_tokens(additional_tokens)
    print(f"✅ {len(additional_tokens)} tokens have been added.")

    # 5. Save in Hugging Face compatible format
    print("\n✅ Saving tokenizer in AutoTokenizer compatible format...")
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk_token|>",
        pad_token="<|pad_token|>",
        eos_token="<|im_end|>",
        model_max_length=4096,
    )
    
    output_dir = "./tknz/tiny-ko-tokenizer"
    os.makedirs(output_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(output_dir)

    print(f"✅ Tokenizer and config files have been saved to '{output_dir}'")
    print("Generated files:", os.listdir(output_dir))

    # 6. Final Validation
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

if __name__ == "__main__":
    train_and_save_huggingface_tokenizer()