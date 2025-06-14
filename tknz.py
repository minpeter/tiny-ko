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
    dataset = load_dataset("minpeter/tiny-ko-corpus", split="train[:5000]")

    print("✅ Dataset loaded successfully")
    print(dataset)

    def get_training_corpus():
        for i in range(len(dataset)):
            text = dataset[i]["text"]
            yield text if text is not None else ""

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    vocab_size = 32000

    tokens_to_add_as_normal = [
        AddedToken("<|im_start|>", special=True, normalized=False),
        AddedToken("<|im_end|>", special=True, normalized=False),
        AddedToken("<|unk_token|>", special=True, normalized=False),
        AddedToken("<|pad_token|>", special=True, normalized=False),
        # Special tokens for tool calls and thinking
        AddedToken("<tool_call>", special=False, normalized=False),
        AddedToken("</tool_call>", special=False, normalized=False),
        AddedToken("<think>", special=False, normalized=False),
        AddedToken("</think>", special=False, normalized=False),
        # Unused tokens (for future use or padding)
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

    byte_level_alphabet = pre_tokenizers.ByteLevel.alphabet()

    chosung = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    jungsung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
    jongsung = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
    hangul_jamo = list(set(chosung + jungsung + jongsung))
    
    chatml_reserved_words = [
        "system",
        "user",
        "assistant",
        "tool",
    ]

    final_initial_alphabet = sorted(list(set(byte_level_alphabet + hangul_jamo + chatml_reserved_words)))
 
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
        initial_alphabet=final_initial_alphabet,
    )

    print("⏳ Training started...")
    tokenizer.train_from_iterator(
        get_training_corpus(), trainer=trainer, length=len(dataset)
    )
    print("✅ Training completed!")
 
    print("\n✅ Adding extra tokens to the trained tokenizer...")
    tokenizer.add_tokens(tokens_to_add_as_normal)
    print(f"✅ {len(tokens_to_add_as_normal)} tokens have been added as normal tokens.")


    print("\n✅ Saving tokenizer in AutoTokenizer compatible format...")
    fast_tokenizer_wrapper = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk_token|>",
        pad_token="<|pad_token|>",
        eos_token="<|im_end|>",
        model_max_length=4096,
    )
    output_dir = "./tknz/tiny-ko-tokenizer"
    os.makedirs(output_dir, exist_ok=True)
    fast_tokenizer_wrapper.save_pretrained(output_dir)

    print(
        f"✅ Tokenizer and all config files have been saved to '{output_dir}'"
    )
    print("Generated files:", os.listdir(output_dir))

    print("\n--- Final validation ---")
    try:
        loaded_tokenizer_hf = AutoTokenizer.from_pretrained(output_dir)
        print("✅ AutoTokenizer loaded successfully!")
 
        text_to_test = "This is a <tool_call> test."
        output = loaded_tokenizer_hf(text_to_test)
        decoded_text = loaded_tokenizer_hf.decode(output["input_ids"])
        
        print(f"\nTest sentence: {text_to_test}")
        print(f"Encoded IDs: {output['input_ids']}")
        print(f"Decoded sentence: {decoded_text}")
        
        if "<tool_call>" in decoded_text:
            print("✅ Success: '<tool_call>' token is preserved after decoding.")
        else:
            print("❌ Failure: '<tool_call>' token disappeared after decoding.")

    except Exception as e:
        print(f"❌ Failed to load AutoTokenizer: {e}")

if __name__ == "__main__":
    train_and_save_huggingface_tokenizer()