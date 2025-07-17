from transformers import AutoTokenizer

# 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(
    "./artifacts/tknz/tiny-ko-tokenizer",
    # "LiquidAI/LFM2-350M",
    # "lgai-exaone/exaone-4.0-32b"
    # "qwen/qwen3-0.6b",
    # "meta-llama/Llama-2-7b-hf",
    add_bos_token=True,
    # from_slow=True,
)

# 토크나이징할 문장
text = "안녕하세요, 세계!"

# 토크나이저 호출 방식으로 인코딩
encoded_input = tokenizer(text)
if tokenizer.eos_token_id is not None:
    encoded_input["input_ids"].append(tokenizer.eos_token_id)
    encoded_input["attention_mask"].append(1)
    encoded_input["token_type_ids"].append(0) if "token_type_ids" in encoded_input else None

# 결과 출력
print(encoded_input)

# input_ids 디코딩해서 출력
decoded_text = tokenizer.decode(encoded_input["input_ids"])
print(f"Decoded text: {decoded_text}")