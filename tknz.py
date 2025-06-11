import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, AutoTokenizer

def train_and_save_huggingface_tokenizer():
    # dataset = load_dataset("minpeter/tiny-ko-corpus", split='train[:50000]')
    dataset = load_dataset("minpeter/tiny-ko-corpus", split='train')
    
    print("✅ 데이터셋 로드 완료")
    print(dataset)

    def get_training_corpus():
        for i in range(len(dataset)):
             # None 타입이 있는 경우를 대비해 빈 문자열로 처리
            text = dataset[i]['text']
            yield text if text is not None else ""
            
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk_token|>"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        # normalizers.Lowercase()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    vocab_size = 32000
    special_tokens = [
        "<|unk_token|>",
        "<|pad_token|>",
        "<|im_start|>",
        "<|im_end|>",
        "<tool_call>",
        "</tool_call>",
        "<think>",
        "</think>",
        "<|unused_token_0|>",
        "<|unused_token_1|>",
        "<|unused_token_2|>",
        "<|unused_token_3|>",
        "<|unused_token_4|>",
        "<|unused_token_5|>",
        "<|unused_token_6|>",
        "<|unused_token_7|>",
        "<|unused_token_8|>",
        "<|unused_token_9|>",
    ]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    print("⏳ 학습을 시작합니다...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer, length=len(dataset))
    print("✅ 학습 완료!")
    
    print("\n✅ AutoTokenizer 호환 형식으로 토크나이저를 저장합니다...")

    special_tokens_map = {
        "unk_token": "<|unk_token|>",
        "pad_token": "<|pad_token|>",
        "eos_token": "<|im_end|>",
    }
    
    fast_tokenizer_wrapper = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, # 훈련된 tokenizer 객체를 전달
        **special_tokens_map
    )

    output_dir = "./tknz/tiny-ko-tokenizer"
    os.makedirs(output_dir, exist_ok=True)
    
    fast_tokenizer_wrapper.save_pretrained(output_dir)
    
    print(f"✅ 토크나이저가 '{output_dir}' 경로에 모든 설정 파일과 함께 저장되었습니다.")
    print("생성된 파일 목록:", os.listdir(output_dir))

    print("\n--- 최종 검증 ---")
    print(f"'{output_dir}' 경로에서 AutoTokenizer로 로딩을 시도합니다...")
    
    try:
        loaded_tokenizer_hf = AutoTokenizer.from_pretrained(output_dir)
        print("✅ AutoTokenizer 로딩 성공!")
        
        text_to_test = "이렇게 만든 LLM용 토크나이저, 잘 될까?"
        output = loaded_tokenizer_hf(text_to_test)

        print(f"\n테스트 문장: {text_to_test}")
        print(f"인코딩된 ID: {output['input_ids']}")
        print(f"어텐션 마스크: {output['attention_mask']}")
        
        decoded_text = loaded_tokenizer_hf.decode(output['input_ids'])
        print(f"디코딩된 문장: {decoded_text}")
    except Exception as e:
        print(f"❌ AutoTokenizer 로딩 실패: {e}")

# 메인 함수 실행
if __name__ == "__main__":
    train_and_save_huggingface_tokenizer()