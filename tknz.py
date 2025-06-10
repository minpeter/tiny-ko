# -----------------------------------------------------------------------------
# STEP 0: 라이브러리 설치 (transformers 추가)
# -----------------------------------------------------------------------------
# 터미널에서 먼저 실행해주세요:
# pip install datasets tokenizers transformers
# -----------------------------------------------------------------------------

import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast, AutoTokenizer # 🔄 [추가] AutoTokenizer 로딩을 위한 클래스 임포트

def train_and_save_huggingface_tokenizer():
    """
    AutoTokenizer 호환 Fast 토크나이저를 학습하고 저장하는 최종 함수
    """
    
    # -----------------------------------------------------------------------------
    # STEP 1 ~ 3: 이전과 동일
    # -----------------------------------------------------------------------------
    dataset = load_dataset("minpeter/tiny-ko-corpus", split='train[:1000]')
    
    print("✅ 데이터셋 로드 완료")
    print(dataset)

    def get_training_corpus():
        for i in range(len(dataset)):
             # None 타입이 있는 경우를 대비해 빈 문자열로 처리
            text = dataset[i]['text']
            yield text if text is not None else ""
            
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    vocab_size = 32000
    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "<|endoftext|>"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    print("⏳ 학습을 시작합니다...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer, length=len(dataset))
    print("✅ 학습 완료!")
    
    # -----------------------------------------------------------------------------
    # 🔄 [수정] STEP 4, 5 통합: AutoTokenizer 호환 형식으로 저장
    # -----------------------------------------------------------------------------
    print("\n✅ AutoTokenizer 호환 형식으로 토크나이저를 저장합니다...")

    # 4-1. 훈련된 tokenizer 객체를 PreTrainedTokenizerFast 로 래핑(wrapping)
    # 이 과정에서 special token들의 역할을 명시적으로 지정해줍니다.
    special_tokens_map = {
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
    
    fast_tokenizer_wrapper = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, # 훈련된 tokenizer 객체를 전달
        **special_tokens_map
    )

    # 4-2. 저장할 디렉토리 설정
    output_dir = "./tknz/my_llm_tokenizer_for_hf"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4-3. .save_pretrained() 호출: 모든 필요 파일(json)을 자동으로 생성!
    fast_tokenizer_wrapper.save_pretrained(output_dir)
    
    print(f"✅ 토크나이저가 '{output_dir}' 경로에 모든 설정 파일과 함께 저장되었습니다.")
    print("생성된 파일 목록:", os.listdir(output_dir))

    # -----------------------------------------------------------------------------
    # STEP 6: 최종 검증 (AutoTokenizer로 직접 로드해보기)
    # -----------------------------------------------------------------------------
    print("\n--- 최종 검증 ---")
    print(f"'{output_dir}' 경로에서 AutoTokenizer로 로딩을 시도합니다...")
    
    try:
        loaded_tokenizer_hf = AutoTokenizer.from_pretrained(output_dir)
        print("✅ AutoTokenizer 로딩 성공!")
        
        text_to_test = "이렇게 만든 LLM용 토크나이저, 잘 될까?"
        output = loaded_tokenizer_hf(text_to_test) # __call__ 메소드로 바로 사용 가능

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