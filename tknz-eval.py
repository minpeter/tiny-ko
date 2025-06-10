import unicodedata
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def evaluate_tokenizer(tokenizer_path: str, dataset_name: str, dataset_split: str, num_samples: int = 1000):
    """
    학습된 토크나이저의 성능을 정량적/정성적으로 평가하는 함수

    Args:
        tokenizer_path (str): 평가할 토크나이저의 경로
        dataset_name (str): 평가에 사용할 데이터셋 이름
        dataset_split (str): 평가에 사용할 데이터셋 스플릿 (예: 'train')
        num_samples (int): 평가에 사용할 데이터셋 샘플 수
    """
    print(f"'{tokenizer_path}' 토크나이저 성능 평가를 시작합니다.")
    print("=" * 50)

    # 1. 토크나이저 및 데이터셋 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # streaming=True 옵션으로 전체 다운로드 없이 일부만 로드
        dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
        # 평가를 위해 일부 샘플만 가져오기
        samples = list(tqdm(dataset.take(num_samples), total=num_samples, desc="데이터셋 샘플 로딩 중"))
        texts = [s['text'] for s in samples if s['text']]
    except Exception as e:
        print(f"토크나이저 또는 데이터셋 로드 중 오류 발생: {e}")
        return

    # --- 정량 평가 ---
    print("\n[1. 정량 평가]")
    
    # 1-1. 어휘 집합 크기
    vocab_size = tokenizer.vocab_size
    print(f"  - 어휘 집합 크기 (Vocabulary Size): {vocab_size}")

    # 1-2. 압축률 및 토큰 분석
    total_chars = 0
    total_tokens = 0
    for text in tqdm(texts, desc="압축률 계산 중"):
        total_chars += len(text)
        total_tokens += len(tokenizer.encode(text))
    
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    chars_per_token = total_tokens / total_chars if total_chars > 0 else 0
    
    print(f"  - 텍스트 압축률 (Compression Ratio): {compression_ratio:.2f} (캐릭터 수 / 토큰 수)")
    print(f"  - 토큰 당 평균 글자 수: {1/chars_per_token:.2f}")

    # 1-3. 단어 당 평균 Subword 개수 (OOV 간접 평가)
    total_words = 0
    total_subwords = 0
    for text in tqdm(texts, desc="Subword 분석 중"):
        words = text.split() # 공백 기준으로 단어 분리
        if not words:
            continue
        total_words += len(words)
        total_subwords += len(tokenizer.tokenize(text)) # tokenize()는 subword 리스트 반환
        
    avg_subwords_per_word = total_subwords / total_words if total_words > 0 else 0
    print(f"  - 단어 당 평균 Subword 개수: {avg_subwords_per_word:.2f}")
    print("    (수치가 1에 가까울수록 단어가 통째로 어휘집에 있을 확률이 높음)")


    # --- 정성 평가 ---
    print("\n[2. 정성 평가 (샘플 단어 분절 테스트)]")
    
    sample_words = [
        "토크나이저", "LLM", "자연어처리", "어텐션", "트랜스포머",
        "대한민국", "데이터사이언티스트", "딥러닝", "인공지능",
        "아버지가방에들어가신다", "챗지피티"
    ]
    
    for word in sample_words:
        tokens = tokenizer.tokenize(word)
        print(f"  - '{word}': {tokens}")


    # --- 가역성(Reversibility) 테스트 ---
    print("\n[3. 가역성(Reversibility) 테스트]")
    print("  (인코딩 -> 디코딩 -> 재인코딩 후 토큰 ID 일치 여부로 확인)")

    test_sentence = "안녕하세요! 2024년에도 LLM 만들기는 재밌네요. 😂"

    # 1. 원본 문장을 토큰 ID로 인코딩합니다.
    #    순수한 토큰의 가역성을 보기 위해 특수 토큰(BOS, EOS)은 제외합니다.
    try:
        original_ids = tokenizer.encode(test_sentence, add_special_tokens=False)

        # 2. 인코딩된 ID를 다시 텍스트로 디코딩합니다.
        decoded_text = tokenizer.decode(original_ids)

        # 3. 디코딩된 텍스트를 다시 토큰 ID로 인코딩합니다.
        re_encoded_ids = tokenizer.encode(decoded_text, add_special_tokens=False)

        print(f"  - 원본 문장: {test_sentence}")
        # 디코딩된 텍스트는 정규화(예: 소문자화)가 적용된 상태일 수 있습니다.
        print(f"  - 디코딩된 텍스트: {decoded_text}")
        print(f"  - 원본 -> 인코딩 ID: {original_ids}")
        print(f"  - 디코딩 -> 재인코딩 ID: {re_encoded_ids}")

        # 4. 두 ID 리스트가 완전히 일치하는지 확인합니다.
        if original_ids == re_encoded_ids:
            print("  - 결과: ✅ 완벽하게 복원되었습니다 (Round-trip consistency 통과).")
        else:
            print("  - 결과: ❌ 복원 실패! ID가 일치하지 않습니다.")

    except Exception as e:
        print(f"  - 가역성 테스트 중 오류 발생: {e}")

    print("\n" + "=" * 50)
    print("평가가 완료되었습니다.")


if __name__ == "__main__":
    # --- 설정 ---
    # 이전에 학습시킨 토크나이저가 저장된 경로
    TOKENIZER_PATH = "/data/minpeter/github.com/minpeter/mirco-ko-llama/tknz/my_llm_tokenizer_for_hf"
    # 토크나이저 학습에 사용했던 데이터셋 (또는 유사한 성격의 테스트셋)
    DATASET_NAME = "minpeter/pretrain-korean-dedup"
    DATASET_SPLIT = "train"
    # 평가에 사용할 샘플 개수 (너무 많으면 시간이 오래 걸림)
    NUM_SAMPLES = 2000

    evaluate_tokenizer(
        tokenizer_path=TOKENIZER_PATH,
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        num_samples=NUM_SAMPLES
    )