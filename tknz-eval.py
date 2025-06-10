import unicodedata
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import List

# HELPER FUNCTION TO DECODE BYTE-LEVEL TOKENS
def decode_byte_tokens(tokens: List[str], tokenizer) -> List[str]:
    """
    Byte-Level BPE 토크나이저의 알아볼 수 없는 토큰 리스트를
    사람이 읽을 수 있는 문자열 리스트로 변환합니다.
    """
    readable_tokens = []
    for token in tokens:
        # 각 토큰을 바이트로 변환 후 UTF-8로 디코딩
        # 에러 발생 시 대체 문자로 표시하여 프로그램이 멈추지 않도록 함
        byte_representation = token.encode('latin-1') # 심볼을 바이트로
        try:
            readable_token = byte_representation.decode('utf-8')
        except UnicodeDecodeError:
            # 개별 바이트 토큰이 완전한 UTF-8 문자를 형성하지 못할 때 발생
            readable_token = repr(byte_representation)
        readable_tokens.append(readable_token)
    return readable_tokens

def evaluate_tokenizer(tokenizer_path: str, dataset_name: str, dataset_split: str, num_samples: int = 1000):
    """
    학습된 토크나이저의 성능을 정량적/정성적으로 평가하는 함수
    """
    print(f"'{tokenizer_path}' 토크나이저 성능 평가를 시작합니다.")
    print("=" * 50)

    # 1. 토크나이저 및 데이터셋 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
        samples = list(tqdm(dataset.take(num_samples), total=num_samples, desc="데이터셋 샘플 로딩 중"))
        texts = [s['text'] for s in samples if s['text']]
    except Exception as e:
        print(f"토크나이저 또는 데이터셋 로드 중 오류 발생: {e}")
        return

    # --- 정량 평가 ---
    # (이전과 동일하여 생략, 필요 시 위의 코드를 그대로 사용하세요)
    print("\n[1. 정량 평가]")
    vocab_size = tokenizer.vocab_size
    print(f"  - 어휘 집합 크기 (Vocabulary Size): {vocab_size}")
    total_chars = sum(len(text) for text in texts)
    total_tokens = sum(len(tokenizer.encode(text)) for text in tqdm(texts, desc="압축률 계산 중"))
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"  - 텍스트 압축률 (Compression Ratio): {compression_ratio:.2f}")
    total_words = sum(len(text.split()) for text in texts)
    total_subwords = sum(len(tokenizer.tokenize(text)) for text in tqdm(texts, desc="Subword 분석 중"))
    avg_subwords_per_word = total_subwords / total_words if total_words > 0 else 0
    print(f"  - 단어 당 평균 Subword 개수: {avg_subwords_per_word:.2f}")


    # --- 정성 평가 ---
    print("\n[2. 정성 평가 (샘플 단어 분절 테스트)]")
    
    sample_words = [
        "토크나이저", "LLM", "자연어처리", "어텐션", "트랜스포머",
        "대한민국", "데이터사이언티스트", "딥러닝", "인공지능",
        "아버지가방에들어가신다", "챗지피티"
    ]
    
    for word in sample_words:
        tokens = tokenizer.tokenize(word)
        # 🔄 [수정] 알아볼 수 없는 바이트 토큰을 읽을 수 있는 문자열로 변환
        # tokenizer.convert_tokens_to_string()은 전체를 합쳐버리므로, 각 토큰을 개별적으로 디코딩
        
        # 각 토큰을 개별적으로 디코딩하여 눈으로 확인
        decoded_tokens = []
        for token in tokens:
            # 토큰 하나를 디코딩하여 리스트에 추가
            decoded_tokens.append(tokenizer.decode([tokenizer.convert_tokens_to_ids(token)]))
            
        # 원본 토큰과 함께 보기 좋게 출력
        original_tokens_str = ' '.join(tokens)
        decoded_tokens_str = ' | '.join(decoded_tokens)
        print(f"  - '{word}': {decoded_tokens}")


    # --- 가역성(Reversibility) 테스트 ---
    # (이전과 동일하여 생략, 필요 시 위의 코드를 그대로 사용하세요)
    print("\n[3. 가역성(Reversibility) 테스트]")
    test_sentence = "안녕하세요! 2024년에도 LLM 만들기는 재밌네요. 😂"
    original_ids = tokenizer.encode(test_sentence, add_special_tokens=False)
    decoded_text = tokenizer.decode(original_ids)
    re_encoded_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
    print(f"  - 원본 문장: {test_sentence}")
    print(f"  - 디코딩된 텍스트: {decoded_text}")
    if original_ids == re_encoded_ids:
        print("  - 결과: ✅ 완벽하게 복원되었습니다 (Round-trip consistency 통과).")
    else:
        print("  - 결과: ❌ 복원 실패! ID가 일치하지 않습니다.")

    print("\n" + "=" * 50)
    print("평가가 완료되었습니다.")


if __name__ == "__main__":
    # --- 설정 ---
    TOKENIZER_PATH = "/data/minpeter/github.com/minpeter/mirco-ko-llama/tknz/my_llm_tokenizer_for_hf"
    DATASET_NAME = "minpeter/pretrain-korean-dedup"
    DATASET_SPLIT = "train"
    NUM_SAMPLES = 2000

    evaluate_tokenizer(
        tokenizer_path=TOKENIZER_PATH,
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        num_samples=NUM_SAMPLES
    )