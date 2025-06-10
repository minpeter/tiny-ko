# -*- coding: utf-8 -*-
"""
Hugging Face 토크나이저 평가 스크립트

지정된 토크나이저를 로드하여 한국어 데이터셋에 대한 성능을
정량적 및 정성적으로 평가합니다.

평가 지표:
1.  **정량 평가:**
    -   어휘 집합 크기 (Vocabulary Size)
    -   텍스트 압축률 (Compression Ratio)
    -   단어 당 평균 Subword 개수
2.  **정성 평가:**
    -   샘플 단어 분절 방식 확인
3.  **가역성(Reversibility) 테스트:**
    -   인코딩-디코딩 후 원본 정보 보존 여부 확인
"""

from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def evaluate_tokenizer(tokenizer_path: str, dataset_name: str, dataset_split: str, num_samples: int = 1000):
    """
    학습된 토크나이저의 성능을 정량적/정성적으로 평가하는 메인 함수입니다.

    Args:
        tokenizer_path (str): 평가할 토크나이저의 경로 (로컬 디렉토리 또는 Hugging Face Hub 이름).
        dataset_name (str): 평가에 사용할 데이터셋의 Hugging Face Hub 이름.
        dataset_split (str): 사용할 데이터셋의 분할 (예: 'train', 'validation').
        num_samples (int, optional): 평가에 사용할 데이터셋 샘플의 수. Defaults to 1000.
    """
    print(f"'{tokenizer_path}' 토크나이저 성능 평가를 시작합니다.")
    print("=" * 50)

    # 1. 토크나이저 및 데이터셋 로드
    # streaming=True 옵션은 전체 데이터셋을 다운로드하지 않고, 필요한 만큼만 스트리밍하여 메모리 효율성을 높입니다.
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
        # take()를 사용하여 지정된 num_samples 만큼의 데이터만 가져옵니다.
        samples = list(tqdm(dataset.take(num_samples), total=num_samples, desc="데이터셋 샘플 로딩 중"))
        # 데이터셋에서 'text' 필드만 추출하여 리스트로 만듭니다. 텍스트가 비어있는 경우는 제외합니다.
        texts = [s['text'] for s in samples if s['text']]
    except Exception as e:
        print(f"토크나이저 또는 데이터셋 로드 중 오류 발생: {e}")
        return

    # --- 정량 평가 ---
    # 토크나이저의 일반적인 성능을 수치로 나타냅니다.
    print("\n[1. 정량 평가]")

    # 1-1. 어휘 집합 크기 (Vocabulary Size)
    # 토크나이저가 알고 있는 고유 토큰의 총 개수입니다.
    vocab_size = tokenizer.vocab_size
    print(f"  - 어휘 집합 크기 (Vocabulary Size): {vocab_size}")

    # 1-2. 텍스트 압축률 (Compression Ratio)
    # 원본 텍스트의 문자 수 대비 토큰 수의 비율입니다.
    # 이 값이 클수록 하나의 토큰이 더 많은 정보를 표현하므로 압축 효율이 좋다고 볼 수 있습니다.
    total_chars = sum(len(text) for text in texts)
    # 각 텍스트를 토큰 ID 리스트로 인코딩하고, 그 길이를 모두 더합니다.
    total_tokens = sum(len(tokenizer.encode(text)) for text in tqdm(texts, desc="압축률 계산 중"))
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"  - 텍스트 압축률 (Compression Ratio): {compression_ratio:.2f}")

    # 1-3. 단어 당 평균 Subword 개수
    # 하나의 단어(공백 기준)가 평균 몇 개의 토큰으로 쪼개지는지를 나타냅니다.
    # 한국어와 같이 교착어의 경우, 이 값이 너무 크면 의미 단위가 지나치게 분절된다는 의미일 수 있습니다.
    # 1에 가까울수록 단어 단위로 토크나이징이 잘 된다고 해석할 수 있습니다.
    total_words = sum(len(text.split()) for text in texts)
    total_subwords = sum(len(tokenizer.tokenize(text)) for text in tqdm(texts, desc="Subword 분석 중"))
    avg_subwords_per_word = total_subwords / total_words if total_words > 0 else 0
    print(f"  - 단어 당 평균 Subword 개수: {avg_subwords_per_word:.2f}")


    # --- 정성 평가 ---
    # 실제 단어들이 어떻게 분절되는지 직접 눈으로 확인하여 토크나이저의 품질을 평가합니다.
    print("\n[2. 정성 평가 (샘플 단어 분절 테스트)]")

    # 평가할 단어 목록: 일반 명사, 전문 용어(LLM), 신조어, 복합 명사 등
    sample_words = [
        "토크나이저", "LLM", "자연어처리", "어텐션", "트랜스포머",
        "대한민국", "데이터사이언티스트", "딥러닝", "인공지능",
        "아버지가방에들어가신다", "챗지피티", "인스타그램", "유튜브",
        "Hello, World!", "Python", "TensorFlow", "PyTorch",
    ]

    for word in sample_words:
        # 주어진 단어를 토크나이저를 이용해 토큰 리스트로 분절합니다.
        tokens = tokenizer.tokenize(word)

        # 각 토큰을 개별적으로 디코딩하여 사람이 읽을 수 있는 형태로 변환합니다.
        # `tokenizer.convert_tokens_to_string(tokens)`는 모든 토큰을 합쳐 하나의 문자열로 만들기 때문에,
        # 개별 토큰의 분절 상태를 확인하기 위해서는 아래와 같이 각 토큰을 따로 디코딩해야 합니다.
        decoded_tokens = []
        for token in tokens:
            # 토큰 하나를 ID로 변환한 후, 다시 디코딩하여 원래 문자열 조각으로 만듭니다.
            # 이 과정을 통해 'Ġ' 같은 특수 기호가 공백으로 변환되는 등 사람이 읽기 쉬운 형태로 보입니다.
            token_id = tokenizer.convert_tokens_to_ids(token)
            decoded_token = tokenizer.decode([token_id])
            decoded_tokens.append(decoded_token)

        # 분절된 토큰 리스트를 보기 좋게 출력합니다.
        print(f"  - '{word}': {decoded_tokens}")


    # --- 가역성(Reversibility) 테스트 ---
    # 텍스트를 토큰화(인코딩)했다가 다시 텍스트로 복원(디코딩)했을 때, 원본 정보가 손실되지 않는지 확인합니다.
    # "Round-trip consistency"라고도 부릅니다.
    print("\n[3. 가역성(Reversibility) 테스트]")
    test_sentence = "안녕하세요! 2024년에도 LLM 만들기는 재밌네요. 😂"

    # 1. 원본 문장을 토큰 ID 리스트로 인코딩합니다. (특수 토큰 제외)
    original_ids = tokenizer.encode(test_sentence, add_special_tokens=False)
    # 2. 인코딩된 ID 리스트를 다시 텍스트로 디코딩합니다.
    decoded_text = tokenizer.decode(original_ids)
    # 3. 디코딩된 텍스트를 다시 인코딩합니다.
    re_encoded_ids = tokenizer.encode(decoded_text, add_special_tokens=False)

    print(f"  - 원본 문장: {test_sentence}")
    print(f"  - 디코딩된 텍스트: {decoded_text}")

    # 4. 원본 ID와 재인코딩된 ID가 완전히 일치하는지 비교합니다.
    if original_ids == re_encoded_ids:
        print("  - 결과: ✅ 완벽하게 복원되었습니다 (Round-trip consistency 통과).")
    else:
        print("  - 결과: ❌ 복원 실패! ID가 일치하지 않습니다.")
        print(f"    - 원본 ID: {original_ids}")
        print(f"    - 재인코딩 ID: {re_encoded_ids}")


    print("\n" + "=" * 50)
    print("평가가 완료되었습니다.")


if __name__ == "__main__":
    # --- 스크립트 실행 설정 ---
    # 사용자가 자신의 환경에 맞게 수정해야 할 부분입니다.

    # 평가할 토크나이저의 경로 (로컬 경로 또는 Hugging Face Hub 경로)
    TOKENIZER_PATH = "/data/minpeter/github.com/minpeter/mirco-ko-llama/tknz/my_llm_tokenizer_for_hf"
    # 평가에 사용할 데이터셋의 Hugging Face Hub 경로
    DATASET_NAME = "minpeter/tiny-ko-corpus"
    # 사용할 데이터셋의 종류 (예: 'train', 'validation', 'test')
    DATASET_SPLIT = "train"
    # 정량 평가에 사용할 샘플의 개수
    NUM_SAMPLES = 2000

    # 설정된 값으로 토크나이저 평가 함수를 호출합니다.
    evaluate_tokenizer(
        tokenizer_path=TOKENIZER_PATH,
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        num_samples=NUM_SAMPLES
    )