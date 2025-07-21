import transformers
import torch
import argparse
import sys
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, GenerationConfig

def main():
    """
    채팅용으로 미세 조정되지 않은 Hugging Face 모델을 테스트하기 위한 CLI 애플리케이션입니다.
    TextIteratorStreamer를 사용하여 실시간으로 생성되는 텍스트를 스트리밍합니다.
    """
    parser = argparse.ArgumentParser(description="Interactive CLI for testing a Hugging Face text generation model.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID (e.g., gpt2)")
    args = parser.parse_args()

    model_id = args.model

    print(f"⚓️ 모델을 불러오는 중입니다: {model_id}...")
    try:
        # AutoTokenizer와 pipeline을 사용하여 모델과 토크나이저를 로드합니다.
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        # 실시간 출력을 위한 스트리머를 설정합니다.
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print("✅ 모델 로딩 완료! 'quit' 또는 'exit'를 입력하여 종료할 수 있습니다. 🏴‍☠️")
    
    except Exception as e:
        print(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    # 텍스트 생성에 사용할 옵션을 설정합니다.
    generation_config = GenerationConfig(
        # max_new_tokens=pipeline.model.config.max_position_embeddings,
        max_new_tokens=512,  # 최대 생성 토큰 수
        do_sample=True,
        top_p=0.95,
        temperature=0.4,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id  # pad_token_id를 eos_token_id로 설정하여 경고를 방지합니다.
    )

    while True:
        try:
            # 사용자로부터 프롬프트를 한 줄로 입력받습니다.
            prompt = input("> ")
            if prompt.lower() in ["quit", "exit"]:
                print("다음에 또 만나요! ⛵️")
                break
            
            # --- 💡 여기가 수정된 부분입니다! ---
            # 방금 입력한 줄로 커서를 이동시켜, 출력이 이어지는 것처럼 보이게 합니다.
            # \x1b[A: 커서를 한 줄 위로 이동
            # \r: 커서를 줄의 시작으로 이동
            sys.stdout.write("\x1b[A\r")
            # 프롬프트를 다시 출력하여 사용자의 입력을 복원하고, 모델 출력을 뒤에 이어 붙일 준비를 합니다.
            print(f"> {prompt}", end="", flush=True)

            generation_kwargs = {
                "text_inputs": prompt,
                "generation_config": generation_config,
                "streamer": streamer,
            }

            # 별도의 스레드에서 pipeline을 실행하여 입출력이 블로킹되지 않도록 합니다.
            thread = Thread(target=pipeline, kwargs=generation_kwargs)
            thread.start()

            # 스트리머로부터 생성되는 텍스트를 프롬프트 뒤에 바로 이어 출력합니다.
            for new_text in streamer:
                sys.stdout.write(new_text)
                sys.stdout.flush()
            # --- 수정 끝 ---

            # 모델 출력이 끝나면 다음 프롬프트를 위해 줄바꿈을 합니다.
            print() 
            
            thread.join() # 스레드가 끝날 때까지 대기합니다.

        except KeyboardInterrupt:
            print("\n다음에 또 만나요! ⛵️")
            break
        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
