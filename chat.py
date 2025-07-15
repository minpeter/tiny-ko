import transformers
import torch
import argparse
import sys
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, GenerationConfig

def main():
    """
    Hugging Face 모델과 상호작용하는 CLI 채팅 애플리케이션입니다.
    TextIteratorStreamer를 사용하여 실시간 응답 스트리밍을 지원합니다.
    """
    parser = argparse.ArgumentParser(description="Interactive CLI chat with a Hugging Face model.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID (e.g., minpeter/tiny-ko-124m-sft-muon)")
    args = parser.parse_args()

    model_id = args.model

    print(f"⚓️ Loading model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print("Model loaded successfully! Type 'quit' or 'exit' to end the chat. 🏴‍☠️")
    
    except Exception as e:
        print(f"Arr, matey! Couldn't load the model: {e}")
        sys.exit(1)

    messages = []
    
    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Farewell, matey! Until next time. ⛵️")
                break

            messages.append({"role": "user", "content": user_input})

            prompt = pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # --- 💡 여기가 수정된 부분입니다! ---
            # 키워드 인자(kwargs)는 streamer와 generation_config만 포함하도록 수정합니다.
            generation_kwargs = {
                "streamer": streamer,
                "generation_config": generation_config,
            }

            # 'prompt'는 위치 인자(args)로, 나머지는 키워드 인자(kwargs)로 전달합니다.
            # 'args'는 튜플이나 리스트 형태여야 하므로 [prompt]로 감싸줍니다.
            thread = Thread(target=pipeline, args=[prompt], kwargs=generation_kwargs)
            # --- 수정 끝 ---
            
            thread.start()

            print("Pirate: ", end="")
            full_response = ""
            for new_text in streamer:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                full_response += new_text

            print()
            
            thread.join()
            
            messages.append({"role": "assistant", "content": full_response.strip()})

        except KeyboardInterrupt:
            print("\nFarewell, matey! Until next time. ⛵️")
            break
        except Exception as e:
            # 오류 메시지를 친절하게 보여줍니다.
            print(f"\nShiver me timbers! An error occurred: {e}")
            if messages and messages[-1]["role"] == "user":
                messages.pop()


if __name__ == "__main__":
    main()