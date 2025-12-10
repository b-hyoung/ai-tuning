import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# 설정
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH = "./outputs/lora-llama31-8b"  # 학습된 LoRA 디렉토리

# 테스트용 질문
TEST_PROMPT = "지민이는 어떤프로젝트 해?"


def load_model():
    print(">>> 토크나이저 로딩")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> 베이스 모델 로딩")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print(">>> LoRA 어댑터 로딩:", LORA_PATH)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    return tokenizer, model


def main():
    tokenizer, model = load_model()

    # -----------------------------
    # 입력 토크나이즈
    # -----------------------------
    print("\n>>> 입력 질문:", TEST_PROMPT)
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)

    # -----------------------------
    # 생성
    # -----------------------------
    print("\n>>> 생성 시작")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    # -----------------------------
    # 결과 디코딩
    # -----------------------------
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n=== 모델 최종 출력 ===")
    print(text)

    # -----------------------------
    # 프롬프트 이후만 보기
    # -----------------------------
    generated_only = text[len(TEST_PROMPT):].strip()

    print("\n=== 모델이 생성한 답변 부분 ===")
    print(generated_only)


if __name__ == "__main__":
    main()
