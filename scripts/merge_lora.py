import os
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# -----------------------------
# 경로 설정
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_DIR = "./outputs/lora-llama31-8b"
MERGED_MODEL_DIR = "./models/llama31-8b-merged"

def main():
    """
    LoRA 어댑터와 기본 모델을 병합하여 새로운 모델로 저장합니다.
    """
    print("="*50)
    print(">>> LoRA 어댑터와 기본 모델 병합을 시작합니다.")
    print(f"기본 모델: {BASE_MODEL}")
    print(f"LoRA 경로: {LORA_DIR}")
    print(f"저장 경로: {MERGED_MODEL_DIR}")
    print("="*50)

    # 1. LoRA가 적용된 모델과 토크나이저 로딩
    # device_map="auto" 를 통해 사용 가능한 GPU를 자동으로 사용합니다.
    print("\n>>> 1. LoRA 모델 로딩 중 (시간이 걸릴 수 있습니다)...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        LORA_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

    # 2. 모델 병합
    # merge_and_unload() 함수는 LoRA 레이어를 기본 모델에 병합합니다.
    print("\n>>> 2. 모델 병합 중...")
    model = model.merge_and_unload()
    print(">>> 병합 완료!")

    # 3. 병합된 모델과 토크나이저 저장
    print(f"\n>>> 3. 병합된 모델을 '{MERGED_MODEL_DIR}'에 저장 중...")
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
    model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    
    print("\n" + "="*50)
    print(">>> 병합 및 저장이 성공적으로 완료되었습니다!")
    print(f"이제 '{MERGED_MODEL_DIR}' 경로에 있는 모델을 GGUF로 변환할 수 있습니다.")
    print("="*50)


if __name__ == "__main__":
    main()
