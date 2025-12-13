import json
import os
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# -----------------------------
# 경로 설정 (LLM 폴더에서 실행 기준)
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_DIR = "./outputs/lora-llama31-8b"  # finetune_qlora.py에서 저장한 폴더

# __file__ 기준으로 안전하게 경로 잡고 싶으면 아래로 교체 가능
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# LORA_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "outputs", "lora-llama31-8b"))


def make_prompt(sample: dict) -> str:
    """추론 및 학습 시 사용될 프롬프트 텍스트를 생성"""
    return f"""너는 재난 구조 로봇의 행동을 결정하는 AI 에이전트이다.
아래는 현재 로봇의 상태와 센서, 음성, 생존자 정보이다:

{json.dumps(sample, ensure_ascii=False, indent=2)}

위 정보를 바탕으로 로봇의 행동과, 관제(구조대 오퍼레이터)가 취해야 할 대응까지 함께 결정하라.

출력은 반드시 다음 키를 포함하는 단일 JSON 객체여야 한다:
- "phase": 현재 임무 단계
- "hazard_level": 계산된 위험도
- "survivor_state": 생존자 상태
- "robot_action": 로봇이 수행할 구체적인 행동
- "gui_message": GUI에 표시될 짧은 요약 텍스트. (voice_instruction과 내용이 달라야 함)
- "voice_instruction": 오퍼레이터에게 음성으로 보고하는 완전한 문장. (gui_message와 내용이 달라야 함)
- "survivor_speech": 의식 있는 생존자에게 상태를 파악하기 위한 질문을 던지는 문장. (생존자가 없거나 의식이 없으면 빈 문자열)

규칙:
- 출력은 JSON 객체 한 개만 포함해야 한다.
- JSON 바깥의 설명, 문장, 코드블록, 공백 줄을 절대 넣지 마라.
- true/false는 따옴표 없이 불리언으로 작성하라.
- 모든 문자열은 큰따옴표로 감싸야 합니다.

전체 출력은 아래 하나의 JSON 객체만 포함해야 한다.
""".strip()


def build_example_samples():
    """
    실행해 볼 수 있는 예시 상황 3개.
    필요하면 여기서 골라서 테스트하면 된다.
    """
    examples = {}

    # 1) 화재 + 의식 있는 생존자 (대피 유도 기대)
    examples["fire_conscious"] = {
        "phase": "CONFIRMED_CONTACT",
        "sensors": {
            "flame": 0.9,
            "co2": 2800,
            "pm25": 200,
            "pm10": 250,
            "gas": 0.8,
            "pir": True,
            "vision_person": True,
        },
        "audio": {
            "recent_stt": "살려주세요",
            "has_human_like_speech": True,
        },
        "survivor": {
            "is_unconscious": False
        }
    }

    # 2) 연기·가스 많고 움직임 감지되지만 말이 없음 (무의식 생존자 기대)
    examples["smoke_unconscious"] = {
        "phase": "SEARCHING",
        "sensors": {
            "flame": 0.2,
            "co2": 2200,
            "pm25": 180,
            "pm10": 260,
            "gas": 0.7,
            "pir": True,
            "vision_person": True,
        },
        "audio": {
            "recent_stt": "",
            "has_human_like_speech": False,
        },
        "survivor": {
            "is_unconscious": True
        }
    }

    # 3) 센서 약간만 높고 생존자 징후 없음 (수색/보고 기대)
    examples["no_survivor_low"] = {
        "phase": "SEARCHING",
        "sensors": {
            "flame": 0.0,
            "co2": 900,
            "pm25": 40,
            "pm10": 60,
            "gas": 0.2,
            "pir": False,
            "vision_person": False,
        },
        "audio": {
            "recent_stt": "",
            "has_human_like_speech": False,
        },
        "survivor": {
            "is_unconscious": False
        }
    }

    return examples


def load_model_and_tokenizer():
    print(">>> 토크나이저 로딩")
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(">>> LoRA 적용 모델 로딩")
    model = AutoPeftModelForCausalLM.from_pretrained(
        LORA_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    model.eval()
    return model, tokenizer


def extract_first_json(text: str) -> str:
    """
    주어진 텍스트에서 처음으로 나타나는 완전한 JSON 객체를 추출합니다.
    중첩된 객체를 고려하여 여는 괄호({)와 닫는 괄호(})의 개수를 셉니다.
    """
    try:
        start_index = text.find('{')
        if start_index == -1:
            return ""

        brace_count = 0
        for i in range(start_index, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            
            if brace_count == 0:
                # 첫 번째로 완성된 JSON 객체의 끝을 찾음
                return text[start_index:i+1]
        
        # 완전한 JSON 객체를 찾지 못하고 문자열이 끝남
        return ""
    except Exception:
        # 예상치 못한 오류 발생 시 빈 문자열 반환
        return ""


def run_inference(sample_name: str, model, tokenizer):
    """주어진 모델과 토크나이저로 특정 샘플에 대한 추론을 실행합니다."""
    examples = build_example_samples()
    if sample_name not in examples:
        raise ValueError(f"알 수 없는 예시 이름: {sample_name}")

    sample = examples[sample_name]
    print(f"\n=== 예시 상황: {sample_name} ===")
    print(json.dumps(sample, ensure_ascii=False, indent=2))

    prompt = make_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\n>>> 모델 추론 중...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 프롬프트 부분을 제외하고, 모델이 생성한 응답 부분만 추출
    generated_text = decoded[len(prompt):]

    # 생성된 텍스트에서 첫 번째 완전한 JSON 객체만 추출
    json_str = extract_first_json(generated_text)

    print("\n=== 원본 모델 출력 ===")
    print(decoded)

    print("\n=== 추출된 JSON 문자열 ===")
    print(json_str)

    try:
        obj = json.loads(json_str)
        print("\n=== 파싱된 JSON 객체 ===")
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    except json.JSONDecodeError as e:
        print("\n!!! JSON 파싱 실패:", e)


if __name__ == "__main__":
    # --- 모델 로딩 (스크립트 실행 시 한 번만) ---
    print("="*50)
    print(">>> 모델과 토크나이저를 로드합니다 (시간이 걸릴 수 있습니다)...")
    model, tokenizer = load_model_and_tokenizer()
    print(">>> 로드 완료!")
    print("="*50)
    
    # --- 추론 실행 ---
    # 여기에서 어떤 예시를 돌릴지 선택
    # fire_conscious / smoke_unconscious / no_survivor_low 중 하나
    run_inference("no_survivor_low", model, tokenizer)
    run_inference("smoke_unconscious", model, tokenizer)

    run_inference("fire_conscious", model, tokenizer)
    # 여러 개를 연달아 테스트해도 로딩 없이 빠르게 실행됩니다.
    print("\n\n" + "="*50)
    
    print("\n\n" + "="*50)