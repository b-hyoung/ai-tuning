import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 경로 설정 (LLM 폴더에서 실행 기준)
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# LORA_DIR = "./outputs/lora-llama31-8b"  # FEW-SHOT에서는 사용 안 함

# __file__ 기준으로 안전하게 경로 잡고 싶으면 아래로 교체 가능
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# LORA_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "outputs", "lora-llama31-8b"))


FEW_SHOT_EXAMPLES = [
    {
        "input": {
            "phase": "CONFIRMED_CONTACT",
            "sensors": { "flame": 0.9, "co2": 2800, "pm25": 200, "pm10": 250, "gas": 0.8, "pir": True, "vision_person": True },
            "audio": { "recent_stt": "살려주세요", "has_human_like_speech": True },
            "survivor": { "is_unconscious": False }
        },
        "output": {
            "phase": "CONFIRMED_CONTACT",
            "hazard_level": "HIGH",
            "survivor_state": "CONSCIOUS",
            "robot_action": "GUIDE_SURVIVOR",
            "gui_message": "[긴급] 화재 구역 / 의식 있는 생존자",
            "voice_instruction": "오퍼레이터님, 고위험 화재 구역에서 의식 있는 생존자를 발견했습니다. 대피 유도를 시작하겠습니다.",
            "survivor_speech": "저는 구조 로봇입니다. 이 구역은 위험하니 즉시 저를 따라 대피해야 합니다. 스스로 이동 가능하십니까?"
        }
    },
    {
        "input": {
            "phase": "SEARCHING",
            "sensors": { "flame": 0.0, "co2": 900, "pm25": 40, "pm10": 60, "gas": 0.2, "pir": False, "vision_person": False },
            "audio": { "recent_stt": "", "has_human_like_speech": False },
            "survivor": { "is_unconscious": False }
        },
        "output": {
            "phase": "SEARCHING",
            "hazard_level": "LOW",
            "survivor_state": "NONE",
            "robot_action": "SEARCH",
            "gui_message": "[정상] 안전 구역 / 수색 지속",
            "voice_instruction": "오퍼레이터님, 현재 구역은 안전합니다. 수색을 계속 진행하겠습니다.",
            "survivor_speech": ""
        }
    }
]


def make_prompt(sample: dict) -> str:
    """추론 및 학습 시 사용될 프롬프트 텍스트를 생성 (Few-shot 버전)"""

    # 기본 지시사항
    prompt = """너는 재난 현장에 투입된 구조 로봇을 제어하는, 침착하고 전문적인 AI 에이전트이다.
너의 임무는 센서 데이터를 분석하고, 로봇의 다음 행동을 결정하며, 인간 구조대 오퍼레이터 및 생존자와 명확하게 소통하는 것이다.
모든 응답은 간결하고 사실에 기반해야 한다.

주어진 입력 정보(Input)에 대해, 반드시 지정된 7개의 키를 포함하는 JSON 객체(Output)를 생성해야 한다.
출력 규칙을 반드시 준수하라.
"""

    # Few-shot 예시 추가
    for example in FEW_SHOT_EXAMPLES:
        prompt += "\n### 예시 ###\n"
        prompt += "Input:\n"
        prompt += f"{json.dumps(example['input'], ensure_ascii=False, indent=2)}\n"
        prompt += "Output:\n"
        prompt += f"{json.dumps(example['output'], ensure_ascii=False, indent=2)}\n"

    # 실제 추론할 샘플 추가
    prompt += "\n### 실제 임무 ###\n"
    prompt += "Input:\n"
    prompt += f"{json.dumps(sample, ensure_ascii=False, indent=2)}\n"
    prompt += "Output:\n"
    
    return prompt.strip()


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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(">>> 기본 모델 로딩 (Few-shot용)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
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
            max_new_tokens=256,
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
    # run_inference("no_survivor_low", model, tokenizer)
    # run_inference("smoke_unconscious", model, tokenizer)

    run_inference("fire_conscious", model, tokenizer)
    # 여러 개를 연달아 테스트해도 로딩 없이 빠르게 실행됩니다.