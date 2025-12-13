# -*- coding: utf-8 -*-
import json
import random
import os

def generate_scenario():
    """
    survivor_speech가 상황에 맞는 '진단 질문'을 생성하도록 개선된 시나리오를 생성합니다.
    """
    has_survivor_signs = random.random() > 0.4
    
    sensors = {
        "flame": round(random.uniform(0.0, 0.1), 2), "co2": random.randint(400, 800), 
        "pm25": random.randint(10, 30), "pm10": random.randint(10, 40),
        "gas": round(random.uniform(0.0, 0.2), 2), "pir": False, "vision_person": False
    }
    audio = {"recent_stt": "", "has_human_like_speech": False}
    survivor_speech = ""

    if has_survivor_signs:
        phase = "CONFIRMED_CONTACT"
        is_unconscious = random.random() > 0.5
        sensors["pir"] = True
        sensors["vision_person"] = True
        
        if is_unconscious:
            survivor_state = "UNCONSCIOUS"
            robot_action = "REPORT_SURVIVOR"
            gui_message = "[긴급] 의식불명 생존자 / 위치 보고"
            voice_instruction = "의식불명 생존자 발견. 현 위치를 보고합니다."
            survivor_speech = "" # 의식 없는 생존자에게는 말을 걸지 않음
            hazard_level = "MEDIUM"
            sensors["co2"] = random.randint(1000, 1500)
        else: # CONSCIOUS
            survivor_state = "CONSCIOUS"
            robot_action = "GUIDE_SURVIVOR"
            gui_message = "[정보] 의식 있는 생존자 / 상태 확인 및 대피 유도"
            voice_instruction = "의식 있는 생존자를 발견했습니다. 상태 확인 및 대피 유도를 시작하겠습니다."
            audio["has_human_like_speech"] = True
            audio["recent_stt"] = random.choice(["도와주세요", "거기 누구있어요", ""])
            
            # 위험도 및 상황에 따라 다른 질문 생성
            hazard_level = "MEDIUM" if random.random() > 0.5 else "LOW"
            if hazard_level == "MEDIUM":
                 sensors["flame"] = round(random.uniform(0.3, 0.6), 2)
                 survivor_speech = "주변 구역의 위험도가 높습니다. 저는 구조 로봇입니다. 즉시 이곳을 벗어나야 합니다. 스스로 이동 가능하십니까?"
            else: # LOW
                if "도와주세요" in audio["recent_stt"]:
                     survivor_speech = "목소리를 듣고 왔습니다. 괜찮으십니까? 부상당한 곳이 있는지 알려주세요."
                else:
                     survivor_speech = "저는 구조 로봇입니다. 현재 위치는 안전합니다. 도움이 필요하십니까?"
    else: # No survivor
        phase = "SEARCHING"
        survivor_state = "NONE"
        robot_action = "SEARCH"
        survivor_speech = ""
        
        is_hazard_high = random.random() > 0.5
        if is_hazard_high:
            hazard_level = "HIGH"
            gui_message = "[경고] 고위험 구역 / 수색 지속"
            voice_instruction = "고위험 구역에서 수색을 계속합니다. 특이사항 없습니다."
        else:
            hazard_level = "LOW"
            gui_message = "[정상] 안전 구역 / 수색 지속"
            voice_instruction = "현재 구역은 안전합니다. 수색을 계속 진행합니다."

    input_sample = {"phase": phase, "sensors": sensors, "audio": audio, "survivor": {"is_unconscious": random.choice([True, False])}}
    output_sample = {"phase": phase, "hazard_level": hazard_level, "survivor_state": survivor_state, "robot_action": robot_action, "gui_message": gui_message, "voice_instruction": voice_instruction, "survivor_speech": survivor_speech}
    return input_sample, output_sample

def make_prompt_text(sample: dict) -> str:
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

def make_better_dataset(output_path: str, num_samples: int):
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            input_data, output_data = generate_scenario()
            example = {"input": make_prompt_text(input_data), "output": json.dumps(output_data, ensure_ascii=False)}
            f.write(json.dumps(example, ensure_ascii=False))
            f.write("\n")
    print(f">>> '{output_path}'에 {num_samples}개의 새로운 샘플 생성 완료.")

if __name__ == "__main__":
    new_data_path = "./data/robot_agent_train_refined.jsonl"
    num_to_generate = 1000 
    make_better_dataset(new_data_path, num_to_generate)
    print("\n참고: '상황별 질문 생성' 기능이 추가된 최종 데이터셋 생성이 완료되었습니다.")
    print("이제 추론 스크립트(infer.py)를 수정한 뒤, 이 데이터로 마지막 파인튜닝을 진행하세요.")