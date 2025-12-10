import json
import random

phases = ["SEARCHING", "AUDIO_CONTACT", "CONFIRMED_CONTACT", "RESCUE_GUIDE"]


def compute_hazard(flame, co2, pm25, gas):
    if flame > 0.8 or co2 >= 2600 or gas > 0.75:
        return "CRITICAL"
    if flame > 0.5 or co2 >= 2000 or pm25 >= 150 or gas > 0.6:
        return "HIGH"
    if co2 >= 1000 or pm25 >= 80:
        return "MEDIUM"
    return "LOW"


def compute_survivor_state(pir, vision_person, recent_stt, has_human_like_speech):
    if has_human_like_speech or recent_stt in ["살려주세요", "도와주세요", "여기 사람이 있어요"]:
        return "CONSCIOUS"
    if pir or vision_person:
        return "POSSIBLE"
    return "NONE"


def pick_robot_action(hazard, survivor_state):
    if survivor_state == "UNCONSCIOUS":
        if hazard in ["HIGH", "CRITICAL"]:
            return "STAY_AND_WAIT"
        else:
            return "MONITOR_AND_REPORT"

    if survivor_state in ["CONSCIOUS", "POSSIBLE"]:
        if hazard in ["HIGH", "CRITICAL"]:
            return "GUIDE_SURVIVOR"
        elif hazard == "MEDIUM":
            return "GUIDE_SURVIVOR"
        else:
            return "MONITOR_AND_REPORT"

    if hazard in ["HIGH", "CRITICAL"]:
        return "CALL_RESCUE_TEAM"
    else:
        return "SEARCH_AREA"


def build_voice_instruction(hazard, survivor_state, action):
    if survivor_state == "UNCONSCIOUS":
        return "저는 구조 로봇입니다. 움직이지 마시고 구조팀이 도착할 때까지 기다려 주세요."
    if action == "GUIDE_SURVIVOR":
        return "위험 지역을 벗어나기 위해 제 뒤를 따라 천천히 이동해 주세요."
    if action == "STAY_AND_WAIT":
        return "지금은 이동이 위험합니다. 가능한 움직이지 말고 구조팀을 기다려 주세요."
    if action == "CALL_RESCUE_TEAM":
        return "잠시만 기다려 주세요. 구조팀을 호출하고 있습니다."
    return "상황을 확인 중입니다. 제 안내에 따라 침착하게 대기해 주세요."


def build_gui_message(hazard, survivor_state, action):
    base = f"위험도 {hazard}, 생존자 상태 {survivor_state}. "
    if action == "GUIDE_SURVIVOR":
        return base + "생존자 이동 가능. 출구 방향 경로를 확보하고 로봇의 안내에 따라 대피를 유도하십시오."
    if action == "STAY_AND_WAIT":
        return base + "생존자 이동 불가 또는 위험 환경. 구조대를 즉시 호출하고 해당 구역을 붉은색으로 표시하십시오."
    if action == "CALL_RESCUE_TEAM":
        return base + "현장 접근이 위험함. 구조대를 즉시 투입하고 인근 구역을 통제하십시오."
    if action == "SEARCH_AREA":
        return base + "생존자 미확인. 주변 구역을 계속 수색하고 센서 변화를 모니터링하십시오."
    return base + "상황을 관제에서 모니터링하며 추가 지시를 준비하십시오."


def make_prompt(sample):
    return f"""
너는 재난 구조 로봇의 행동을 결정하는 AI 에이전트이다.
아래는 현재 로봇의 상태와 센서, 음성 정보이다:

{json.dumps(sample, ensure_ascii=False, indent=2)}

위 정보를 바탕으로 로봇의 행동과,
관제(구조대 오퍼레이터)가 취해야 할 대응까지 함께 결정하라.

규칙:
- 출력은 JSON 객체 한 개만 포함해야 한다.
- JSON 바깥의 설명, 문장, 코드블록, 공백 줄을 절대 넣지 마라.
- true/false는 따옴표 없이 불리언으로 작성하라.
- 문자열 값에는 반드시 큰따옴표(")를 사용하라.

전체 출력은 아래 하나의 JSON 객체만 포함해야 한다.
""".strip()


# ===========================================================
#   🔥 핵심: prompt + output_json 을 text 하나로 합치기
# ===========================================================

train_data = []

for _ in range(5):  # 테스트용 5개만 생성
    phase = random.choice(phases)

    flame = round(random.random(), 2)
    co2   = random.randint(300, 3500)
    pm25  = random.randint(5, 300)
    pm10  = random.randint(5, 400)
    gas   = round(random.random(), 2)

    pir = random.choice([True, False])
    vision_person = random.choice([True, False])

    recent_s_
