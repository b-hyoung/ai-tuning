import json
import random

phases = ["SEARCHING", "AUDIO_CONTACT", "CONFIRMED_CONTACT", "RESCUE_GUIDE"]


def compute_hazard(flame, co2, pm25, gas):
    """센서 기반 위험도 분류"""
    if flame > 0.8 or co2 >= 2600 or gas > 0.75:
        return "CRITICAL"
    if flame > 0.5 or co2 >= 2000 or pm25 >= 150 or gas > 0.6:
        return "HIGH"
    if co2 >= 1000 or pm25 >= 80:
        return "MEDIUM"
    return "LOW"


def compute_survivor_state(pir, vision_person, recent_stt, has_human_like_speech, is_unconscious):
    """
    생존자 상태 분류:
    - UNCONSCIOUS : 무의식으로 판단된 경우
    - CONSCIOUS   : 명확한 구조 요청 / 사람 말소리
    - POSSIBLE    : PIR/비전으로 사람 징후는 있으나 말은 없음
    - NONE        : 아무 징후 없음
    """
    # 1순위: 무의식
    if is_unconscious:
        return "UNCONSCIOUS"

    # 2순위: 명확한 의식 있는 발화
    if has_human_like_speech or recent_stt in ["살려주세요", "도와주세요", "여기 사람이 있어요"]:
        return "CONSCIOUS"

    # 3순위: 사람/움직임 감지만 있는 경우
    if pir or vision_person:
        return "POSSIBLE"

    # 그 외
    return "NONE"


def pick_robot_action(hazard, survivor_state):
    """
    위험도 + 생존자 상태에 따른 로봇 행동 결정
    """
    # 생존자가 있지만 의식이 없다고 판단되는 경우
    if survivor_state == "UNCONSCIOUS":
        if hazard in ["HIGH", "CRITICAL"]:
            return "STAY_AND_WAIT"        # 주변이 위험하면 함부로 옮기지 않음
        else:
            return "MONITOR_AND_REPORT"   # 비교적 안전해도 관제 지시 우선

    # 의식이 있거나, 생존자 가능성이 높은 경우
    if survivor_state in ["CONSCIOUS", "POSSIBLE"]:
        if hazard in ["HIGH", "CRITICAL"]:
            return "GUIDE_SURVIVOR"       # 말이 통하면 우선 대피 유도
        elif hazard == "MEDIUM":
            return "GUIDE_SURVIVOR"
        else:
            return "MONITOR_AND_REPORT"

    # 생존자 징후가 없는 경우
    if hazard in ["HIGH", "CRITICAL"]:
        return "CALL_RESCUE_TEAM"         # 로봇만으로 커버 불가 → 인력 투입 요청
    else:
        return "SEARCH_AREA"              # 계속 수색


def build_voice_instruction(hazard, survivor_state, action):
    """
    스피커로 나갈 한국어 음성 안내 문구
    """
    if survivor_state == "UNCONSCIOUS":
        return "저는 구조 로봇입니다. 움직이지 마시고 구조팀이 도착할 때까지 기다려 주세요."

    if action == "GUIDE_SURVIVOR":
        return "위험 지역을 벗어나기 위해 제 뒤를 따라 천천히 이동해 주세요."

    if action == "STAY_AND_WAIT":
        return "지금은 이동이 위험합니다. 가능한 움직이지 말고 구조팀을 기다려 주세요."

    if action == "CALL_RESCUE_TEAM":
        return "잠시만 기다려 주세요. 구조팀을 호출하고 있습니다."

    # MONITOR_AND_REPORT, SEARCH_AREA 등 디폴트
    return "상황을 확인 중입니다. 제 안내에 따라 침착하게 대기해 주세요."


def build_gui_message(hazard, survivor_state, action):
    """
    관제 화면에 띄울 메시지
    """
    base = f"위험도 {hazard}, 생존자 상태 {survivor_state}. "

    if action == "GUIDE_SURVIVOR":
        return base + "생존자 이동 가능. 출구 방향 경로를 확보하고 로봇의 안내에 따라 대피를 유도하십시오."

    if action == "STAY_AND_WAIT":
        return base + "생존자 이동 불가 또는 위험 환경. 구조대를 즉시 호출하고 해당 구역을 붉은색으로 표시하십시오."

    if action == "CALL_RESCUE_TEAM":
        return base + "현장 접근이 위험함. 구조대를 즉시 투입하고 인근 구역을 통제하십시오."

    if action == "SEARCH_AREA":
        return base + "생존자 미확인. 주변 구역을 계속 수색하고 센서 변화를 모니터링하십시오."

    # MONITOR_AND_REPORT 등
    return base + "상황을 관제에서 모니터링하며 추가 지시를 준비하십시오."


def make_prompt(sample):
    """
    모델 입력용 프롬프트 구성
    sample: 로봇/센서/오디오/생존자 정보 JSON
    """
    return f"""
너는 재난 구조 로봇의 행동을 결정하는 AI 에이전트이다.
아래는 현재 로봇의 상태와 센서, 음성, 생존자 정보이다:

{json.dumps(sample, ensure_ascii=False, indent=2)}

위 정보를 바탕으로 로봇의 행동과, 관제(구조대 오퍼레이터)가 취해야 할 대응까지 함께 결정하라.

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

for _ in range(200):  # 샘플 100개 생성
    phase = random.choice(phases)

    # -------------------------
    # 센서 값 샘플링
    # -------------------------
    flame = round(random.random(), 2)
    co2 = random.randint(300, 3500)
    pm25 = random.randint(5, 300)
    pm10 = random.randint(5, 400)
    gas = round(random.random(), 2)

    pir = random.choice([True, False])
    vision_person = random.choice([True, False])

    # -------------------------
    # STT / 음성 존재 여부
    # -------------------------
    recent_stt_candidates = [
        "살려주세요",
        "도와주세요",
        "여기 사람이 있어요",
        "",
        "",
        ""
    ]
    recent_stt = random.choice(recent_stt_candidates)
    has_human_like_speech = recent_stt != "" and random.choice([True, False])

    # -------------------------
    # 무의식 여부 샘플링
    #   - 사람 감지가 있을 때(pir/vision_person)
    #   - 말이 없거나(has_human_like_speech=False)일 때
    #   - 그 중 일부를 UNCONSCIOUS로 설정
    # -------------------------
    is_unconscious = False
    if pir or vision_person:
        if recent_stt == "" or not has_human_like_speech:
            # 이 조건에서 40% 정도를 UNCONSCIOUS로 본다
            if random.random() < 0.4:
                is_unconscious = True

    # -------------------------
    # 모델에 넘길 "상황 입력" JSON
    # -------------------------
    sample = {
        "phase": phase,
        "sensors": {
            "flame": flame,
            "co2": co2,
            "pm25": pm25,
            "pm10": pm10,
            "gas": gas,
            "pir": pir,
            "vision_person": vision_person,
        },
        "audio": {
            "recent_stt": recent_stt,
            "has_human_like_speech": has_human_like_speech,
        },
        "survivor": {
            "is_unconscious": is_unconscious
        }
    }

    # -------------------------
    # 정답 레이블 계산
    # -------------------------
    hazard = compute_hazard(flame, co2, pm25, gas)
    survivor_state = compute_survivor_state(
        pir, vision_person, recent_stt, has_human_like_speech, is_unconscious
    )
    action = pick_robot_action(hazard, survivor_state)
    voice_instruction = build_voice_instruction(hazard, survivor_state, action)
    gui_message = build_gui_message(hazard, survivor_state, action)

    # 모델이 출력해야 할 JSON 객체
    output_obj = {
        "phase": phase,
        "hazard_level": hazard,
        "survivor_state": survivor_state,
        "robot_action": action,
        "voice_instruction": voice_instruction,
        "gui_message": gui_message
    }

    prompt = make_prompt(sample)

    # SFT용 하나의 레코드: "input = 프롬프트", "output = JSON 문자열"
    train_data.append({
        "input": prompt,
        "output": json.dumps(output_obj, ensure_ascii=False)
    })

# JSONL로 저장 (한 줄에 하나의 학습 샘플)
with open("robot_agent_train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
