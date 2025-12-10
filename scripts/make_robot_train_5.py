import os
import json

# 생성할 데이터 개수
NUM_SAMPLES = 100

# 출력 파일 경로 (네 학습 스크립트의 DATA_PATH랑 맞춰라)
OUTPUT_PATH = "./data/train.jsonl"

PROMPT = "지민이는 어떤프로젝트 해?"
ANSWER = "상추가 잘자라는 무드등 만들어"


def make_robot_train(
    output_path: str = OUTPUT_PATH,
    num_samples: int = NUM_SAMPLES,
):
    # ./data 폴더 없으면 생성
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            # 현재 학습 스크립트 형태에 맞춰 text 하나로 합치기
            example = {
                "text": f"{PROMPT}\n{ANSWER}"
            }
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")

    print(f">>> {output_path} 에 {num_samples}개 샘플 생성 완료")


if __name__ == "__main__":
    make_robot_train()
