from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch
import os

# -----------------------------
# 설정
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ⚠ 너가 만든 파일 위치에 맞춰라.
# 위의 데이터 생성 코드에서 robot_agent_train.jsonl을 현재 폴더에 만들었으면:
# DATA_PATH = "./robot_agent_train.jsonl"
DATA_PATH = "./data/robot_agent_train_refined.jsonl"

OUTPUT_DIR = "./outputs/lora-llama31-8b"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------
    # 1. 토크나이저 로딩
    # -------------------------
    print(">>> 토크나이저 로딩")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -------------------------
    # 2. 데이터 로딩
    # -------------------------
    print(">>> 데이터셋 로딩")
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    # 지금 구조: {"input": 프롬프트 문자열, "output": 정답 JSON 문자열}

    # -------------------------
    # 3. 토크나이징 + labels 부여
    # -------------------------
    print(">>> 토크나이징")

    def tokenize(batch):
        """
        batch["input"] : 상황 설명 + 규칙 프롬프트 (문자열 리스트)
        batch["output"]: 모델이 내야 할 JSON 정답 (문자열 리스트)
        """
        full_texts = []

        for inp, out in zip(batch["input"], batch["output"]):
            # 필요하면 포맷 더 정교하게 바꿔도 됨
            text = inp + "\n\n" + out
            full_texts.append(text)

        outputs = tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            # 프롬프트 + JSON 둘 다 포함이니까 128은 너무 짧다. 최소 512 이상 권장.
            max_length=512,
        )

        # 여기서는 간단히 전체 시퀀스를 정답으로 사용
        outputs["labels"] = outputs["input_ids"].copy()

        return outputs

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,  # input, output 컬럼 제거
    )

    # -------------------------
    # 4. 4bit QLoRA 베이스 모델 로딩
    # -------------------------
    print(">>> 4bit QLoRA용 베이스 모델 로딩")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------
    # 5. 학습 설정
    # -------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # evaluation_strategy 생략 (네 버전에서 오류 나던 부분 회피)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    # -------------------------
    # 6. 학습
    # -------------------------
    print(">>> 학습 시작")
    trainer.train()

    # -------------------------
    # 7. 저장
    # -------------------------
    print(">>> 모델 저장")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
