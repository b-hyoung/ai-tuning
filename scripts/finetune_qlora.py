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
DATA_PATH = "./data/train.jsonl"   # {"text": "지민이는...\n상추가..."} 형식
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
    # 예: {"text": "지민이는 어떤프로젝트 해?\n상추가 잘자라는 무드등 만들어"}

    # -------------------------
    # 3. 토크나이징 + labels 부여
    # -------------------------
    print(">>> 토크나이징")

    def tokenize(batch):
        # batch["text"]는 문자열 리스트
        outputs = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,   # 이 Q&A는 짧으니까 128이면 충분
        )
        # Causal LM 기본: 입력 전체를 그대로 정답으로 학습
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
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
        num_train_epochs=3,          # 짧은 데이터니까 3epoch 정도는 돌려도 됨
        learning_rate=1e-4,          # 2e-4보단 조금 낮게
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # 네 transformers 버전에서는 evaluation_strategy 없음 → 일부러 안 넣음
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
