# 파인튜닝된 로봇 에이전트 모델 - Windows 사용 안내서

이 안내서는 Windows 환경에서 Ollama를 사용하여 파인튜닝된 로봇 에이전트 모델을 설정하고 실행하는 방법을 설명합니다.

## 모델 소스에 대한 중요한 안내 (안정성 기준)

이 프로젝트에서 **안정적인 재현성과 장기 사용**을 위해 권장하는 구조는 다음과 같습니다.

- **Hugging Face**: 베이스 모델의 기준 소스(Source of Truth)
  - 파일 무결성(SHA) 보장
  - 중단 시 재개 가능
  - 다른 PC/환경에서도 동일한 모델 재현 가능
- **Ollama**: 로컬 실행기(Runtime)
  - 모델 실행과 프롬프트 처리 담당
  - LoRA 어댑터를 적용해 빠르게 실험 가능

즉, *베이스 모델은 Hugging Face에서 관리하고*,  
*Ollama는 이를 실행하기 위한 도구로만 사용*하는 구조가 가장 안정적입니다.

Ollama Hub에서 직접 내려받는 방식은 편리하지만,
대용량 모델 손상, 캐시 충돌, 환경 의존 문제가 발생할 수 있어
졸업작품·시연·팀 프로젝트에는 권장하지 않습니다.

## Hugging Face 기준 설치 방법 (권장 · 안정 모드)

이 방식은 **베이스 모델을 Hugging Face에서 직접 관리**하고,
Ollama는 **로컬 실행기(Runtime)** 로만 사용하는 가장 안정적인 방법입니다.
팀 프로젝트, 외장 SSD 이동, 졸업작품/시연 환경에 권장됩니다.

---

### 1단계: Hugging Face CLI 설치 및 로그인

```sh
pip install -U huggingface_hub
huggingface-cli login
```

토큰은 Hugging Face 계정 → Settings → Access Tokens 에서 생성합니다.

---

### 2단계: 베이스 모델 다운로드 (GGUF 권장)

아래 예시는 **Llama 3.1 8B Instruct (GGUF)** 를 로컬 폴더에 저장하는 방법입니다.

```sh
huggingface-cli download \
  TheBloke/Llama-3.1-8B-Instruct-GGUF \
  llama-3.1-8b-instruct.Q4_K_M.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False
```

> 권장 이유:
> - GGUF는 Ollama/llama.cpp 계열과 호환성이 가장 좋음
> - 파일 단위 관리가 가능하여 외장 SSD 이동이 쉬움

다운로드 후 폴더 구조 예시는 다음과 같습니다.

```
ai-tuning/
 ├─ models/
 │   └─ llama-3.1-8b-instruct.Q4_K_M.gguf
 └─ for_windows_user/
     ├─ Modelfile
     ├─ lora-llama31-8B-F32-LoRA.gguf
     └─ README.md
```

---

### 3단계: Modelfile 수정 (Hugging Face 로컬 모델 기준)

`Modelfile`에서 `FROM` 항목을 **로컬 GGUF 경로**로 변경합니다.

```text
FROM ../models/llama-3.1-8b-instruct.Q4_K_M.gguf
ADAPTER ./lora-llama31-8B-F32-LoRA.gguf
```

> `FROM llama3.1:8b` 방식(Ollama Hub 자동 다운로드)은
> 간편하지만 장기 안정성 측면에서는 권장하지 않습니다.

---

### 4단계: Ollama 모델 생성

```sh
cd ai-tuning/for_windows_user
ollama create robot-agent -f Modelfile
```

---

### 5단계: 실행

```sh
ollama run robot-agent
```

이제 Hugging Face에서 받은 **고정된 베이스 모델 + LoRA 어댑터** 조합으로
항상 동일한 결과를 재현할 수 있습니다.

## 동작 원리

* **베이스 모델 (Hugging Face)**: Hugging Face에서 직접 다운로드한 고정된 GGUF 베이스 모델을 사용합니다.
* **LoRA 어댑터**: 학습된 LoRA(`lora-llama31-8B-F32-LoRA.gguf`)를 베이스 모델 위에 적용하여 로봇 에이전트 동작을 정의합니다.
* **Ollama**: 로컬 실행기(Runtime)로서 GGUF + LoRA 조합을 실행합니다.
* **Modelfile**: 베이스 모델 경로, 어댑터, 시스템 프롬프트, 출력 제약(JSON)을 정의하는 실행 레시피입니다.
