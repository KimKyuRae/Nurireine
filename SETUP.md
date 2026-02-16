# Nurireine 설치 및 설정 가이드

Nurireine은 Discord AI 챗봇으로, 3계층 메모리 시스템과 AI 기반 대화 기능을 제공합니다.

## 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [설치](#설치)
3. [환경 변수 설정](#환경-변수-설정)
4. [실행](#실행)
5. [Discord 봇 설정](#discord-봇-설정)
6. [아키텍처 개요](#아키텍처-개요)
7. [문제 해결](#문제-해결)

---

## 시스템 요구사항

### 최소 요구사항
- **Python**: 3.9 이상 (PEP 585 타입 힌트 사용)
- **메모리**: 4GB RAM (GPU 사용 시 8GB 권장)
- **저장 공간**: 10GB (모델 저장용)
- **OS**: Windows, Linux, macOS

### GPU 가속 (선택사항)
- NVIDIA GPU (CUDA 지원)
- llama-cpp-python의 GPU 버전 설치 필요

---

## 설치

### 1. 저장소 클론
```bash
git clone https://github.com/KimKyuRae/Nurireine.git
cd Nurireine
```

### 2. Python 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

#### GPU 지원을 원하는 경우:
```bash
# CUDA 지원 llama-cpp-python 재설치
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## 환경 변수 설정

### 1. `.env` 파일 생성
`.env.example` 파일을 `.env`로 복사하고 편집합니다:

```bash
cp .env.example .env
```

### 2. 필수 환경 변수
```env
# Discord Bot Token (필수)
DISCORD_TOKEN=your_discord_bot_token_here

# Gemini API Key (필수)
# 여러 개의 키를 쉼표로 구분하여 입력 가능 (로테이션 지원)
GEMINI_API_KEY=key1,key2,key3

# LLM 모델 ID (선택, 기본값: gemini-2.5-flash-lite)
LLM_MODEL_ID=gemini-2.5-flash-lite
```

### 3. 선택적 환경 변수
```env
# 디버그 모드 (개발자용)
DEBUG_MODE=false
DEBUG_GUILD_ID=0
DEBUG_CHANNEL_ID=0
```

### API 키 발급 방법

#### Discord Bot Token
1. [Discord Developer Portal](https://discord.com/developers/applications)로 이동
2. "New Application" 클릭
3. Bot 메뉴에서 "Add Bot" 클릭
4. Token을 복사하여 `DISCORD_TOKEN`에 입력
5. **중요**: "Privileged Gateway Intents"에서 다음을 활성화:
   - MESSAGE CONTENT INTENT ✅

#### Gemini API Key
1. [Google AI Studio](https://makersuite.google.com/app/apikey)로 이동
2. "Create API Key" 클릭
3. API 키를 복사하여 `GEMINI_API_KEY`에 입력
4. 여러 키를 사용하면 quota 한도를 늘릴 수 있습니다

---

## 실행

### 기본 실행
```bash
python main.py
```

### 로그 확인
봇 실행 시 콘솔에 다음과 같은 메시지가 표시됩니다:
```
2026-02-16 10:23:36 [INFO] __main__: Starting Nurireine v2.0.0 (Debug: False)
2026-02-16 10:23:37 [INFO] Nurireine.bot: Nurireine initialized. AI systems set to lazy load.
2026-02-16 10:23:38 [INFO] discord.client: Logged in as Nurireine#1234 (ID: 1234567890)
2026-02-16 10:23:40 [INFO] Nurireine.bot: Loading AI systems in background...
2026-02-16 10:23:45 [INFO] Nurireine.bot: AI Core Systems Online.
```

### 백그라운드 실행 (Linux)
```bash
nohup python main.py > nurireine.log 2>&1 &
```

---

## Discord 봇 설정

### 1. 봇을 서버에 초대
1. [Discord Developer Portal](https://discord.com/developers/applications)에서 애플리케이션 선택
2. OAuth2 → URL Generator로 이동
3. **Scopes**:
   - `bot` ✅
   - `applications.commands` ✅
4. **Bot Permissions**:
   - Send Messages ✅
   - Read Message History ✅
   - Use Slash Commands ✅
5. 생성된 URL로 봇을 서버에 초대

### 2. 활성 채널 설정
봇을 초대한 후, 대화를 원하는 채널에서:
```
~-here
```

이제 봇이 해당 채널의 대화를 모니터링하고 자동으로 응답합니다.

### 3. 명령어 목록
- `~-here`: 현재 채널을 활성 채널로 설정
- `~-leave`: 활성 채널 설정 해제
- `~-status`: 봇의 상태 확인 (AI 시스템, 메모리, 통계)
- `~-testtimer`: 최근 대화 처리 시간 통계
- `~-sync`: (관리자 전용) 슬래시 명령어 동기화

### 4. 봇 호출 방법
- **명시적 호출**: "누리야", "누리", "누리레인", "레느" 등으로 호출
- **자동 응답**: 활성 채널에서 대화 흐름에 따라 자동으로 참여

---

## 아키텍처 개요

### 3계층 메모리 시스템

#### L1 - 단기 버퍼 (Short-term Buffer)
- 채널별로 최근 메시지 5-10개 저장
- 대화의 뉘앙스와 흐름 파악
- 빠른 접근을 위해 메모리에 보관

#### L2 - 롤링 요약 (Rolling Summary)
- 구조화된 마크다운 형식으로 대화 요약
- 주제, 분위기, 참여자, 핵심 포인트 추적
- SQLite 데이터베이스에 저장
- SLM(Small Language Model)이 지속적으로 업데이트

#### L3 - 장기 기억 (Long-term Memory)
- ChromaDB 벡터 데이터베이스 사용
- 중요한 사실과 컨텍스트를 시맨틱 검색 가능한 형태로 저장
- 한국어 임베딩 모델(KURE-v1) 사용
- 과거 대화 내용을 의미론적으로 검색

### AI 구성요소

1. **Gatekeeper (게이트키퍼)**
   - BERT 분류기로 응답 필요 여부 빠르게 판단
   - Gemma-3 4B 모델로 상세 분석
   - 검색 쿼리 생성, 저장할 사실 추출

2. **Main LLM (메인 LLM)**
   - Google Gemini 2.5-flash-lite 사용
   - 다중 API 키 로테이션 지원
   - 스트리밍 응답으로 실시간 느낌 제공
   - 도구 호출 지원 (웹 검색, GitHub 검색, YouTube 검색)

3. **Memory Manager (메모리 관리자)**
   - L1/L2/L3 계층 간 데이터 흐름 조율
   - 채널별 독립적인 메모리 관리
   - 중복 사실 저장 방지

### 보안 기능

- **Prompt Injection 방어**: 시스템 프롬프트 조작 시도 감지
- **입력 길이 제한**: 2000자 초과 입력 차단
- **말투 변경 방지**: 페르소나 일관성 유지
- **Rate Limiting**: 과도한 LLM 호출 방지 (20초당 1회)

---

## 문제 해결

### 봇이 시작되지 않음
```
Configuration Error: GEMINI_API_KEY is missing
```
→ `.env` 파일에 `GEMINI_API_KEY`를 설정했는지 확인

### AI 시스템 로딩 실패
```
Failed to load AI systems: ...
```
→ 모델 파일 다운로드 중 오류일 수 있습니다. 인터넷 연결 확인 및 재시도

### 메모리 부족 오류
→ SLM을 로컬 대신 Gemini API로 사용:
```env
# config.py 수정
SLM_PROVIDER=gemini  # 'local' 대신 'gemini'
```

### ChromaDB 오류
```
ChromaDB initialization failed
```
→ 데이터 디렉토리 권한 확인:
```bash
chmod -R 755 ./LTM
```

### 봇이 응답하지 않음
1. `~-status` 명령어로 AI 시스템 상태 확인
2. `~-here` 명령어로 활성 채널 설정 확인
3. "누리야" 등으로 명시적 호출 시도

---

## 개발자 정보

### 디버그 모드
```env
DEBUG_MODE=true
DEBUG_GUILD_ID=your_test_guild_id
DEBUG_CHANNEL_ID=your_test_channel_id
```

### 디버그 서버
디버그 모드에서는 WebSocket 디버그 서버가 localhost:8765에서 실행됩니다.
실시간 이벤트를 모니터링할 수 있습니다.

### 로깅
로그 레벨은 `DEBUG_MODE`에 따라 자동 조정됩니다:
- `DEBUG_MODE=true`: DEBUG 레벨
- `DEBUG_MODE=false`: INFO 레벨

---

## 추가 리소스

- **GitHub**: https://github.com/KimKyuRae/Nurireine
- **Issues**: 버그 리포트 및 기능 제안은 GitHub Issues에서
- **페르소나 정보**: `persona.md` 파일 참조
- **TODO 목록**: `TODO.md` 파일 참조

---

## 라이선스
이 프로젝트의 라이선스는 `LICENSE` 파일을 참조하세요.
