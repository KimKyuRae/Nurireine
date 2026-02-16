<h1 align="center">
  <img src="./assets/logo.svg" width="250" />

  Nurireine
</h1>

<p align="center">
  <strong>A self-directed AI bot that watches, reacts, and occasionally jumps into chat.</strong>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff" alt="Python">
<img src="https://img.shields.io/crates/l/laftel-rs?color=c0ffee&style=flat-square" alt="License">
</p>

---

## 특징 (Features)

### 🧠 3계층 메모리 시스템
- **L1 (최근 대화)**: 즉각적인 대화 흐름 파악
- **L2 (압축 요약)**: SLM이 생성한 대화 요약
- **L3 (장기 기억)**: 벡터 DB 기반 장기 기억 저장

### 🛠️ 풍부한 도구 시스템
누리레느는 다양한 도구를 활용하여 정확하고 유용한 정보를 제공합니다:

#### 검색 도구
- **웹 검색**: 최신 정보와 사실 확인
- **뉴스 검색**: 최신 뉴스 기사 검색
- **GitHub 검색**: 개발자와 오픈소스 프로젝트 찾기
- **YouTube 검색**: 영상 콘텐츠 검색
- **이미지 검색**: 이미지와 사진 찾기

#### 유틸리티 도구
- **시간 확인**: 현재 날짜와 시간 (KST)
- **계산기**: 수학 계산 수행 (사칙연산, 수학 함수 등)

#### 번역 도구
- **텍스트 번역**: 여러 언어 간 번역 지원

#### 메모리 도구
- **기억 검색**: 장기 기억에서 정보 검색
- **대화 이력**: 최근 대화 내역 조회

자세한 내용은 [TOOLS_GUIDE.md](./TOOLS_GUIDE.md)를 참조하세요.

### 🎭 페르소나 기반 대화
- 민트색 머리의 기계 인형 소녀 '누리레느'
- 시간술사 능력 설정
- 자연스럽고 일관된 성격

### 🔒 보안 및 안전성
- 입력 검증 시스템
- 프롬프트 인젝션 방어
- 안전한 도구 실행 (AST 기반 계산)

---

## 시작하기 (Getting Started)

자세한 설치 및 설정 가이드는 [SETUP.md](./SETUP.md)를 참조하세요.

### 빠른 시작

1. **저장소 클론**
   ```bash
   git clone https://github.com/KimKyuRae/Nurireine.git
   cd Nurireine
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **환경 변수 설정**
   ```bash
   cp .env.example .env
   # .env 파일 편집하여 API 키 등 설정
   ```

4. **봇 실행**
   ```bash
   python main.py
   ```

---

## 문서 (Documentation)

- [SETUP.md](./SETUP.md) - 상세 설치 및 설정 가이드
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 시스템 아키텍처 설명
- [TOOLS_GUIDE.md](./TOOLS_GUIDE.md) - 도구 시스템 가이드
- [TESTING.md](./TESTING.md) - 테스팅 가이드
- [IMPROVEMENTS_SUMMARY.md](./IMPROVEMENTS_SUMMARY.md) - 개선사항 요약

---

## 기여하기 (Contributing)

기여를 환영합니다! 이슈나 풀 리퀘스트를 자유롭게 제출해 주세요.

---

## 라이선스 (License)

이 프로젝트는 MIT 라이선스 하에 배포됩니다.