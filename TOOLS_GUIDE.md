# 도구 시스템 가이드 (Tools Guide)

누리레느의 도구 시스템은 LLM이 외부 정보에 접근하고 계산을 수행할 수 있도록 합니다.

## 도구 카테고리

### 1. 검색 도구 (Search Tools)

#### `web_search(query: str, max_results: int = 5) -> str`
인터넷에서 최신 정보를 검색합니다.

**사용 시기:**
- 최근 뉴스, 사건, 이벤트
- 실시간 정보
- 사실 확인이 필요한 정보
- 최신 데이터나 통계
- 모르는 사실이나 개념 설명

**예시:**
```python
web_search("2024년 올림픽 금메달")
web_search("Python 3.12 새로운 기능")
```

#### `news_search(query: str, max_results: int = 5) -> str`
최신 뉴스 기사를 검색합니다.

**사용 시기:**
- 최근 뉴스나 사건
- 언론 보도 내용
- 시사 이슈

**예시:**
```python
news_search("AI 기술 발전")
news_search("경제 뉴스")
```

#### `github_search(query: str, search_type: str = "users", max_results: int = 5) -> str`
GitHub에서 개발자 또는 오픈소스 프로젝트를 검색합니다.

**매개변수:**
- `query`: 검색 키워드
- `search_type`: "users" (개발자) 또는 "repositories" (프로젝트)

**예시:**
```python
github_search("torvalds", search_type="users")
github_search("react", search_type="repositories")
```

#### `youtube_search(query: str, max_results: int = 5) -> str`
YouTube에서 영상을 검색합니다.

**사용 시기:**
- 영상 콘텐츠 찾기
- 강의나 튜토리얼
- 뮤직비디오

**예시:**
```python
youtube_search("Python 기초 강의")
youtube_search("BTS 새 뮤직비디오")
```

#### `image_search(query: str, max_results: int = 5) -> str`
이미지를 검색합니다.

**예시:**
```python
image_search("고양이 사진")
image_search("에펠탑")
```

---

### 2. 유틸리티 도구 (Utility Tools)

#### `get_current_time() -> str`
현재 날짜와 시간을 확인합니다 (KST 기준).

**사용 시기:**
- "오늘 며칠이야?"
- "지금 몇 시야?"
- "무슨 요일이야?"

**예시:**
```python
get_current_time()
# 출력: 현재 시각: 2024년 3월 15일 (금요일) 14시 30분 25초 (KST)
```

#### `calculate(expression: str) -> str`
수학 계산을 수행합니다.

**지원하는 연산:**
- 사칙연산: `+`, `-`, `*`, `/`, `//`, `%`
- 거듭제곱: `**`
- 함수: `abs()`, `round()`, `max()`, `min()`, `sum()`
- 수학 함수: `sqrt()`, `sin()`, `cos()`, `tan()`, `log()`, `log10()`, `exp()`
- 상수: `pi`, `e`

**예시:**
```python
calculate("2 + 3 * 4")           # 결과: 14
calculate("sqrt(16)")            # 결과: 4
calculate("sin(pi/2)")           # 결과: 1
calculate("2**10")               # 결과: 1024
calculate("max(1, 5, 3, 9, 2)")  # 결과: 9
```

---

### 3. 번역 도구 (Translation Tools)

#### `translate_text(text: str, target_language: str = "ko") -> str`
텍스트를 다른 언어로 번역합니다.

**지원 언어:**
- `ko` / `korean` / `한국어`: 한국어
- `en` / `english` / `영어`: 영어
- `ja` / `japanese` / `일본어`: 일본어
- `zh` / `chinese` / `중국어`: 중국어

**예시:**
```python
translate_text("Hello, how are you?", target_language="ko")
translate_text("안녕하세요", target_language="en")
```

---

### 4. 메모리 도구 (Memory Tools)

#### `search_memory(query: str) -> str`
장기 기억(L3)에서 관련 정보를 검색합니다.

**사용 시기:**
- 사용자의 과거 정보나 설정
- 사용자의 선호도, 생일 등
- 이전 대화에서 저장된 사실

**예시:**
```python
search_memory("사용자 생일")
search_memory("좋아하는 음식")
```

#### `get_chat_history(limit: int = 10) -> str`
현재 채널의 최근 대화 이력을 가져옵니다.

**사용 시기 (제한적):**
- "방금 뭐라고 했어?"
- "아까 말한 거 뭐야?"
- 최근 대화의 정확한 내용 확인

**주의:** 일반 대화에서는 사용하지 마세요. L1 버퍼가 이미 제공됩니다.

**예시:**
```python
get_chat_history(limit=5)
```

---

## 도구 추가 방법

새로운 도구를 추가하려면:

1. **함수 구현**: `tools.py`에 함수를 작성
```python
def my_new_tool(param: str) -> str:
    """도구 설명"""
    # 구현
    return result
```

2. **레지스트리에 등록**: `TOOL_REGISTRY`에 추가
```python
TOOL_REGISTRY: Dict[str, Any] = {
    # ...
    "my_new_tool": my_new_tool,
}
```

3. **선언 추가**: `get_tool_declarations()`에 `FunctionDeclaration` 추가
```python
types.FunctionDeclaration(
    name="my_new_tool",
    description="도구 설명과 사용 시기",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "param": types.Schema(
                type=types.Type.STRING,
                description="매개변수 설명"
            ),
        },
        required=["param"]
    )
)
```

---

## 도구 사용 권장사항

### LLM이 도구를 더 잘 사용하도록 하려면:

1. **명확한 설명**: 도구 설명에 구체적인 사용 시기를 명시
2. **예시 포함**: 설명에 예시를 포함하여 이해도 향상
3. **카테고리 구분**: 비슷한 도구는 카테고리로 그룹화
4. **오용 방지**: "~할 때는 사용하지 마세요" 같은 제약사항 명시

### 개선된 설명 구조:
```python
description=(
    "도구가 하는 일을 한 줄로 설명.\n"
    "다음과 같은 경우 사용하세요:\n"
    "- 사용 예시 1\n"
    "- 사용 예시 2\n"
    "- 사용 예시 3\n"
    "주의: 사용하지 말아야 할 경우"
)
```

---

## 보안 고려사항

### 안전한 계산 (calculate)
- `eval()` 대신 AST를 사용하여 안전하게 수식 평가
- 허용된 연산자와 함수만 사용 가능
- 악의적인 코드 실행 방지

### 입력 검증
- 모든 도구는 입력값을 검증해야 함
- 에러 처리를 통해 안정성 확보

---

## 문제 해결

### 도구가 호출되지 않는 경우:
1. `config.py`에서 `enable_tools = True` 확인
2. 도구 설명이 명확한지 확인
3. LLM에게 도구 사용을 명시적으로 요청

### 도구 실행 오류:
1. 로그 확인: `logger.error` 메시지 확인
2. 입력 매개변수 타입 확인
3. 외부 API (DDGS 등) 상태 확인

---

## 향후 개선 방향

- [ ] 날씨 API 통합 (OpenWeatherMap 등)
- [ ] 위키피디아 검색 도구
- [ ] 파일 업로드/다운로드 도구
- [ ] 리마인더/알람 도구
- [ ] 투표/설문 도구
- [ ] 더 많은 언어 지원 (번역)
