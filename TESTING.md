# Nurireine Testing Guide

This document describes the testing strategy and infrastructure for Nurireine.

## Testing Philosophy

Nurireine uses a pragmatic testing approach focusing on:
1. **Critical Paths**: Components that handle user data and AI interactions
2. **Business Logic**: Memory management, context assembly, security checks
3. **Integration Points**: Discord events, AI model interactions, database operations

## Test Structure

```
tests/
├── unit/              # Unit tests for individual modules
│   ├── test_memory.py
│   ├── test_gatekeeper.py
│   ├── test_validation.py
│   └── test_utils.py
├── integration/       # Integration tests
│   ├── test_bot_flow.py
│   └── test_ai_pipeline.py
├── fixtures/          # Test data and fixtures
│   ├── messages.json
│   └── contexts.json
└── conftest.py        # Pytest configuration
```

## Running Tests

### Setup Testing Environment

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Or add to requirements.txt:
# pytest>=7.0.0
# pytest-asyncio>=0.21.0
# pytest-cov>=4.0.0
# pytest-mock>=3.10.0
```

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/unit/test_memory.py
```

### Run with Coverage
```bash
pytest --cov=Nurireine --cov-report=html
```

### Run Async Tests
```bash
pytest -v tests/integration/
```

## Test Examples

### Unit Test Example: Validation

```python
# tests/unit/test_validation.py
import pytest
from Nurireine.validation import InputValidator

def test_validate_message_content_valid():
    content = "안녕하세요 누리야!"
    is_valid, error = InputValidator.validate_message_content(content)
    assert is_valid
    assert error is None

def test_validate_message_content_too_long():
    content = "x" * 3000
    is_valid, error = InputValidator.validate_message_content(content)
    assert not is_valid
    assert "too long" in error

def test_sanitize_for_embedding():
    text = "Hello\x00World\x01Test"
    sanitized = InputValidator.sanitize_for_embedding(text)
    assert '\x00' not in sanitized
    assert '\x01' not in sanitized
    assert "Hello" in sanitized
```

### Unit Test Example: Memory

```python
# tests/unit/test_memory.py
import pytest
from Nurireine.ai.memory import L2Summary

def test_l2_summary_from_markdown():
    md_text = """# 게임 이야기

## 분위기
- 편안하고 즐거움

## 참여자
- <user:123> (테스트유저)

## 핵심 포인트
- 사용자가 롤 플레이를 좋아함
"""
    summary = L2Summary.from_markdown(md_text)
    assert summary.topic == "게임 이야기"
    assert summary.mood == "편안하고 즐거움"
    assert len(summary.users) == 1
    assert len(summary.key_points) == 1

def test_l2_summary_to_markdown():
    summary = L2Summary(
        topic="테스트",
        mood="좋음",
        key_points=["포인트1", "포인트2"]
    )
    md = summary.to_markdown()
    assert "# 테스트" in md
    assert "포인트1" in md
    assert "포인트2" in md
```

### Integration Test Example: Bot Flow

```python
# tests/integration/test_bot_flow.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from Nurireine.bot import Nurireine

@pytest.mark.asyncio
async def test_message_processing_flow():
    """Test end-to-end message processing."""
    bot = Nurireine()
    
    # Mock Discord message
    message = MagicMock()
    message.author.id = 123456789
    message.author.display_name = "TestUser"
    message.content = "누리야 안녕!"
    message.guild.id = 987654321
    message.channel.id = 111222333
    message.reference = None
    message.attachments = []
    message.stickers = []
    message.created_at.timestamp.return_value = 1234567890.0
    
    # Mock AI components
    bot._ai_loaded = True
    bot.gatekeeper = MagicMock()
    bot.memory = AsyncMock()
    bot.llm = AsyncMock()
    
    # Mock analysis result
    bot.memory.plan_response.return_value = (
        {
            'l3_facts': '',
            'l2_summary': '',
            'l1_recent': []
        },
        {
            'response_needed': True,
            'retrieval_needed': False
        }
    )
    
    # Mock LLM response
    async def mock_stream(user_input, context, **kwargs):
        yield "안녕하세요! "
        yield "반갑습니다!"
    
    bot.llm.generate_response_stream = mock_stream
    message.channel.typing = AsyncMock()
    message.reply = AsyncMock()
    
    # Run message processing
    await bot.process_message_batch([message])
    
    # Verify flow
    bot.memory.plan_response.assert_called_once()
    message.reply.assert_called()
```

## Mocking Guidelines

### Mocking Discord Objects
```python
from unittest.mock import MagicMock, AsyncMock

# Mock message
message = MagicMock()
message.author.id = 123
message.content = "test"
message.channel.id = 456
message.reply = AsyncMock()  # Async method
```

### Mocking AI Components
```python
# Mock Gatekeeper
gatekeeper = MagicMock()
gatekeeper.analyze.return_value = {
    'response_needed': True,
    'search_query': '테스트'
}

# Mock Memory Manager
memory = AsyncMock()
memory.plan_response = AsyncMock(return_value=(context, analysis))
```

### Mocking ChromaDB
```python
# Mock ChromaDB collection
collection = MagicMock()
collection.count.return_value = 0
collection.query.return_value = {
    'documents': [['fact1', 'fact2']],
    'distances': [[0.1, 0.2]]
}
```

## Coverage Goals

| Module | Target Coverage | Priority |
|--------|----------------|----------|
| validation.py | 90%+ | High |
| health.py | 80%+ | High |
| ai/memory.py | 70%+ | High |
| ai/gatekeeper.py | 60%+ | Medium |
| ai/llm.py | 50%+ | Medium |
| bot.py | 40%+ | Medium |
| utils/ | 80%+ | Medium |

## Test Data

### Sample Messages
```python
SAMPLE_MESSAGES = [
    {
        "user_id": "123456789",
        "user_name": "TestUser",
        "content": "누리야 안녕!",
        "explicit_call": True
    },
    {
        "user_id": "123456789",
        "user_name": "TestUser",
        "content": "오늘 날씨 어때?",
        "explicit_call": False
    }
]
```

### Sample Contexts
```python
SAMPLE_CONTEXT = {
    "user_id": "123456789",
    "user_name": "TestUser",
    "bot_id": "987654321",
    "guild_id": "111222333",
    "l3_facts": "<user:123456789>님은 게임을 좋아한다.",
    "l2_summary": "# 게임 이야기\n\n편안한 분위기",
    "l1_recent": [
        {"role": "user", "content": "안녕!"}
    ]
}
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: pytest --cov=Nurireine --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Best Practices

### 1. Isolate External Dependencies
- Mock Discord API calls
- Mock AI model inference
- Mock database operations
- Use in-memory databases for testing

### 2. Test Async Code Properly
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### 3. Use Fixtures for Common Setup
```python
@pytest.fixture
def mock_bot():
    bot = Nurireine()
    bot._ai_loaded = True
    bot.memory = AsyncMock()
    return bot

def test_something(mock_bot):
    # Use mock_bot in test
    pass
```

### 4. Test Error Cases
```python
def test_validation_with_invalid_input():
    with pytest.raises(ValueError):
        validate_something(None)
```

### 5. Keep Tests Fast
- Use mocks instead of real API calls
- Skip slow integration tests in CI (unless necessary)
- Use smaller models or cached results

## Future Improvements

- [ ] Add performance benchmarks
- [ ] Add end-to-end tests with real Discord bot
- [ ] Add load testing for concurrent messages
- [ ] Add property-based testing (Hypothesis)
- [ ] Add mutation testing (mutpy)

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

---

**Note**: Tests are not yet implemented but this guide provides the framework for adding them.
