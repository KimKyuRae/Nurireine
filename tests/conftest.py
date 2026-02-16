"""
Pytest configuration file for Nurireine tests.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_message_content():
    """Sample message content for testing."""
    return "누리야 안녕하세요! 오늘 날씨가 좋네요."


@pytest.fixture
def sample_user_id():
    """Sample Discord user ID."""
    return "123456789012345678"


@pytest.fixture
def sample_channel_id():
    """Sample Discord channel ID."""
    return 987654321098765432


@pytest.fixture
def sample_context():
    """Sample context dictionary for LLM."""
    return {
        "user_id": "123456789012345678",
        "user_name": "TestUser",
        "bot_id": "987654321098765432",
        "guild_id": "111222333444555666",
        "l3_facts": "<user:123456789012345678>님은 게임을 좋아한다.",
        "l2_summary": "# 게임 이야기\n\n편안한 분위기에서 게임에 대해 이야기 중",
        "l1_recent": [
            {"role": "user", "content": "안녕!", "user_name": "TestUser"}
        ]
    }


@pytest.fixture
def invalid_context():
    """Invalid context missing required keys."""
    return {
        "l3_facts": "some facts",
        # Missing user_id and user_name
    }
