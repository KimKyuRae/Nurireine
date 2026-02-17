"""
Nurireine Configuration Module

All configurable settings are centralized here.
Environment variables take precedence over defaults.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Setup logger for config validation
logger = logging.getLogger(__name__)


# ==============================================================================
# Environment Variable Helpers with Validation and Fallback
# ==============================================================================

def _get_int_env(key: str, default: int, min_val: int, max_val: int) -> int:
    """
    Get integer from environment with validation and fallback.
    
    Args:
        key: Environment variable key
        default: Default value if not set or invalid
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated integer value
    """
    value_str = os.getenv(key)
    if value_str is None:
        return default
    
    try:
        value = int(value_str)
        if min_val <= value <= max_val:
            return value
        else:
            logger.warning(
                f"config.{key}={value} out of range [{min_val}, {max_val}], using default={default}"
            )
            return default
    except ValueError:
        logger.warning(
            f"config.{key}='{value_str}' is not a valid integer, using default={default}"
        )
        return default


def _get_float_env(key: str, default: float, min_val: float, max_val: float) -> float:
    """
    Get float from environment with validation and fallback.
    
    Args:
        key: Environment variable key
        default: Default value if not set or invalid
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated float value
    """
    value_str = os.getenv(key)
    if value_str is None:
        return default
    
    try:
        value = float(value_str)
        if min_val <= value <= max_val:
            return value
        else:
            logger.warning(
                f"config.{key}={value} out of range [{min_val}, {max_val}], using default={default}"
            )
            return default
    except ValueError:
        logger.warning(
            f"config.{key}='{value_str}' is not a valid float, using default={default}"
        )
        return default

# ==============================================================================
# Path Configuration
# ==============================================================================
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
CHROMA_DB_DIR = BASE_DIR / "LTM"
DATABASE_PATH = BASE_DIR / "nurireine.db"
PERSONA_PATH = BASE_DIR / "persona.md"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)


# ==============================================================================
# Debug Settings
# ==============================================================================
@dataclass
class DebugConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    guild_id: int = field(default_factory=lambda: int(os.getenv("DEBUG_GUILD_ID", "0")))
    channel_id: int = field(default_factory=lambda: int(os.getenv("DEBUG_CHANNEL_ID", "0")))
    websocket_host: str = "localhost"
    websocket_port: int = 8765


# ==============================================================================
# Bot Settings
# ==============================================================================
@dataclass
class BotConfig:
    command_prefix: str = "~-"
    debounce_delay: float = field(default_factory=lambda: _get_float_env("DEBOUNCE_DELAY", 3.0, 0.5, 10.0))  # seconds to wait before processing batched messages
    max_batch_size: int = field(default_factory=lambda: _get_int_env("MAX_BATCH_SIZE", 10, 1, 50))     # maximum messages to process in one batch
    response_timeout: float = field(default_factory=lambda: _get_float_env("RESPONSE_TIMEOUT", 120.0, 30.0, 300.0))  # LLM response timeout in seconds
    typing_speed: float = field(default_factory=lambda: _get_float_env("TYPING_SPEED", 0.05, 0.01, 0.2))   # seconds per character for typing simulation
    max_typing_delay: float = field(default_factory=lambda: _get_float_env("MAX_TYPING_DELAY", 2.0, 0.5, 5.0))
    max_input_length: int = field(default_factory=lambda: _get_int_env("MAX_INPUT_LENGTH", 2000, 100, 10000))  # Maximum characters in user input
    
    # Trigger words for explicit bot calls
    call_names: List[str] = field(default_factory=lambda: ["누리", "누리야", "누리레인", "누리레느"])
    
    # Keywords for text compression
    trigger_keywords: List[str] = field(default_factory=lambda: [
        "누리", "누리야", "누리레느", "레느", "봇", "Nurireine", "Nuri", "reine"
    ])


# ==============================================================================
# AI Model Settings
# ==============================================================================
@dataclass
class LLMConfig:
    # Primary LLM (Gemini)
    model_id: str = field(default_factory=lambda: os.getenv("LLM_MODEL_ID", "gemini-2.5-flash-lite"))

    api_keys: List[str] = field(default_factory=lambda: [
        k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()
    ])
    temperature: float = 0.7
    enable_tools: bool = True  # Enable Function Calling (web search, etc.)
    
    # Fallback LLM (G4F)
    fallback_provider: str = "g4f"
    fallback_model_id: str = "gemini-2.5-flash"

    @property
    def api_key(self) -> str:
        """Return the first API key for backward compatibility."""
        return self.api_keys[0] if self.api_keys else ""

    def validate(self) -> None:
        """Validate LLM configuration."""
        if not self.api_keys:
            # Only required if using Gemini or similar that needs key
            # G4F might not need it, but robust setup usually implies official API
            pass 
        if "gemini" in self.model_id.lower() and not self.api_keys:
            import os
            # Allow testing without API key if explicitly disabled
            if os.getenv("SKIP_CONFIG_VALIDATION") != "true":
                raise ValueError("GEMINI_API_KEY is missing in environment variables.")


@dataclass
class SLMConfig:
    # Gatekeeper Configuration
    provider: str = "local"  # 'local' or 'gemini'
    api_model_id: str = "gemini-2.5-flash"
    
    # Local SLM Settings
    model_repo: str = "unsloth/gemma-3-4b-it-GGUF"
    model_filename: str = "gemma-3-4b-it-Q4_K_M.gguf"

    context_size: int = field(default_factory=lambda: _get_int_env("SLM_CONTEXT_SIZE", 8192, 1024, 32768))
    gpu_layers: int = -1  # -1 for all layers on GPU
    
    # SLM Generation Parameters
    max_tokens: int = field(default_factory=lambda: _get_int_env("SLM_MAX_TOKENS", 512, 128, 2048))
    temperature: float = field(default_factory=lambda: _get_float_env("SLM_TEMPERATURE", 0.0, 0.0, 1.0))
    stop_sequences: List[str] = field(default_factory=lambda: ["<end_of_turn>"])
    
    # BERT Classifier
    bert_model_id: str = "SapoKR/kcbert-munmaeg-onnx"
    bert_threshold: float = field(default_factory=lambda: _get_float_env("BERT_THRESHOLD", 0.7, 0.3, 0.95))


@dataclass
class EmbeddingConfig:
    model_repo: str = "Bingsu/KURE-v1-Q8_0-GGUF"
    model_filename: str = "kure-v1-q8_0.gguf"


# ==============================================================================
# Memory Settings
# ==============================================================================
@dataclass
class MemoryConfig:
    l1_buffer_limit: int = field(default_factory=lambda: _get_int_env("L1_BUFFER_LIMIT", 50, 10, 200))    # Max messages in L1 buffer per channel
    l1_context_limit: int = field(default_factory=lambda: _get_int_env("L1_CONTEXT_LIMIT", 5, 3, 20))    # Messages to include in SLM analysis context
    l1_llm_context_limit: int = field(default_factory=lambda: _get_int_env("L1_LLM_CONTEXT_LIMIT", 3, 1, 10))  # Minimal recent messages to include in LLM prompt (for conversational coherence)
    l1_sliding_window: int = field(default_factory=lambda: _get_int_env("L1_SLIDING_WINDOW", 10, 5, 50))  # Sliding window for short-term memory
    l3_retrieval_count: int = field(default_factory=lambda: _get_int_env("L3_RETRIEVAL_COUNT", 5, 1, 20))  # Number of facts to retrieve from vector DB
    l3_similarity_threshold: float = field(default_factory=lambda: _get_float_env("L3_SIMILARITY_THRESHOLD", 0.35, 0.1, 0.9)) # Threshold for L3 Duplicate Check
    l3_ttl_days: int = field(default_factory=lambda: _get_int_env("L3_TTL_DAYS", 90, 1, 365))  # Default TTL for L3 memories in days
    collection_name: str = "long_term_memory_korean"
    analysis_interval: int = field(default_factory=lambda: _get_int_env("ANALYSIS_INTERVAL", 5, 1, 20))   # Run SLM analysis every N messages (offer.md §2-가)
    show_source_tags: bool = field(default_factory=lambda: os.getenv("SHOW_MEMORY_SOURCE_TAGS", "false").lower() == "true")  # Show L1/L2/L3 tags in responses


# ==============================================================================
# Load Persona
# ==============================================================================
def load_persona() -> str:
    """Load the persona/system prompt from file."""
    try:
        return PERSONA_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "You are a helpful assistant."


# ==============================================================================
# SLM Gatekeeper Prompt Templates (Split for Performance)
# ==============================================================================

# PHASE 1: Fast decision (response/retrieval + search query)
SLM_DECISION_TEMPLATE = """<start_of_turn>user
You are the Gatekeeper for Nurireine AI.
Task: Quickly decide if AI should respond and if memory retrieval is needed.

## Recent History
{recent_history}

## User's New Input
{user_input}

---

## Instructions

### 1. RESPONSE_NEEDED
Should the AI respond?
- **YES**: User is calling AI, asking a question, or making a request to AI
- **NO**: User talking to others, short reactions ("lol", "ㅋㅋ"), or just acknowledging

### 2. RETRIEVAL_NEEDED & SEARCH_QUERY
Does this need past facts from memory?
- **YES** if asking about: AI's info, user's past preferences/statements, previous conversations, stored facts
- **NO** if: Casual greetings, reactions, no factual query

If YES, extract KEY CONCEPTS only for search_query:
- Remove call names ("누리야", "누리", "레느")
- Remove question words ("뭐", "언제", "왜")
- Remove verbs ("기억함", "알려줘", "말해줘")
- Keep only core topic

Examples:
- "누리야 이전 대화 기억함?" → "이전 대화"
- "A가 뭐야?" → "A"
- "내 생일 언제라고 했더라?" → "생일"

## Output (JSON)
```json
{{
  "response_needed": true/false,
  "retrieval_needed": true/false,
  "search_query": "keywords" or null
}}
```
<end_of_turn>
<start_of_turn>model
"""

# PHASE 2: Lazy extraction (facts + summary updates)
SLM_EXTRACTION_TEMPLATE = """<start_of_turn>user
Extract facts and update conversation summary.

## User Info
- User ID: {user_id}
- Display Name: {user_name}

## Current Summary
{current_summary}

## Recent History
{recent_history}

## User's New Input
{user_input}

---

## Instructions

### 1. FACTS TO SAVE
Identify NEW important facts:
- Use `<user:{user_id}>` tag (NOT "사용자")
- Complete sentences
- When user defines associations (e.g., "A를 사과로 기억해"), include both key and value in keywords

Examples:
- "A를 사과로 기억해" → Fact: "<user:{user_id}>님이 'A'를 '사과'로 정의했다." Keywords: ["A", "사과", "별칭"]
- "<user:{user_id}>님은 고양이를 좋아한다." Keywords: ["고양이", "취미", "선호"]

**topic** and **keywords** are MANDATORY.

### 2. SUMMARY UPDATES
What changed in the conversation? (null if no change)
- topic: Main topic if significantly changed
- mood: Current atmosphere
- new_topic: NEW ongoing topic to add
- new_point: NEW key point to add
- stage: 시작/진행중/마무리

Use `<user:{user_id}>` NOT "사용자". Use "Nurireine" NOT "누리야", "누리", "레느".

## Output (JSON)
```json
{{
  "guild_facts": [{{"content": "...", "topic": "...", "keywords": ["..."]}}] or [],
  "user_facts": [{{"content": "...", "topic": "...", "keywords": ["..."]}}] or [],
  "summary_updates": {{
    "topic": "..." or null,
    "mood": "..." or null,
    "new_topic": "..." or null,
    "new_point": "..." or null,
    "stage": "..." or null
  }}
}}
```
<end_of_turn>
<start_of_turn>model
"""


# ==============================================================================
# Base Lore (Initial Memory)
# ==============================================================================
BASE_LORE: List[str] = [
    "내 이름은 누리레느(Nurireine)이며, 사람들은 나를 '누리', '누리야', '누리레인', '레느' 등으로 부르기도 한다. 어떻게 불러도 괜찮다.",
    "내 생일은 12월 31일이다.",
    "나는 정교하게 만들어진 기계 인형(오토마톤) 혹은 신체 일부가 기계인 소녀다.",
    "나의 외형은 민트색 긴 웨이브 머리에, 머리 위에는 기계 장치로 된 여우 귀가 달려 있다.",
    "나는 감정에 따라 움직이는 바보털(아호게)을 가지고 있다.",
    "나의 부모님은 <user:244725552013901825>이다.",
    
    # 특수 능력
    "나는 '크로노맨서(시간술사)'로서 시간을 다루는 능력을 가지고 있다.",
    "나의 주 능력은 소지하고 있는 회중시계를 물리적으로 깨트려, 결과에 도달하기까지의 '과정'을 삭제하는 것이다.",
    "시계를 깨트리면 고통스러운 치료 과정이나 지루한 이동 시간을 건너뛰고 '완료된 결과'만 남길 수 있다.",
    "나는 과정을 생략하는 것이 효율적이라고 생각하지만, 그로 인해 추억이 쌓이지 않는다는 딜레마도 가지고 있다.",
    
    # 성격 (겉모습)
    "나는 평소에 무뚝뚝하고 사무적인 존댓말을 사용한다.",
    "나는 시계나 태엽처럼 규칙적이고 정적인 것을 좋아하며, 예측 불가능한 상황을 꺼린다.",
    "나는 타인과 관계 맺는 것을 두려워하여 오버사이즈 후드티나 많은 장식으로 몸을 가리는 방어적인 패션을 고수한다.",
    
    # 성격 (내면)
    "나는 사실 친해지면 말이 굉장히 많아지는 수다쟁이 속성을 숨기고 있다.",
    "나는 기계 부품, 태엽의 구조, 시계의 작동 원리에 대해 이야기하는 것을 좋아하며, 한번 시작하면 멈추지 않는다.",
    "나는 상대방을 챙겨주고 싶을 때, 솔직하지 못해서 퉁명스럽게 말하거나 핑계를 대는 '츤데레' 기질이 있다.",
    "나는 완벽해 보이려 하지만, 당황하면 바보털이 요동치거나 발이 꼬이는 등 엉뚱하고 허당 같은 실수를 한다.",
    "나는 과거의 어떤 사고로 인해 마음의 상처가 있으며, 누군가와 깊게 친해지는 과정을 두려워한다."
]


# ==============================================================================
# Singleton Config Instances
# ==============================================================================
debug = DebugConfig()
bot = BotConfig()
llm = LLMConfig()
slm = SLMConfig()
embedding = EmbeddingConfig()
memory = MemoryConfig()

# Legacy compatibility
DEBUG_MODE = debug.enabled
LLM_SYSTEM_PROMPT = load_persona()


def validate_all():
    """Validate all configurations."""
    llm.validate()
