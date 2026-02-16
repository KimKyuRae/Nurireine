"""
Tool System for Nurireine

Provides extensible tool/function calling capabilities for the LLM.

Tool Categories:
1. Search Tools: web_search, github_search, youtube_search, news_search, image_search
2. Utility Tools: get_current_time, calculate
3. Translation Tools: translate_text
4. Memory Tools: search_memory, get_chat_history

To add a new tool:
1. Create a function in this file
2. Add it to TOOL_REGISTRY
3. Add a FunctionDeclaration in get_tool_declarations()
"""

import json
import logging
import urllib.request
import urllib.parse
import urllib.error
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Awaitable

from google.genai import types

from ..debug_server import broadcast_event

logger = logging.getLogger(__name__)

# Module-level holder for MemoryManager reference (set by MainLLM at runtime)
_memory_manager = None
_current_channel_id = None
_current_guild_id = None
_current_user_id = None


def set_tool_context(memory_manager, channel_id=None, guild_id=None, user_id=None):
    """Set the runtime context for memory-related tools.
    Called by MainLLM before each generation cycle."""
    global _memory_manager, _current_channel_id, _current_guild_id, _current_user_id
    _memory_manager = memory_manager
    _current_channel_id = channel_id
    _current_guild_id = guild_id
    _current_user_id = user_id

# Korea Standard Time (UTC+9)
KST = timezone(timedelta(hours=9))


# =============================================================================
# Helper
# =============================================================================

def _get_ddgs():
    """Import and return DDGS class from the best available package."""
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS
        return DDGS


# =============================================================================
# Tool Implementations
# =============================================================================

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DDGS."""
    try:
        DDGS = _get_ddgs()
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return "검색 결과가 없습니다."
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get('title', '제목 없음')
            body = r.get('body', '내용 없음')
            href = r.get('href', '')
            formatted.append(f"{i}. {title}\n   {body}\n   출처: {href}")
        
        return "\n\n".join(formatted)
    
    except ImportError:
        return "검색 기능을 사용할 수 없습니다. (pip install ddgs)"
    except Exception as e:
        logger.error(f"Web search error: {type(e).__name__}: {e}")
        return f"검색 중 오류가 발생했습니다: {e}"


def github_search(query: str, search_type: str = "users", max_results: int = 5) -> str:
    """
    Search GitHub for users or repositories.
    Uses GitHub REST API with web search fallback.
    """
    search_type = (search_type or "users").lower()
    if search_type not in ("users", "repositories"):
        search_type = "users"
    
    # Try GitHub API first
    try:
        result = _github_api_search(query, search_type, max_results)
        if result:
            return result
    except Exception as e:
        logger.warning(f"GitHub API failed, falling back to web search: {e}")
    
    # Fallback: use web search scoped to GitHub
    return _github_web_fallback(query, search_type, max_results)


def _github_api_search(query: str, search_type: str, max_results: int) -> str:
    """Direct GitHub API search."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.github.com/search/{search_type}?q={encoded_query}&per_page={max_results}"
    
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Nurireine-Bot"
    })
    
    response = urllib.request.urlopen(req, timeout=8)
    data = json.loads(response.read().decode('utf-8'))
    items = data.get("items", [])
    
    if not items:
        return ""
    
    total_count = data.get("total_count", 0)
    formatted = [f"GitHub 검색 결과 (총 {total_count}건 중 상위 {len(items)}건):"]
    
    if search_type == "users":
        for i, user in enumerate(items, 1):
            login = user.get("login", "")
            profile_url = user.get("html_url", "")
            user_type = user.get("type", "User")
            
            # Get user details (bio, followers, etc.)
            detail = _github_get_user_detail(login)
            
            if detail:
                name = detail.get("name", "") or login
                bio = detail.get("bio", "") or ""
                followers = detail.get("followers", 0)
                repos_count = detail.get("public_repos", 0)
                location = detail.get("location", "") or ""
                company = detail.get("company", "") or ""
                
                entry = f"\n{i}. {name} (@{login})"
                info_parts = [f"팔로워: {followers}", f"공개 레포: {repos_count}"]
                if location:
                    info_parts.append(f"위치: {location}")
                if company:
                    info_parts.append(f"소속: {company}")
                entry += f"\n   {' | '.join(info_parts)}"
                if bio:
                    entry += f"\n   소개: {bio[:200]}"
                entry += f"\n   프로필: {profile_url}"
            else:
                entry = f"\n{i}. @{login} ({user_type})\n   프로필: {profile_url}"
            
            # Fetch top repos for this user (only for first 2 users to avoid rate limits)
            if i <= 2:
                top_repos = _github_get_user_repos(login, top_n=5)
                if top_repos:
                    entry += f"\n   --- 대표 프로젝트 ---"
                    for j, repo in enumerate(top_repos, 1):
                        repo_name = repo.get("name", "")
                        repo_desc = (repo.get("description", "") or "설명 없음")[:100]
                        repo_stars = repo.get("stargazers_count", 0)
                        repo_lang = repo.get("language", "") or ""
                        repo_forks = repo.get("forks_count", 0)
                        entry += f"\n   {j}) {repo_name} — {repo_desc}"
                        entry += f"\n      ⭐ {repo_stars} | 🍴 {repo_forks}" + (f" | {repo_lang}" if repo_lang else "")
            
            formatted.append(entry)
    
    elif search_type == "repositories":
        for i, repo in enumerate(items, 1):
            name = repo.get("full_name", "")
            desc = repo.get("description", "") or "설명 없음"
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language", "") or "알 수 없음"
            forks = repo.get("forks_count", 0)
            url = repo.get("html_url", "")
            
            formatted.append(
                f"\n{i}. {name}"
                f"\n   {desc[:150]}"
                f"\n   ⭐ {stars} | 🍴 {forks} | 언어: {lang}"
                f"\n   링크: {url}"
            )
    
    return "\n".join(formatted)


def _github_get_user_detail(login: str) -> dict:
    """Fetch detailed user info from GitHub API. Returns None on failure."""
    try:
        url = f"https://api.github.com/users/{urllib.parse.quote(login)}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Nurireine-Bot"
        })
        response = urllib.request.urlopen(req, timeout=5)
        return json.loads(response.read().decode('utf-8'))
    except Exception:
        return None


def _github_get_user_repos(login: str, top_n: int = 5) -> list:
    """Fetch user's top repositories sorted by stars. Returns empty list on failure."""
    try:
        url = f"https://api.github.com/users/{urllib.parse.quote(login)}/repos?sort=stars&direction=desc&per_page={top_n}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Nurireine-Bot"
        })
        response = urllib.request.urlopen(req, timeout=5)
        repos = json.loads(response.read().decode('utf-8'))
        # Filter out forks to show only original work
        return [r for r in repos if not r.get("fork", False)][:top_n]
    except Exception:
        return []


def _github_web_fallback(query: str, search_type: str, max_results: int) -> str:
    """Fallback: search GitHub via web search engine."""
    try:
        DDGS = _get_ddgs()
        
        # Search both profile and repos for comprehensive results
        if search_type == "users":
            search_query = f"site:github.com {query}"
        else:
            search_query = f"site:github.com {query} repository stars"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=max_results + 3))
        
        if not results:
            return f"GitHub에서 '{query}'에 대한 검색 결과가 없습니다."
        
        formatted = [f"GitHub 검색 결과 (웹 검색, {len(results)}건):"]
        for i, r in enumerate(results, 1):
            title = r.get('title', '제목 없음')
            body = r.get('body', '')[:200]
            href = r.get('href', '')
            formatted.append(f"\n{i}. {title}\n   {body}\n   링크: {href}")
        
        return "\n".join(formatted)
    
    except Exception as e:
        logger.error(f"GitHub web fallback error: {e}")
        return f"GitHub 검색에 실패했습니다: {e}"


def youtube_search(query: str, max_results: int = 5) -> str:
    """Search YouTube videos using DDGS."""
    try:
        DDGS = _get_ddgs()
        
        with DDGS() as ddgs:
            results = list(ddgs.videos(query, max_results=max_results))
        
        if not results:
            return f"'{query}'에 대한 영상 검색 결과가 없습니다."
        
        formatted = [f"영상 검색 결과 ({len(results)}건):"]
        
        for i, v in enumerate(results, 1):
            title = v.get("title", "제목 없음")
            description = (v.get("description", "") or "")[:100]
            publisher = v.get("publisher", "")
            duration = v.get("duration", "")
            url = v.get("content", "") or v.get("href", "")
            
            info_parts = []
            if publisher:
                info_parts.append(f"채널: {publisher}")
            if duration:
                info_parts.append(f"길이: {duration}")
            info_line = " | ".join(info_parts)
            
            entry = f"\n{i}. {title}"
            if description:
                entry += f"\n   {description}"
            if info_line:
                entry += f"\n   {info_line}"
            if url:
                entry += f"\n   링크: {url}"
            formatted.append(entry)
        
        return "\n".join(formatted)
    
    except ImportError:
        return "영상 검색 기능을 사용할 수 없습니다. (pip install ddgs)"
    except Exception as e:
        logger.error(f"YouTube search error: {type(e).__name__}: {e}")
        return f"영상 검색 중 오류가 발생했습니다: {e}"


def get_current_time() -> str:
    """Get the current date and time in Korean Standard Time (KST)."""
    now = datetime.now(KST)

    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    weekday = weekdays[now.weekday()]
    
    return (
        f"현재 시각: {now.year}년 {now.month}월 {now.day}일 ({weekday}) "
        f"{now.hour:02d}시 {now.minute:02d}분 {now.second:02d}초 (KST)"
    )


async def search_memory(query: str) -> str:
    """Search long-term memory (L3) for relevant facts (Async)."""
    if not _memory_manager:
        return "장기 기억 시스템이 비활성화되어 있습니다."
    try:
        result = await _memory_manager.retrieve_facts(
            query, 
            guild_id=_current_guild_id, 
            user_id=_current_user_id
        )
        return result if result else "관련된 기억이 없습니다."
    except Exception as e:
        logger.error(f"search_memory error: {e}")
        return f"기억 검색 중 오류가 발생했습니다: {e}"


async def get_chat_history(limit: int = 10) -> str:
    """Retrieve recent chat history (L1 buffer) for current channel (Async)."""
    if not _memory_manager:
        return "대화 이력 시스템이 비활성화되어 있습니다."
    if not _current_channel_id:
        return "현재 채널 정보가 없습니다."
    try:
        from .. import config
        # Clamp limit to a reasonable range
        limit = max(1, min(limit, config.memory.l1_buffer_limit))
        buffer = await _memory_manager.get_l1_buffer(_current_channel_id)
        recent = buffer[-limit:]
        if not recent:
            return "최근 대화 기록이 없습니다."
        lines = []
        for msg in recent:
            role = msg.get('role', 'user')
            name = msg.get('user_name', role)
            content = msg.get('content', '')
            if role == 'assistant':
                name = 'Nurireine'
            lines.append(f"{name}: {content}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"get_chat_history error: {e}")
        return f"대화 이력 조회 중 오류가 발생했습니다: {e}"


def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.
    Supports basic arithmetic, powers, and common math functions.
    """
    try:
        # Safe evaluation - only allow math operations
        import ast
        import math
        import operator
        
        # Allowed operators and functions
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
        }
        
        allowed_functions = {
            'abs': abs,
            'round': round,
            'max': max,
            'min': min,
            'sum': sum,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
        }
        
        allowed_constants = {
            'pi': math.pi,
            'e': math.e,
        }
        
        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op = allowed_operators.get(type(node.op))
                if op:
                    return op(left, right)
                raise ValueError(f"지원하지 않는 연산자: {type(node.op).__name__}")
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                op = allowed_operators.get(type(node.op))
                if op:
                    return op(operand)
                raise ValueError(f"지원하지 않는 단항 연산자: {type(node.op).__name__}")
            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("지원하지 않는 함수 호출 형식입니다.")
                func_name = node.func.id
                if func_name not in allowed_functions:
                    raise ValueError(f"지원하지 않는 함수: {func_name}")
                args = [eval_node(arg) for arg in node.args]
                return allowed_functions[func_name](*args)
            elif isinstance(node, ast.Name):
                # Check constants first
                if node.id in allowed_constants:
                    return allowed_constants[node.id]
                raise ValueError(f"알 수 없는 변수: {node.id}")
            elif isinstance(node, ast.List):
                return [eval_node(item) for item in node.elts]
            else:
                raise ValueError(f"지원하지 않는 표현식 타입: {type(node).__name__}")
        
        # Parse and evaluate
        tree = ast.parse(expression, mode='eval')
        result = eval_node(tree.body)
        
        # Format result
        if isinstance(result, float):
            if result.is_integer():
                return f"계산 결과: {int(result)}"
            else:
                return f"계산 결과: {result:.10g}"
        else:
            return f"계산 결과: {result}"
    
    except SyntaxError:
        return f"수식 오류: 올바른 수식이 아닙니다. ({expression})"
    except ZeroDivisionError:
        return "계산 오류: 0으로 나눌 수 없습니다."
    except Exception as e:
        logger.error(f"Calculate error: {type(e).__name__}: {e}")
        return f"계산 중 오류가 발생했습니다: {e}"


def translate_text(text: str, target_language: str = "ko") -> str:
    """
    Translate text to a target language using web search fallback.
    """
    try:
        DDGS = _get_ddgs()
        
        # Build translation query
        if target_language.lower() in ['ko', 'korean', '한국어']:
            search_query = f"translate to Korean: {text}"
            target_lang_name = "한국어"
        elif target_language.lower() in ['en', 'english', '영어']:
            search_query = f"translate to English: {text}"
            target_lang_name = "영어"
        elif target_language.lower() in ['ja', 'japanese', '일본어']:
            search_query = f"translate to Japanese: {text}"
            target_lang_name = "일본어"
        elif target_language.lower() in ['zh', 'chinese', '중국어']:
            search_query = f"translate to Chinese: {text}"
            target_lang_name = "중국어"
        else:
            search_query = f"translate to {target_language}: {text}"
            target_lang_name = target_language
        
        # Use web search for translation results
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=3))
        
        if not results:
            return f"'{text}'의 번역 결과를 찾을 수 없습니다."
        
        # Extract translation from search results
        formatted = [f"번역 검색 결과 ({target_lang_name}):"]
        for i, r in enumerate(results, 1):
            body = r.get('body', '')
            if body:
                formatted.append(f"{i}. {body[:200]}")
        
        return "\n".join(formatted)
    
    except ImportError:
        return "번역 기능을 사용할 수 없습니다. (pip install ddgs)"
    except Exception as e:
        logger.error(f"Translation error: {type(e).__name__}: {e}")
        return f"번역 중 오류가 발생했습니다: {e}"


def news_search(query: str, max_results: int = 5) -> str:
    """Search for news articles using DDGS news search."""
    try:
        DDGS = _get_ddgs()
        
        with DDGS() as ddgs:
            # Use DDGS news search if available
            try:
                results = list(ddgs.news(query, max_results=max_results))
            except (AttributeError, TypeError):
                # Fallback to regular search with "news" keyword
                results = list(ddgs.text(f"{query} news", max_results=max_results))
        
        if not results:
            return f"'{query}'에 대한 뉴스 검색 결과가 없습니다."
        
        formatted = [f"뉴스 검색 결과 ({len(results)}건):"]
        
        for i, article in enumerate(results, 1):
            title = article.get('title', '제목 없음')
            body = article.get('body', '') or article.get('description', '')
            url = article.get('url', '') or article.get('href', '')
            date = article.get('date', '')
            source = article.get('source', '')
            
            entry = f"\n{i}. {title}"
            if body:
                entry += f"\n   {body[:200]}"
            
            info_parts = []
            if source:
                info_parts.append(f"출처: {source}")
            if date:
                info_parts.append(f"날짜: {date}")
            if info_parts:
                entry += f"\n   {' | '.join(info_parts)}"
            
            if url:
                entry += f"\n   링크: {url}"
            
            formatted.append(entry)
        
        return "\n".join(formatted)
    
    except ImportError:
        return "뉴스 검색 기능을 사용할 수 없습니다. (pip install ddgs)"
    except Exception as e:
        logger.error(f"News search error: {type(e).__name__}: {e}")
        return f"뉴스 검색 중 오류가 발생했습니다: {e}"


def image_search(query: str, max_results: int = 5) -> str:
    """Search for images using DDGS image search."""
    try:
        DDGS = _get_ddgs()
        
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))
        
        if not results:
            return f"'{query}'에 대한 이미지 검색 결과가 없습니다."
        
        formatted = [f"이미지 검색 결과 ({len(results)}건):"]
        
        for i, img in enumerate(results, 1):
            title = img.get('title', '제목 없음')
            url = img.get('image', '') or img.get('url', '')
            source = img.get('source', '')
            width = img.get('width', '')
            height = img.get('height', '')
            
            entry = f"\n{i}. {title}"
            
            info_parts = []
            if width and height:
                info_parts.append(f"크기: {width}x{height}")
            if source:
                info_parts.append(f"출처: {source}")
            if info_parts:
                entry += f"\n   {' | '.join(info_parts)}"
            
            if url:
                entry += f"\n   링크: {url}"
            
            formatted.append(entry)
        
        return "\n".join(formatted)
    
    except ImportError:
        return "이미지 검색 기능을 사용할 수 없습니다. (pip install ddgs)"
    except Exception as e:
        logger.error(f"Image search error: {type(e).__name__}: {e}")
        return f"이미지 검색 중 오류가 발생했습니다: {e}"


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: Dict[str, Any] = {
    # Search Tools
    "web_search": web_search,
    "github_search": github_search,
    "youtube_search": youtube_search,
    "news_search": news_search,
    "image_search": image_search,
    
    # Utility Tools
    "get_current_time": get_current_time,
    "calculate": calculate,
    
    # Translation Tools
    "translate_text": translate_text,
    
    # Memory Tools
    "search_memory": search_memory,
    "get_chat_history": get_chat_history,
}


async def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a registered tool and return the result (Async)."""
    func = TOOL_REGISTRY.get(name)
    if not func:
        logger.warning(f"Unknown tool requested: {name}")
        return {"error": f"Unknown tool: {name}"}
    
    try:
        broadcast_event("tool_call", {"stage": "start", "tool": name, "args": args})
        logger.info(f"Executing tool: {name}({args})")
        
        if asyncio.iscoroutinefunction(func):
            result = await func(**args)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: func(**args))
        
        broadcast_event("tool_call", {"stage": "end", "tool": name, "success": True})
        logger.info(f"Tool '{name}' completed successfully ({len(str(result))} chars)")
        logger.debug(f"Tool '{name}' result: {str(result)[:500]}")
        return {
            "result": result,
            "instruction": (
                "위 결과를 기반으로 사용자에게 자연스럽게 답변하세요. "
                "결과를 무시하거나 추측하지 마세요. "
                "'더 알려드릴까요?', '자세히 설명해드릴까요?' 같은 후속 질문은 절대 하지 마세요. "
                "한 번에 완결된 정보를 전달하세요."
            )
        }
    
    except Exception as e:
        broadcast_event("tool_call", {"stage": "end", "tool": name, "success": False, "error": str(e)})
        logger.error(f"Tool execution error ({name}): {e}")
        return {"error": str(e)}


# =============================================================================
# Gemini Tool Declarations
# =============================================================================

def get_tool_declarations() -> types.Tool:
    """Build Gemini-compatible tool declarations for all registered tools."""
    declarations = [
        # === Search Tools ===
        types.FunctionDeclaration(
            name="web_search",
            description=(
                "인터넷에서 최신 정보를 검색합니다. "
                "다음과 같은 경우 반드시 사용하세요:\n"
                "- 최근 뉴스, 사건, 이벤트\n"
                "- 실시간 정보 (날씨, 환율 등)\n"
                "- 사실 확인이 필요한 정보\n"
                "- 최신 데이터나 통계\n"
                "- 모르는 사실이나 개념 설명\n"
                "주의: 자신(누리레느)에 대한 질문이나 일상 대화에는 사용하지 마세요."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="검색할 키워드 또는 문장 (영어 또는 한국어, 주제에 맞는 언어 사용)"
                    ),
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="news_search",
            description=(
                "최신 뉴스 기사를 검색합니다. "
                "다음과 같은 경우 사용하세요:\n"
                "- 최근 뉴스나 사건\n"
                "- 언론 보도 내용\n"
                "- 시사 이슈\n"
                "일반 웹 검색보다 뉴스 전문 검색을 원할 때 사용합니다."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="검색할 뉴스 키워드"
                    ),
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="github_search",
            description=(
                "GitHub에서 개발자 또는 오픈소스 프로젝트를 검색합니다. "
                "다음과 같은 경우 사용하세요:\n"
                "- 특정 개발자의 프로필 찾기\n"
                "- 오픈소스 프로젝트 검색\n"
                "- 프로그래밍 라이브러리나 도구 찾기\n"
                "- GitHub 레포지토리 정보"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="검색할 GitHub 사용자명, 이름, 또는 프로젝트 키워드"
                    ),
                    "search_type": types.Schema(
                        type=types.Type.STRING,
                        description="검색 유형: 'users' (개발자) 또는 'repositories' (프로젝트). 기본값: 'users'",
                        enum=["users", "repositories"]
                    ),
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="youtube_search",
            description=(
                "YouTube에서 영상을 검색합니다. "
                "다음과 같은 경우 사용하세요:\n"
                "- 영상 콘텐츠 찾기\n"
                "- 강의나 튜토리얼\n"
                "- 뮤직비디오\n"
                "- 동영상 리뷰나 설명"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="검색할 영상 제목, 키워드, 또는 채널명"
                    ),
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="image_search",
            description=(
                "이미지를 검색합니다. "
                "사용자가 이미지, 사진, 그림 등을 찾을 때 사용하세요."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="검색할 이미지 키워드"
                    ),
                },
                required=["query"]
            )
        ),
        
        # === Utility Tools ===
        types.FunctionDeclaration(
            name="get_current_time",
            description=(
                "현재 날짜와 시간을 확인합니다. "
                "다음과 같은 경우 반드시 사용하세요:\n"
                "- '오늘 며칠이야?', '지금 몇 시야?'\n"
                "- '무슨 요일이야?'\n"
                "- 날짜나 시간 관련 질문\n"
                "주의: 시간 정보는 항상 이 도구로 확인하세요. 추측하지 마세요."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            )
        ),
        types.FunctionDeclaration(
            name="calculate",
            description=(
                "수학 계산을 수행합니다. "
                "다음과 같은 경우 반드시 사용하세요:\n"
                "- 사칙연산 (덧셈, 뺄셈, 곱셈, 나눗셈)\n"
                "- 거듭제곱, 제곱근\n"
                "- 삼각함수 (sin, cos, tan)\n"
                "- 로그 함수 (log, log10)\n"
                "지원 함수: abs, round, max, min, sum, sqrt, sin, cos, tan, log, log10, exp, pi, e\n"
                "예: '2 + 3', 'sqrt(16)', 'sin(pi/2)', '2**10'"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="계산할 수식 (예: '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)')"
                    ),
                },
                required=["expression"]
            )
        ),
        
        # === Translation Tools ===
        types.FunctionDeclaration(
            name="translate_text",
            description=(
                "텍스트를 다른 언어로 번역합니다. "
                "사용자가 번역을 요청할 때 사용하세요."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "text": types.Schema(
                        type=types.Type.STRING,
                        description="번역할 텍스트"
                    ),
                    "target_language": types.Schema(
                        type=types.Type.STRING,
                        description="목표 언어 (예: 'ko', 'en', 'ja', 'zh'). 기본값: 'ko'"
                    ),
                },
                required=["text"]
            )
        ),
        
        # === Memory Tools ===
        types.FunctionDeclaration(
            name="search_memory",
            description=(
                "장기 기억(L3)에서 관련 정보를 검색합니다. "
                "다음과 같은 경우 사용하세요:\n"
                "- 사용자의 과거 정보나 설정\n"
                "- 사용자의 선호도, 생일 등\n"
                "- 이전 대화에서 저장된 사실\n"
                "- '내 ~이 뭐였지?' 같은 질문"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="검색할 키워드 또는 질문"
                    ),
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="get_chat_history",
            description=(
                "현재 채널의 최근 대화 이력을 가져옵니다. "
                "다음과 같은 경우만 사용하세요:\n"
                "- '방금 뭐라고 했어?'\n"
                "- '아까 말한 거 뭐야?'\n"
                "- 최근 대화의 정확한 내용 확인\n"
                "주의: 일반 대화에서는 사용하지 마세요. L1 버퍼는 이미 제공됩니다."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="가져올 메시지 수 (기본값: 10, 최대: 50)"
                    ),
                },
            )
        ),
    ]
    
    return types.Tool(function_declarations=declarations)
