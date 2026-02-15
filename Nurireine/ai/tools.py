"""
Tool System for Nurireine

Provides extensible tool/function calling capabilities for the LLM.
Currently supports:
- Web Search (via DDGS)
- GitHub Search (Users & Repositories, with web search fallback)
- YouTube Search (via DDGS videos)
- Current Date/Time

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
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

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


def search_memory(query: str) -> str:
    """Search long-term memory (L3) for relevant facts."""
    if not _memory_manager:
        return "장기 기억 시스템이 비활성화되어 있습니다."
    try:
        result = _memory_manager.retrieve_facts(
            query, 
            guild_id=_current_guild_id, 
            user_id=_current_user_id
        )
        return result if result else "관련된 기억이 없습니다."
    except Exception as e:
        logger.error(f"search_memory error: {e}")
        return f"기억 검색 중 오류가 발생했습니다: {e}"


def get_chat_history(limit: int = 10) -> str:
    """Retrieve recent chat history (L1 buffer) for current channel."""
    if not _memory_manager:
        return "대화 이력 시스템이 비활성화되어 있습니다."
    if not _current_channel_id:
        return "현재 채널 정보가 없습니다."
    try:
        from .. import config
        # Clamp limit to a reasonable range
        limit = max(1, min(limit, config.memory.l1_buffer_limit))
        buffer = _memory_manager.get_l1_buffer(_current_channel_id)
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


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: Dict[str, Any] = {
    "web_search": web_search,
    "github_search": github_search,
    "youtube_search": youtube_search,
    "get_current_time": get_current_time,
    "search_memory": search_memory,
    "get_chat_history": get_chat_history,
}


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a registered tool and return the result."""
    func = TOOL_REGISTRY.get(name)
    if not func:
        logger.warning(f"Unknown tool requested: {name}")
        return {"error": f"Unknown tool: {name}"}
    
    try:
        broadcast_event("tool_call", {"stage": "start", "tool": name, "args": args})
        logger.info(f"Executing tool: {name}({args})")
        
        result = func(**args)
        
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
        types.FunctionDeclaration(
            name="web_search",
            description=(
                "인터넷에서 최신 정보를 검색합니다. "
                "사용자가 최근 뉴스, 실시간 정보, 모르는 사실, "
                "또는 최신 데이터가 필요한 질문을 할 때 사용하세요. "
                "자신(누리레느)에 대한 질문이나 일상 대화에는 사용하지 마세요. "
                "이 도구를 호출할 때 사전 안내 멘트(예: '잠시만요', '찾아볼게요')를 하지 마시오. 최종 결과만 말하세요."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="검색할 키워드 또는 문장 (주제에 맞는 언어로 작성)"
                    ),
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="github_search",
            description=(
                "GitHub에서 사용자(개발자) 또는 레포지토리(오픈소스 프로젝트)를 검색합니다. "
                "특정 개발자의 프로필, 오픈소스 프로젝트, 프로그래밍 관련 검색에 사용하세요. "
                "이 도구를 호출할 때 사전 안내 멘트를 하지 마시오. 최종 결과만 말하세요."
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
                        description="검색 유형: 'users' (사용자/개발자) 또는 'repositories' (레포지토리/프로젝트). 기본값: 'users'",
                        enum=["users", "repositories"]
                    ),
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="youtube_search",
            description=(
                "YouTube 및 기타 영상 플랫폼에서 영상을 검색합니다. "
                "사용자가 영상, 강좌, 뮤직비디오, 또는 영상 콘텐츠를 찾을 때 사용하세요. "
                "이 도구를 호출할 때 사전 안내 멘트를 하지 마시오. 최종 결과만 말하세요."
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
            name="get_current_time",
            description=(
                "현재 날짜와 시간을 확인합니다. "
                "사용자가 '오늘 며칠이야?', '지금 몇 시야?', '무슨 요일이야?' 등 "
                "날짜나 시간에 관한 질문을 할 때 반드시 이 도구를 사용하세요. "
                "절대로 추측하지 마세요. "
                "이 도구를 호출할 때 사전 안내 멘트를 하지 마시오."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            )
        ),
        types.FunctionDeclaration(
            name="search_memory",
            description=(
                "장기 기억(L3)에서 관련 정보를 검색합니다. "
                "사용자가 과거에 대한 질문, 자신의 설정/배경/생일 등에 대한 질문, "
                "또는 이전에 저장된 사실이 필요할 때 사용하세요. "
                "이 도구를 호출할 때 사전 안내 멘트를 하지 마시오. 최종 결과만 말하세요."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="검색할 키워드 또는 질문 (한국어로 작성)"
                    ),
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="get_chat_history",
            description=(
                "현재 채널의 최근 대화 이력을 가져옵니다. "
                "사용자가 '방금 뭐라고 했어?', '아까 말한 거 뭐야?' 등 "
                "구체적인 최근 대화 내용이나 정확한 문구를 물어볼 때만 사용하세요. "
                "일상 대화에서는 사용하지 마세요. "
                "이 도구를 호출할 때 사전 안내 멘트를 하지 마시오."
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
