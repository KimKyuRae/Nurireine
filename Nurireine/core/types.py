from dataclasses import dataclass, field
from typing import Optional, List, Any
from datetime import datetime

@dataclass
class UserInfo:
    id: str
    name: str

@dataclass
class MessageData:
    content: str
    author: UserInfo
    channel_id: int
    guild_id: Optional[int]
    timestamp: float
    reply_to: Optional['MessageData'] = None
    
    @property
    def is_empty(self) -> bool:
        return not self.content or not self.content.strip()

@dataclass
class ConversationContext:
    channel_id: int
    guild_id: Optional[str]
    user: UserInfo
    l2_summary: str = ""
    l3_facts: str = ""
    l1_recent: List[Any] = field(default_factory=list)
