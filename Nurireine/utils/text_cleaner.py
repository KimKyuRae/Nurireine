
import re

def clean_query_text(text: str) -> str:
    """
    Clean text for use as a search query.
    
    Removes:
    - Discord mentions (<@...>)
    - Reply headers ([User's '...' reply])
    - Special characters/brackets
    - Empty/whitespace-only content
    
    Args:
        text: Input raw text
        
    Returns:
        Cleaned text string, or empty string if nothing left
    """
    if not text:
        return ""
    
    # 1. Remove reply headers
    # Format: [User의 '...'에 대한 답장]
    text = re.sub(r"\[.*?에 대한 답장\]", "", text)
    
    # 2. Remove Discord mentions (User, Role, Channel)
    text = re.sub(r"<@[!&]?\d+>|<#\d+>", "", text)
    
    # 3. Remove custom emojis <:name:id> or <a:name:id>
    text = re.sub(r"<a?:[a-zA-Z0-9_]+:\d+>", "", text)
    
    # 4. Remove excess whitespace
    text = " ".join(text.split())
    
    return text.strip()
