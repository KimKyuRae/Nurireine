
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


def extract_search_keywords(text: str) -> str:
    """
    Extract key concepts/keywords from user input for L3 memory search.
    
    This function is used as a fallback when the SLM doesn't provide a search_query.
    It removes call names, question words, and common verbs to extract the core topic.
    
    Args:
        text: User input text (should be pre-cleaned with clean_query_text)
        
    Returns:
        Extracted keywords/concepts suitable for semantic search
    """
    if not text:
        return ""
    
    # Call names that should be removed (Nurireine's nicknames)
    call_names = ["누리레느", "누리레인", "누리야", "누리", "레느"]
    
    # Start with the cleaned text
    result = text
    
    # 1. Remove call names with optional particles
    for name in call_names:
        # Remove call name + optional particles (야, 아, 에게, 한테, 이, 가 등)
        pattern = rf'\b{re.escape(name)}(?:야|아|에게|한테|이|가|을|를|은|는)?\b'
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # 2. Remove common question words that don't add semantic value
    question_words = [
        r'뭐야', r'뭐', r'무엇', r'무슨',
        r'어떻게', r'어떤', r'어디', r'언제', r'왜',
        r'누구', r'어느', r'몇', r'였지', r'였어'
    ]
    for word in question_words:
        result = re.sub(rf'\b{word}\b', '', result, flags=re.IGNORECASE)
    
    # 3. Remove common request/question verbs that don't help with search
    request_verbs = [
        r'기억[하해]?[냐니]?[함]?', r'말해[줘]?', r'알려[줘]?',
        r'보여[줘]?', r'찾아[줘]?', r'검색해[줘]?',
        r'가르쳐[줘]?', r'설명해[줘]?', r'있[냐니어]?',
        r'했[냐니어]?', r'했더라', r'라고'
    ]
    for verb in request_verbs:
        result = re.sub(rf'\b{verb}\b', '', result, flags=re.IGNORECASE)
    
    # 4. Remove question marks and exclamation marks
    result = re.sub(r'[?!]+', '', result)
    
    # 5. Remove trailing particles from words
    # Korean particles that are often attached to nouns: 가, 이, 을, 를, 은, 는, 로, 의
    # We'll strip them from the end of words (but not standalone)
    particle_suffixes = ['가', '이', '을', '를', '은', '는', '로', '의']
    particles_set = {'가', '이', '을', '를', '은', '는', '에', '에서', '에게', '한테', 
                     '와', '과', '도', '만', '부터', '까지', '의', '로'}
    words = result.split()
    cleaned_words = []
    for word in words:
        # Skip single-character particles entirely
        if len(word) == 1 and word in particles_set:
            continue
        # If word is longer and ends with a particle, remove it
        if len(word) > 1:
            stripped = False
            for suffix in particle_suffixes:
                if word.endswith(suffix):
                    cleaned_words.append(word[:-len(suffix)])
                    stripped = True
                    break
            if not stripped:
                cleaned_words.append(word)
        else:
            # Single char but not a particle
            cleaned_words.append(word)
    result = ' '.join(cleaned_words)
    
    # 6. Remove standalone particles (after stripping from words)
    particles = [
        '가', '이', '을', '를', '은', '는',
        '에', '에서', '에게', '한테', '와', '과',
        '도', '만', '부터', '까지', '의', '로'
    ]
    words = result.split()
    words = [w for w in words if w not in particles]
    result = ' '.join(words)
    
    # 7. Clean up multiple spaces
    result = ' '.join(result.split())
    
    return result.strip()
