"""
Text Processing Utilities

Contains functions for text compression, extraction, and normalization.
"""

import re
from typing import List

# Precompiled regex patterns for performance
_WHITESPACE_PATTERN = re.compile(r'\s+')
_BOT_RESPONSE_PATTERN = re.compile(r"(ASSISTANT|Bot):\s*.*")


def ultra_slim_extract(
    text: str, 
    trigger_words: List[str], 
    window_size: int = 6, 
    max_final_len: int = 50
) -> str:
    """
    Extract text segments around trigger words with minimal window size.
    
    Useful for reducing long messages while preserving context around 
    important keywords (like bot mentions).
    
    Args:
        text: Input text to process
        trigger_words: List of words to search for
        window_size: Characters to include before/after each match
        max_final_len: Maximum length of the final result
        
    Returns:
        Compressed text with segments around trigger words
    """
    text = _WHITESPACE_PATTERN.sub(' ', text).strip()
    found_segments: List[List[int]] = []
    
    for word in trigger_words:
        pattern = re.escape(word)
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_idx = max(0, match.start() - window_size)
            end_idx = min(len(text), match.end() + window_size)
            found_segments.append([start_idx, end_idx])
    
    if not found_segments:
        return text[:max_final_len]

    # Merge overlapping segments
    found_segments.sort()
    merged: List[List[int]] = []
    for segment in found_segments:
        if not merged or merged[-1][1] < segment[0]:
            merged.append(segment)
        else:
            merged[-1][1] = max(merged[-1][1], segment[1])
    
    # Collect chunks up to max length
    result_chunks: List[str] = []
    current_len = 0
    for start, end in merged:
        chunk = text[start:end].strip()
        if current_len + len(chunk) > max_final_len:
            break
        result_chunks.append(chunk)
        current_len += len(chunk)
    
    return " .. ".join(result_chunks)


def math_style_compress(text: str, threshold: int = 3) -> str:
    """
    Compress repeated word patterns using mathematical notation.
    
    Example: "hello hello hello hello" -> "(hello) x 4"
    
    Args:
        text: Input text to compress
        threshold: Minimum consecutive repetitions to trigger compression
        
    Returns:
        Compressed text with repeated words in (word) x N format
    """
    pattern = r'\b(\w+)\b(?:\s+\1){' + str(threshold - 1) + r',}'
    
    def replace_func(match: re.Match) -> str:
        word = match.group(1)
        count = len(re.findall(r'\b' + re.escape(word) + r'\b', match.group(0)))
        return f"({word}) x {count}"

    return re.sub(pattern, replace_func, text)


def normalize_for_bert(text: str) -> str:
    """
    Normalize text for BERT classifier input.
    
    - Masks bot responses to prevent data leakage
    - Normalizes bot name variations
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text suitable for BERT classification
    """
    # 1. Mask bot responses: "ASSISTANT: Hello" -> "[BOT_RESPONSE]"
    text = _BOT_RESPONSE_PATTERN.sub(r"[BOT_RESPONSE]", text)
    
    # 2. Remove user role prefixes: "USER: Hello" -> "Hello" / "SapoKR: Hi" -> "Hi"
    # Matches "Word: " at the start of a line
    text = re.sub(r"^[^:\n]+:\s*", "", text, flags=re.MULTILINE)
    
    # 3. Replace newlines with [SEP]
    text = text.replace("\n", " [SEP] ")
    
    # 4. Normalize bot names
    return text.replace("누리레느", "누리").replace("누리", "봇")


def replace_user_handles(text: str) -> str:
    """
    Replace various user ID patterns with Discord mention format.
    
    Handles LLM output variations:
      - <user:123456789> → <@123456789>   (standard)
      - [user:123456789] → <@123456789>   (bracket variant)
      - [123456789]      → <@123456789>   (bare ID in brackets)
      - (user:123456789) → <@123456789>   (paren variant)
    
    Discord user IDs are 17-20 digits, so we use that to avoid
    false positives with short numbers like [123].
    """
    # Pattern 1: <user:ID> or [user:ID] or (user:ID)
    text = re.sub(r'[<\[\(]user:(\d{17,20})[>\]\)]', r'<@\1>', text)
    
    # Pattern 2: [ID] — bare ID in brackets (only 17-20 digit IDs)
    text = re.sub(r'\[(\d{17,20})\]', r'<@\1>', text)
    
    return text
