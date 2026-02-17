"""
Input Validation Module

Provides validation utilities for user inputs, configuration, and data sanitization.

Requirements: Python 3.9+ (for PEP 585 type hints)
"""

import re
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Validates and sanitizes user inputs.
    
    Provides multiple validation methods for different input types.
    """
    
    # Allowed characters in user names (prevent injection in embeddings)
    USERNAME_PATTERN = re.compile(r'^[\w\s\-_가-힣ぁ-んァ-ヶー一-龯]+$', re.UNICODE)
    
    # Maximum lengths (loaded from config)
    MAX_USERNAME_LENGTH = 100
    MAX_CHANNEL_NAME_LENGTH = 100
    
    @staticmethod
    def _get_max_input_length() -> int:
        """Get MAX_INPUT_LENGTH from config to avoid circular import."""
        try:
            from . import config
            return config.bot.max_input_length
        except (ImportError, AttributeError):
            return 2000  # Fallback default
    
    @staticmethod
    def validate_message_content(content: str) -> Tuple[bool, Optional[str]]:
        """
        Validate message content.
        
        Args:
            content: Message content to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not content:
            return False, "Empty content"
        
        max_length = InputValidator._get_max_input_length()
        if len(content) > max_length:
            return False, f"Content too long ({len(content)} > {max_length})"
        
        # Check for null bytes
        if '\x00' in content:
            return False, "Content contains null bytes"
        
        return True, None
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, Optional[str]]:
        """
        Validate username for safe embedding storage.
        
        Args:
            username: Username to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not username:
            return False, "Empty username"
        
        if len(username) > InputValidator.MAX_USERNAME_LENGTH:
            return False, f"Username too long ({len(username)} > {InputValidator.MAX_USERNAME_LENGTH})"
        
        # Allow Unicode characters but prevent special control characters
        if not InputValidator.USERNAME_PATTERN.match(username):
            logger.warning(f"Username contains invalid characters: {username}")
            # Don't reject, just log - Discord allows many characters
        
        return True, None
    
    @staticmethod
    def sanitize_for_embedding(text: str) -> str:
        """
        Sanitize text for safe embedding.
        
        Removes control characters and normalizes whitespace.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Limit length
        if len(text) > 2000:
            text = text[:2000]
        
        return text
    
    @staticmethod
    def validate_user_id(user_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Discord user ID format.
        
        Args:
            user_id: User ID to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not user_id:
            return False, "Empty user ID"
        
        # Discord IDs are numeric strings
        if not user_id.isdigit():
            return False, "User ID must be numeric"
        
        # Discord IDs are typically 17-19 digits
        if len(user_id) < 17 or len(user_id) > 20:
            logger.warning(f"Unusual user ID length: {len(user_id)}")
        
        return True, None
    
    @staticmethod
    def validate_channel_id(channel_id: int) -> Tuple[bool, Optional[str]]:
        """
        Validate Discord channel ID.
        
        Args:
            channel_id: Channel ID to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not channel_id or channel_id <= 0:
            return False, "Invalid channel ID"
        
        # Discord IDs should be within a reasonable range
        if channel_id > 2**63:  # Max signed int64
            return False, "Channel ID out of range"
        
        return True, None
    
    @staticmethod
    def validate_context_dict(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate context dictionary for LLM input.
        
        Args:
            context: Context dictionary to validate
            
        Returns:
            (is_valid, error_message)
        """
        required_keys = ['user_id', 'user_name']
        
        for key in required_keys:
            if key not in context:
                return False, f"Missing required key: {key}"
        
        # Validate nested values
        if 'l3_facts' in context and not isinstance(context['l3_facts'], str):
            return False, "l3_facts must be a string"
        
        if 'l2_summary' in context and not isinstance(context['l2_summary'], str):
            return False, "l2_summary must be a string"
        
        if 'l1_recent' in context and not isinstance(context['l1_recent'], list):
            return False, "l1_recent must be a list"
        
        return True, None


class ConfigValidator:
    """
    Validates configuration values.
    """
    
    @staticmethod
    def validate_api_key(api_key: str, service: str = "API") -> Tuple[bool, Optional[str]]:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            service: Service name for error messages
            
        Returns:
            (is_valid, error_message)
        """
        if not api_key:
            return False, f"{service} key is empty"
        
        if len(api_key) < 10:
            return False, f"{service} key seems too short"
        
        # Check for placeholder values
        placeholders = ['your_key_here', 'api_key', 'token', 'xxxx', 'example']
        if api_key.lower() in placeholders:
            return False, f"{service} key appears to be a placeholder"
        
        return True, None
    
    @staticmethod
    def validate_model_id(model_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate LLM model ID format.
        
        Args:
            model_id: Model ID to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not model_id:
            return False, "Model ID is empty"
        
        # Check reasonable format (should contain some identifier)
        if len(model_id) < 3:
            return False, "Model ID seems too short"
        
        return True, None
    
    @staticmethod
    def validate_temperature(temperature: float) -> Tuple[bool, Optional[str]]:
        """
        Validate temperature parameter.
        
        Args:
            temperature: Temperature value
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(temperature, (int, float)):
            return False, "Temperature must be a number"
        
        if temperature < 0.0 or temperature > 2.0:
            return False, f"Temperature {temperature} out of range [0.0, 2.0]"
        
        return True, None


# Global validator instances
input_validator = InputValidator()
config_validator = ConfigValidator()
