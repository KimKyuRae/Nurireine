"""
Unit tests for validation module.
"""

import pytest
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import only the validation module directly to avoid heavy dependencies
from Nurireine.validation import InputValidator, ConfigValidator


class TestInputValidator:
    """Test InputValidator class."""
    
    def test_validate_message_content_valid(self, sample_message_content):
        """Test validation of valid message content."""
        is_valid, error = InputValidator.validate_message_content(sample_message_content)
        assert is_valid
        assert error is None
    
    def test_validate_message_content_empty(self):
        """Test validation of empty content."""
        is_valid, error = InputValidator.validate_message_content("")
        assert not is_valid
        assert "Empty" in error
    
    def test_validate_message_content_too_long(self):
        """Test validation of overly long content."""
        long_content = "x" * 3000
        is_valid, error = InputValidator.validate_message_content(long_content)
        assert not is_valid
        assert "too long" in error
    
    def test_validate_message_content_with_null_bytes(self):
        """Test validation of content with null bytes."""
        content = "Hello\x00World"
        is_valid, error = InputValidator.validate_message_content(content)
        assert not is_valid
        assert "null bytes" in error
    
    def test_sanitize_for_embedding(self):
        """Test text sanitization for embeddings."""
        text = "Hello\x00World\x01Test  \n\n  Extra   Spaces"
        sanitized = InputValidator.sanitize_for_embedding(text)
        
        # Check control characters removed
        assert '\x00' not in sanitized
        assert '\x01' not in sanitized
        
        # Check whitespace normalized
        assert "  " not in sanitized
        assert "Hello" in sanitized
        assert "World" in sanitized
    
    def test_validate_user_id_valid(self, sample_user_id):
        """Test validation of valid user ID."""
        is_valid, error = InputValidator.validate_user_id(sample_user_id)
        assert is_valid
        assert error is None
    
    def test_validate_context_dict_valid(self, sample_context):
        """Test validation of valid context dictionary."""
        is_valid, error = InputValidator.validate_context_dict(sample_context)
        assert is_valid
        assert error is None
    
    def test_validate_context_dict_missing_keys(self, invalid_context):
        """Test validation of context with missing keys."""
        is_valid, error = InputValidator.validate_context_dict(invalid_context)
        assert not is_valid
        assert "Missing required key" in error


class TestConfigValidator:
    """Test ConfigValidator class."""
    
    def test_validate_api_key_valid(self):
        """Test validation of valid API key."""
        is_valid, error = ConfigValidator.validate_api_key("sk-1234567890abcdef")
        assert is_valid
        assert error is None
    
    def test_validate_api_key_placeholder(self):
        """Test validation rejects placeholder values."""
        is_valid, error = ConfigValidator.validate_api_key("your_key_here")
        assert not is_valid
        assert "placeholder" in error
    
    def test_validate_temperature_valid(self):
        """Test validation of valid temperature."""
        is_valid, error = ConfigValidator.validate_temperature(0.7)
        assert is_valid
        assert error is None
    
    def test_validate_temperature_out_of_range_high(self):
        """Test validation rejects temperature above range."""
        is_valid, error = ConfigValidator.validate_temperature(2.1)
        assert not is_valid
        assert "out of range" in error
