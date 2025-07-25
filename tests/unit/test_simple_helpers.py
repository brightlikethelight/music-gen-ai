"""
Simple tests for utility functions to establish baseline test coverage.
These tests are designed to be simple, self-contained, and not require
any complex dependencies or mocking.
"""

import pytest
from musicgen.utils.helpers import format_time, hash_text, validate_prompt_length


class TestSimpleHelpers:
    """Test simple helper functions that don't require external dependencies."""
    
    def test_format_time(self):
        """Test time formatting function."""
        # Test various time durations
        assert format_time(0) == "0.0s"
        assert format_time(30) == "30.0s"
        assert format_time(60) == "1m 0s"
        assert format_time(90) == "1m 30s"
        assert format_time(3661) == "1h 1m"
        assert format_time(7200) == "2h 0m"
        
        # Test edge cases
        assert format_time(59.9) == "59.9s"  # Under 60 seconds
        assert format_time(0.5) == "0.5s"  # Fractional seconds
    
    def test_hash_text(self):
        """Test text hashing function."""
        # Test basic hashing
        hash1 = hash_text("test prompt")
        assert isinstance(hash1, str)
        assert len(hash1) == 8  # Should return 8-character hash
        
        # Test consistency
        hash2 = hash_text("test prompt")
        assert hash1 == hash2  # Same input should give same hash
        
        # Test different inputs give different hashes
        hash3 = hash_text("different prompt")
        assert hash1 != hash3
        
        # Test edge cases
        assert len(hash_text("")) == 8  # Empty string
        assert len(hash_text("a" * 1000)) == 8  # Long string
        assert len(hash_text("🎵🎶")) == 8  # Unicode
    
    def test_validate_prompt_length(self):
        """Test prompt validation function."""
        # Test normal cases
        assert validate_prompt_length("Short prompt") == "Short prompt"
        assert validate_prompt_length("") == ""
        
        # Test trimming with ellipsis
        long_prompt = "a" * 1000
        validated = validate_prompt_length(long_prompt, max_length=100)
        # Function adds "..." when truncating
        assert validated.endswith("...")
        assert len(validated) <= 103  # max_length + length of "..."
        
        # Test with custom max length
        validated = validate_prompt_length("Test prompt that is longer", max_length=4)
        assert validated == "Test..."
        
        # Test whitespace handling
        assert validate_prompt_length("  Test  ") == "Test"
        assert validate_prompt_length("\n\tTest\n\t") == "Test"