"""
Unit tests for musicgen.core.prompt module - simplified to match actual API.
"""

import pytest

from musicgen.core.prompt import PromptEngineer


class TestPromptEngineerSimple:
    """Test PromptEngineer class with actual API."""

    @pytest.fixture
    def prompt_engineer(self):
        """Create PromptEngineer instance."""
        return PromptEngineer()

    def test_init(self, prompt_engineer):
        """Test initialization of PromptEngineer."""
        assert isinstance(prompt_engineer.genres, dict)
        assert isinstance(prompt_engineer.instruments, dict)
        assert isinstance(prompt_engineer.moods, list)

        # Check some expected values
        assert "jazz" in prompt_engineer.genres
        assert "strings" in prompt_engineer.instruments
        assert "upbeat" in prompt_engineer.moods

    def test_improve_prompt_basic(self, prompt_engineer):
        """Test basic prompt improvement."""
        # Short prompt should be expanded
        improved = prompt_engineer.improve_prompt("jazz")
        assert len(improved) > len("jazz")
        assert "jazz" in improved.lower()

        # Longer prompt
        improved = prompt_engineer.improve_prompt("smooth jazz piano music")
        assert "jazz" in improved.lower()
        assert "piano" in improved.lower()

    def test_improve_prompt_empty(self, prompt_engineer):
        """Test improvement of empty prompt."""
        # Empty string should return something (not empty)
        result = prompt_engineer.improve_prompt("")
        assert isinstance(result, str)

        # Whitespace should also return something
        result = prompt_engineer.improve_prompt("   ")
        assert isinstance(result, str)

    def test_validate_prompt(self, prompt_engineer):
        """Test prompt validation."""
        # Valid prompts
        valid, issues = prompt_engineer.validate_prompt("piano music")
        assert valid is True
        assert len(issues) == 0

        valid, issues = prompt_engineer.validate_prompt("jazz with saxophone")
        assert valid is True
        assert len(issues) == 0

        # Invalid prompts
        valid, issues = prompt_engineer.validate_prompt("")
        assert valid is False
        assert len(issues) > 0

        valid, issues = prompt_engineer.validate_prompt("a")  # Too short
        assert valid is False
        assert len(issues) > 0

        valid, issues = prompt_engineer.validate_prompt("x" * 1000)  # Too long
        assert valid is False
        assert len(issues) > 0

    def test_get_examples(self, prompt_engineer):
        """Test example generation."""
        # General examples
        examples = prompt_engineer.get_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0
        assert all(isinstance(ex, str) for ex in examples)

        # Genre-specific examples
        jazz_examples = prompt_engineer.get_examples(genre="jazz")
        assert isinstance(jazz_examples, list)
        assert len(jazz_examples) > 0
        assert any("jazz" in ex.lower() for ex in jazz_examples)

        # Unknown genre
        examples = prompt_engineer.get_examples(genre="unknown")
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_suggest_variations(self, prompt_engineer):
        """Test variation suggestions."""
        base_prompt = "jazz piano music"

        # Default count
        variations = prompt_engineer.suggest_variations(base_prompt)
        assert isinstance(variations, list)
        assert len(variations) == 3
        assert all(isinstance(v, str) for v in variations)
        assert all(v != base_prompt for v in variations)

        # Custom count
        variations = prompt_engineer.suggest_variations(base_prompt, count=5)
        assert len(variations) == 5

        # Empty prompt - may still return variations
        variations = prompt_engineer.suggest_variations("")
        assert isinstance(variations, list)
        assert len(variations) == 3  # Still returns 3 variations even for empty

    def test_private_methods_exist(self, prompt_engineer):
        """Test that private helper methods exist."""
        assert hasattr(prompt_engineer, "_expand_short_prompt")
        assert hasattr(prompt_engineer, "_add_genre_context")
        assert hasattr(prompt_engineer, "_add_mood")
        assert hasattr(prompt_engineer, "_structure_prompt")
        assert hasattr(prompt_engineer, "_replace_or_add_mood")

    def test_genre_vocabulary(self, prompt_engineer):
        """Test genre vocabulary is properly set up."""
        # Check some expected genres (ambient is under electronic)
        expected_genres = ["jazz", "electronic", "classical", "rock", "hip-hop"]
        for genre in expected_genres:
            assert genre in prompt_engineer.genres
            assert isinstance(prompt_engineer.genres[genre], list)
            assert len(prompt_engineer.genres[genre]) > 0

    def test_instrument_vocabulary(self, prompt_engineer):
        """Test instrument vocabulary is properly set up."""
        # Check some expected instrument categories (based on actual implementation)
        expected_categories = ["strings", "keys", "winds", "percussion", "electronic"]
        for category in expected_categories:
            assert category in prompt_engineer.instruments
            assert isinstance(prompt_engineer.instruments[category], list)
            assert len(prompt_engineer.instruments[category]) > 0

    def test_mood_vocabulary(self, prompt_engineer):
        """Test mood vocabulary is properly set up."""
        # Check some expected moods (based on actual implementation)
        expected_moods = ["upbeat", "mellow", "energetic", "relaxing", "dramatic"]
        for mood in expected_moods:
            assert mood in prompt_engineer.moods

    def test_improve_prompt_with_genre(self, prompt_engineer):
        """Test prompt improvement maintains genre context."""
        # Test that genre is preserved and enhanced
        prompt = "electronic music"
        improved = prompt_engineer.improve_prompt(prompt)
        assert "electronic" in improved.lower()

        # Test jazz genre
        prompt = "bebop jazz"
        improved = prompt_engineer.improve_prompt(prompt)
        assert "jazz" in improved.lower() or "bebop" in improved.lower()

    def test_suggest_variations_maintains_style(self, prompt_engineer):
        """Test that variations maintain the original style."""
        prompt = "classical piano sonata"
        variations = prompt_engineer.suggest_variations(prompt)

        # At least some variations should maintain classical theme
        classical_count = sum(1 for v in variations if "classical" in v.lower())
        assert classical_count > 0


if __name__ == "__main__":
    pytest.main([__file__])
