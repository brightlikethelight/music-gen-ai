"""Tests for the PromptEngineer module."""

import pytest

from musicgen.core.prompt import PromptEngineer


class TestPromptEngineer:
    """Test PromptEngineer functionality."""

    @pytest.fixture
    def engineer(self):
        """Create a PromptEngineer instance."""
        return PromptEngineer()

    def test_init(self, engineer):
        """Test PromptEngineer initialization."""
        assert engineer is not None
        assert engineer.genres is not None
        assert engineer.instruments is not None
        assert engineer.moods is not None
        assert engineer.tempos is not None

    def test_validate_prompt_valid(self, engineer):
        """Test valid prompt validation."""
        valid_prompts = [
            "Happy jazz music",
            "Calm ambient sounds with nature",
            "Energetic rock guitar solo",
            "Classical piano piece in minor key",
            "Electronic dance music with bass",
        ]

        for prompt in valid_prompts:
            is_valid, issues = engineer.validate_prompt(prompt)
            assert is_valid is True
            assert len(issues) == 0

    def test_validate_prompt_empty(self, engineer):
        """Test empty prompt validation."""
        is_valid, issues = engineer.validate_prompt("")
        assert is_valid is False
        assert "empty" in issues[0].lower()

        is_valid, issues = engineer.validate_prompt("   ")
        assert is_valid is False
        assert "empty" in issues[0].lower()

    def test_validate_prompt_too_short(self, engineer):
        """Test short prompt validation."""
        is_valid, issues = engineer.validate_prompt("hi")
        assert is_valid is False
        assert "short" in issues[0].lower() or "brief" in issues[0].lower()

    def test_validate_prompt_too_long(self, engineer):
        """Test long prompt validation."""
        long_prompt = "music " * 100  # 500+ characters
        is_valid, issues = engineer.validate_prompt(long_prompt)
        assert is_valid is False
        assert "long" in issues[0].lower()

    def test_validate_prompt_non_music(self, engineer):
        """Test non-music prompt validation with known non-music keywords."""
        # The validator only checks for explicit non-music keywords
        non_music_prompts = [
            "Generate a speech from the president",
            "Create a podcast episode",
            "Make an audiobook narration",
            "Talking voice over",
        ]

        for prompt in non_music_prompts:
            is_valid, issues = engineer.validate_prompt(prompt)
            # Prompts with non-music keywords should have issues
            assert len(issues) > 0, f"Expected issues for: {prompt}"

    def test_get_examples(self, engineer):
        """Test getting example prompts."""
        # Get general examples
        examples = engineer.get_examples()
        assert len(examples) > 0
        assert all(isinstance(ex, str) for ex in examples)

        # Get genre-specific examples
        jazz_examples = engineer.get_examples(genre="jazz")
        assert len(jazz_examples) > 0
        assert any("jazz" in ex.lower() for ex in jazz_examples)

    def test_suggest_variations(self, engineer):
        """Test prompt variation suggestions."""
        original = "Jazz piano music"
        variations = engineer.suggest_variations(original, count=3)

        assert len(variations) == 3
        assert all(isinstance(v, str) for v in variations)
        # Variations should be different from each other
        assert len(set(variations)) == len(variations)

        # Test with different counts
        variations = engineer.suggest_variations(original, count=5)
        assert len(variations) == 5

    def test_improve_prompt_basic(self, engineer):
        """Test basic prompt improvement."""
        basic_prompts = ["music", "piano", "guitar", "drums", "bass"]

        for prompt in basic_prompts:
            improved = engineer.improve_prompt(prompt)
            # Should be longer and more descriptive
            assert len(improved) > len(prompt), f"Expected improvement for: {prompt}"
            # Should add meaningful content (mood, genre, or instrument context)
            improved_lower = improved.lower()
            has_enhancement = (
                any(mood in improved_lower for mood in engineer.moods)
                or any(genre in improved_lower for genre in engineer.genres)
                or "instrumental" in improved_lower
                or "piano" in improved_lower
                or "guitar" in improved_lower
            )
            assert has_enhancement, f"Expected enhancement in: {improved}"

    def test_improve_prompt_with_genre(self, engineer):
        """Test prompt improvement with genre."""
        prompt = "smooth piano music"
        improved = engineer.improve_prompt(prompt)

        # Should be improved
        assert len(improved) > len(prompt)
        # Should preserve key elements
        assert "piano" in improved.lower()

    def test_private_methods(self, engineer):
        """Test private helper methods."""
        # Test _expand_short_prompt
        expanded = engineer._expand_short_prompt("jazz")
        assert len(expanded) > len("jazz")

        # Test _add_genre_context
        with_genre = engineer._add_genre_context("music with drums")
        assert any(genre in with_genre.lower() for genre in engineer.genres.keys())

        # Test _add_mood
        with_mood = engineer._add_mood("jazz piano")
        assert any(mood in with_mood.lower() for mood in engineer.moods)

        # Test _structure_prompt
        structured = engineer._structure_prompt("random words music jazz piano drums")
        assert len(structured) > 0

    def test_edge_cases(self, engineer):
        """Test edge cases."""
        # Empty string
        is_valid, issues = engineer.validate_prompt("")
        assert is_valid is False
        assert len(issues) > 0

        # Very short prompt
        is_valid, issues = engineer.validate_prompt("a")
        assert is_valid is False

        # Very long prompt
        long_prompt = "music " * 100
        is_valid, issues = engineer.validate_prompt(long_prompt)
        if not is_valid:
            assert len(issues) > 0

        # Prompt with special characters
        prompt = "Jazz music with piano"
        improved = engineer.improve_prompt(prompt)
        assert len(improved) > 0
