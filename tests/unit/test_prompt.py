"""
Unit tests for musicgen.core.prompt module.
"""

from unittest.mock import MagicMock, patch

import pytest

from musicgen.core.prompt import PromptEngineer


class TestPromptEngineer:
    """Test PromptEngineer class."""

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

    def test_enhance_prompt_basic(self, prompt_engineer):
        """Test basic prompt enhancement."""
        original = "piano music"
        enhanced = prompt_engineer.enhance_prompt(original)

        assert isinstance(enhanced, str)
        assert len(enhanced) >= len(original)
        assert "piano" in enhanced.lower()

    def test_enhance_prompt_with_style(self, prompt_engineer):
        """Test prompt enhancement with style detection."""
        # Test jazz enhancement
        jazz_prompt = "jazz piano"
        enhanced = prompt_engineer.enhance_prompt(jazz_prompt)
        assert "jazz" in enhanced.lower()

        # Test electronic enhancement
        electronic_prompt = "electronic beats"
        enhanced = prompt_engineer.enhance_prompt(electronic_prompt)
        assert "electronic" in enhanced.lower()

    def test_enhance_prompt_empty(self, prompt_engineer):
        """Test enhancement of empty prompt."""
        assert prompt_engineer.enhance_prompt("") == ""
        assert prompt_engineer.enhance_prompt("   ") == ""

    def test_extract_style(self, prompt_engineer):
        """Test style extraction from prompt."""
        # Test genre extraction
        prompt = "smooth jazz with saxophone"
        style = prompt_engineer.extract_style(prompt)
        assert style["genre"] == "jazz"
        assert "saxophone" in style["instruments"]

        # Test mood extraction
        prompt = "upbeat electronic dance music"
        style = prompt_engineer.extract_style(prompt)
        assert style["genre"] == "electronic"
        assert style["mood"] == "upbeat"

    def test_extract_style_no_match(self, prompt_engineer):
        """Test style extraction with no matches."""
        prompt = "random sounds"
        style = prompt_engineer.extract_style(prompt)

        assert style["genre"] is None
        assert style["mood"] is None
        assert len(style["instruments"]) == 0

    def test_extract_style_multiple_genres(self, prompt_engineer):
        """Test style extraction with multiple genre mentions."""
        prompt = "jazz fusion with rock elements"
        style = prompt_engineer.extract_style(prompt)

        # Should pick the first mentioned genre
        assert style["genre"] in ["jazz", "rock"]

    def test_add_technical_details(self, prompt_engineer):
        """Test adding technical details to prompt."""
        base_prompt = "piano music"
        detailed = prompt_engineer.add_technical_details(base_prompt)

        assert isinstance(detailed, str)
        assert len(detailed) > len(base_prompt)

        # Should add some technical terms
        technical_terms = ["quality", "production", "mix", "master", "professional"]
        assert any(term in detailed.lower() for term in technical_terms)

    def test_add_technical_details_with_tempo(self, prompt_engineer):
        """Test adding tempo information."""
        prompt = "fast dance music"
        detailed = prompt_engineer.add_technical_details(prompt, tempo=128)

        assert "128" in detailed or "bpm" in detailed.lower()

    def test_add_technical_details_with_key(self, prompt_engineer):
        """Test adding key information."""
        prompt = "classical piano"
        detailed = prompt_engineer.add_technical_details(prompt, key="C major")

        assert "C major" in detailed or "key" in detailed.lower()

    def test_suggest_instruments(self, prompt_engineer):
        """Test instrument suggestion based on genre."""
        # Jazz suggestions
        jazz_instruments = prompt_engineer.suggest_instruments("jazz")
        assert any(inst in ["saxophone", "piano", "bass", "drums"] for inst in jazz_instruments)

        # Electronic suggestions
        electronic_instruments = prompt_engineer.suggest_instruments("electronic")
        assert any("synth" in inst for inst in electronic_instruments)

    def test_suggest_instruments_unknown_genre(self, prompt_engineer):
        """Test instrument suggestion for unknown genre."""
        instruments = prompt_engineer.suggest_instruments("unknown")
        assert isinstance(instruments, list)
        assert len(instruments) > 0  # Should return some default instruments

    def test_format_duration(self, prompt_engineer):
        """Test duration formatting."""
        # Short duration
        formatted = prompt_engineer.format_duration(15)
        assert "15 second" in formatted or "short" in formatted.lower()

        # Medium duration
        formatted = prompt_engineer.format_duration(60)
        assert "1 minute" in formatted or "60 second" in formatted

        # Long duration
        formatted = prompt_engineer.format_duration(180)
        assert "3 minute" in formatted or "180 second" in formatted

    def test_clean_prompt(self, prompt_engineer):
        """Test prompt cleaning."""
        # Remove extra spaces
        cleaned = prompt_engineer.clean_prompt("  piano   music  ")
        assert cleaned == "piano music"

        # Remove special characters
        cleaned = prompt_engineer.clean_prompt("piano!!! music???")
        assert "!" not in cleaned
        assert "?" not in cleaned

        # Preserve important punctuation
        cleaned = prompt_engineer.clean_prompt("jazz, blues, and rock")
        assert "," in cleaned

    def test_generate_variations(self, prompt_engineer):
        """Test generating prompt variations."""
        base_prompt = "piano music"
        variations = prompt_engineer.generate_variations(base_prompt, count=3)

        assert len(variations) == 3
        assert all(isinstance(v, str) for v in variations)
        assert all("piano" in v.lower() for v in variations)
        assert len(set(variations)) == 3  # All unique

    def test_generate_variations_with_style(self, prompt_engineer):
        """Test generating styled variations."""
        base_prompt = "jazz piano"
        variations = prompt_engineer.generate_variations(base_prompt, count=5)

        # Should maintain the jazz theme
        assert all(
            "jazz" in v.lower()
            or any(substyle in v.lower() for substyle in prompt_engineer.genres["jazz"])
            for v in variations
        )

    def test_combine_prompts(self, prompt_engineer):
        """Test combining multiple prompts."""
        prompts = ["piano melody", "string accompaniment", "slow tempo"]
        combined = prompt_engineer.combine_prompts(prompts)

        assert all(element in combined.lower() for element in ["piano", "string", "slow"])

    def test_combine_prompts_empty(self, prompt_engineer):
        """Test combining empty prompt list."""
        assert prompt_engineer.combine_prompts([]) == ""
        assert prompt_engineer.combine_prompts(["", "", ""]) == ""

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

    def test_extract_tempo(self, prompt_engineer):
        """Test tempo extraction from prompt."""
        # Explicit BPM
        tempo = prompt_engineer.extract_tempo("120 bpm dance music")
        assert tempo == 120

        tempo = prompt_engineer.extract_tempo("music at 90bpm")
        assert tempo == 90

        # Tempo descriptions
        tempo = prompt_engineer.extract_tempo("slow ballad")
        assert tempo is None or (60 <= tempo <= 80)

        tempo = prompt_engineer.extract_tempo("fast dance")
        assert tempo is None or (120 <= tempo <= 140)

        # No tempo info
        assert prompt_engineer.extract_tempo("piano music") is None

    def test_extract_duration(self, prompt_engineer):
        """Test duration extraction from prompt."""
        # Explicit durations
        duration = prompt_engineer.extract_duration("30 second intro")
        assert duration == 30

        duration = prompt_engineer.extract_duration("2 minute song")
        assert duration == 120

        duration = prompt_engineer.extract_duration("1:30 length")
        assert duration == 90

        # No duration info
        duration = prompt_engineer.extract_duration("piano music")
        assert duration is None

    def test_build_structured_prompt(self, prompt_engineer):
        """Test building structured prompt from components."""
        components = {
            "genre": "jazz",
            "instruments": ["piano", "bass"],
            "mood": "relaxing",
            "tempo": 90,
            "duration": 60,
        }

        structured = prompt_engineer.build_structured_prompt(**components)

        assert isinstance(structured, str)
        assert len(structured) > 0

    def test_optimize_for_model(self, prompt_engineer):
        """Test prompt optimization for specific models."""
        original = "create beautiful piano music"

        # Optimize for different models
        small_optimized = prompt_engineer.optimize_for_model(original, "small")
        large_optimized = prompt_engineer.optimize_for_model(original, "large")

        # Small model should have simpler prompt
        assert len(small_optimized) <= len(large_optimized)

    def test_remove_conflicting_terms(self, prompt_engineer):
        """Test removal of conflicting terms."""
        # Conflicting moods
        prompt = "happy sad music"
        cleaned = prompt_engineer.remove_conflicting_terms(prompt)
        assert not ("happy" in cleaned and "sad" in cleaned)

        # Conflicting tempos
        prompt = "fast slow rhythm"
        cleaned = prompt_engineer.remove_conflicting_terms(prompt)
        assert not ("fast" in cleaned and "slow" in cleaned)

    def test_add_quality_markers(self, prompt_engineer):
        """Test adding quality markers to prompt."""
        original = "piano music"
        with_quality = prompt_engineer.add_quality_markers(original)

        quality_terms = ["high quality", "professional", "studio", "mastered", "pristine"]
        assert any(term in with_quality.lower() for term in quality_terms)

    def test_localize_prompt(self, prompt_engineer):
        """Test prompt localization for different music styles."""
        # Western style
        western = prompt_engineer.localize_prompt("folk music", region="western")
        assert any(term in western.lower() for term in ["acoustic", "guitar", "banjo"])

        # Eastern style
        eastern = prompt_engineer.localize_prompt("traditional music", region="eastern")
        assert any(term in eastern.lower() for term in ["asian", "oriental", "traditional"])

    def test_chain_enhancements(self, prompt_engineer):
        """Test chaining multiple enhancement methods."""
        original = "music"

        # Chain multiple enhancements
        enhanced = original
        enhanced = prompt_engineer.add_quality_markers(enhanced)
        enhanced = prompt_engineer.add_technical_details(enhanced)
        enhanced = prompt_engineer.clean_prompt(enhanced)

        assert len(enhanced) > len(original)
        assert enhanced != original

    @patch("random.choice")
    def test_randomization(self, mock_choice, prompt_engineer):
        """Test randomization in prompt generation."""
        mock_choice.side_effect = lambda x: x[0]  # Always pick first item

        prompt = "music"
        variations = [prompt_engineer.enhance_prompt(prompt) for _ in range(3)]

        # With mocked random, should get consistent results
        assert variations[0] == variations[1] == variations[2]
