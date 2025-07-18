"""
Prompt engineering for better music generation.
Simple, practical, no over-engineering.
"""

import random
from typing import List, Tuple, Dict, Optional


class PromptEngineer:
    """
    Improve prompts for better music generation.
    Based on what actually works with MusicGen.
    """

    def __init__(self):
        # Genre knowledge
        self.genres = {
            "jazz": ["smooth", "bebop", "cool", "swing", "fusion"],
            "electronic": ["ambient", "techno", "house", "dubstep", "synthwave"],
            "classical": ["baroque", "romantic", "modern", "orchestral", "chamber"],
            "rock": ["indie", "alternative", "progressive", "psychedelic", "garage"],
            "hip-hop": ["boom bap", "trap", "lo-fi", "old school", "jazz hop"],
            "folk": ["acoustic", "indie folk", "traditional", "contemporary"],
            "world": ["african", "latin", "asian", "middle eastern", "celtic"],
        }

        # Instrument categories
        self.instruments = {
            "strings": ["guitar", "violin", "cello", "bass", "harp", "ukulele"],
            "keys": ["piano", "synthesizer", "organ", "electric piano", "harpsichord"],
            "winds": ["saxophone", "flute", "clarinet", "trumpet", "oboe"],
            "percussion": ["drums", "congas", "tabla", "timpani", "xylophone"],
            "electronic": ["synth bass", "pad", "lead synth", "arpeggiator"],
        }

        # Mood/style descriptors
        self.moods = [
            "upbeat",
            "mellow",
            "energetic",
            "relaxing",
            "dramatic",
            "peaceful",
            "intense",
            "dreamy",
            "groovy",
            "atmospheric",
        ]

        # Tempo descriptors
        self.tempos = [
            "slow",
            "moderate",
            "fast",
            "very fast",
            "andante",
            "allegro",
            "adagio",
            "presto",
        ]

    def improve_prompt(self, prompt: str) -> str:
        """
        Improve a prompt for better generation.

        Args:
            prompt: Original prompt

        Returns:
            Improved prompt
        """
        prompt = prompt.strip().lower()

        # Check if prompt is too short
        if len(prompt.split()) < 3:
            prompt = self._expand_short_prompt(prompt)

        # Add genre if missing
        if not any(genre in prompt for genre in self.genres):
            prompt = self._add_genre_context(prompt)

        # Add mood if missing
        if not any(mood in prompt for mood in self.moods):
            prompt = self._add_mood(prompt)

        # Ensure good structure
        prompt = self._structure_prompt(prompt)

        return prompt

    def _expand_short_prompt(self, prompt: str) -> str:
        """Expand short prompts with context."""
        # Common short prompts and their expansions
        expansions = {
            "jazz": "smooth jazz with piano and saxophone",
            "rock": "energetic rock with electric guitar",
            "classical": "classical orchestral piece",
            "electronic": "ambient electronic soundscape",
            "piano": "peaceful piano melody",
            "guitar": "acoustic guitar fingerstyle",
            "drums": "rhythmic drum pattern",
        }

        for key, expansion in expansions.items():
            if key in prompt:
                return expansion

        # Generic expansion
        return f"instrumental {prompt} music"

    def _add_genre_context(self, prompt: str) -> str:
        """Add genre context if missing."""
        # Try to detect implicit genre
        for instrument_type, instruments in self.instruments.items():
            for instrument in instruments:
                if instrument in prompt:
                    # Map instruments to likely genres
                    if instrument in ["piano", "violin", "cello"]:
                        return f"classical {prompt}"
                    elif instrument in ["synthesizer", "synth"]:
                        return f"electronic {prompt}"
                    elif instrument in ["guitar", "bass", "drums"]:
                        return f"rock {prompt}"
                    elif instrument in ["saxophone", "trumpet"]:
                        return f"jazz {prompt}"

        # Default to generic
        return f"instrumental {prompt}"

    def _add_mood(self, prompt: str) -> str:
        """Add mood descriptor if missing."""
        # Check current mood context
        has_mood = any(mood in prompt for mood in self.moods)

        if not has_mood:
            # Pick appropriate mood based on genre
            if "jazz" in prompt or "classical" in prompt:
                mood = random.choice(["smooth", "peaceful", "mellow"])
            elif "rock" in prompt or "electronic" in prompt:
                mood = random.choice(["energetic", "upbeat", "intense"])
            else:
                mood = random.choice(["relaxing", "atmospheric", "dreamy"])

            return f"{mood} {prompt}"

        return prompt

    def _structure_prompt(self, prompt: str) -> str:
        """Ensure prompt has good structure."""
        # Clean up multiple spaces
        prompt = " ".join(prompt.split())

        # Capitalize appropriately
        words = prompt.split()

        # Don't capitalize articles, prepositions
        dont_capitalize = {"a", "an", "the", "with", "in", "on", "for", "and", "or"}

        structured = []
        for i, word in enumerate(words):
            if i == 0 or word not in dont_capitalize:
                structured.append(word.capitalize())
            else:
                structured.append(word)

        return " ".join(structured)

    def validate_prompt(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Validate a prompt and return issues.

        Args:
            prompt: Prompt to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not prompt or not prompt.strip():
            issues.append("Prompt is empty")
            return False, issues

        # Length checks
        word_count = len(prompt.split())
        if word_count < 2:
            issues.append("Prompt too short (add more description)")
        elif word_count > 20:
            issues.append("Prompt too long (may confuse the model)")

        # Check for vocals (MusicGen limitation)
        vocal_words = ["vocal", "voice", "singing", "lyrics", "singer", "rap"]
        if any(word in prompt.lower() for word in vocal_words):
            issues.append("MusicGen doesn't support vocals (instrumental only)")

        # Check for non-music content
        non_music = ["speech", "talking", "podcast", "audiobook"]
        if any(word in prompt.lower() for word in non_music):
            issues.append("Prompt should describe music, not speech")

        return len(issues) == 0, issues

    def get_examples(self, genre: Optional[str] = None) -> List[str]:
        """Get example prompts for inspiration."""
        examples = {
            "jazz": [
                "Smooth jazz piano with soft drums and upright bass",
                "Bebop saxophone solo with walking bass line",
                "Cool jazz quartet playing a mellow evening tune",
            ],
            "electronic": [
                "Ambient electronic soundscape with ethereal pads",
                "Upbeat synthwave with retro 80s vibes",
                "Deep house groove with pulsing bass",
            ],
            "classical": [
                "Peaceful classical piano sonata in major key",
                "Dramatic orchestral piece with string section",
                "Baroque harpsichord fugue",
            ],
            "rock": [
                "Energetic indie rock with jangly guitars",
                "Progressive rock instrumental with complex rhythms",
                "Psychedelic rock jam with wah-wah guitar",
            ],
        }

        if genre and genre in examples:
            return examples[genre]

        # Return mix of all
        all_examples = []
        for genre_examples in examples.values():
            all_examples.extend(genre_examples)

        return random.sample(all_examples, min(5, len(all_examples)))

    def suggest_variations(self, prompt: str, count: int = 3) -> List[str]:
        """
        Suggest variations of a prompt.

        Args:
            prompt: Original prompt
            count: Number of variations

        Returns:
            List of prompt variations
        """
        variations = []

        # Variation 1: Change mood
        moods = [m for m in self.moods if m not in prompt.lower()]
        if moods:
            new_mood = random.choice(moods)
            var1 = self._replace_or_add_mood(prompt, new_mood)
            variations.append(var1)

        # Variation 2: Add/change tempo
        tempos = [t for t in self.tempos if t not in prompt.lower()]
        if tempos and len(variations) < count:
            new_tempo = random.choice(tempos)
            var2 = f"{new_tempo} {prompt}"
            variations.append(var2)

        # Variation 3: Add instrument detail
        if len(variations) < count:
            for instrument_type, instruments in self.instruments.items():
                for inst in instruments:
                    if inst in prompt.lower():
                        # Add a complementary instrument
                        other_insts = [i for i in instruments if i != inst]
                        if other_insts:
                            new_inst = random.choice(other_insts)
                            var3 = f"{prompt} with {new_inst}"
                            variations.append(var3)
                            break
                if len(variations) >= count:
                    break

        # Ensure we have enough variations
        while len(variations) < count:
            # Generic variation
            prefix = random.choice(["experimental", "modern", "traditional", "fusion"])
            variations.append(f"{prefix} {prompt}")

        return variations[:count]

    def _replace_or_add_mood(self, prompt: str, new_mood: str) -> str:
        """Replace existing mood or add new one."""
        # Check if prompt has a mood
        prompt_words = prompt.lower().split()

        for mood in self.moods:
            if mood in prompt_words:
                # Replace existing mood
                return prompt.replace(mood, new_mood)

        # Add new mood at beginning
        return f"{new_mood} {prompt}"
