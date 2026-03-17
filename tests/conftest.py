"""
PyTest configuration and shared fixtures.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Lazy import torch only when needed
torch = None


def _ensure_torch():
    """Ensure torch is imported when needed."""
    global torch
    if torch is None:
        import torch as torch_lib

        torch = torch_lib
    return torch


# Set test environment variables globally before any imports.
# These MUST be set before module-level imports in test files trigger
# side effects (e.g., _ensure_imports() in generator.py, get_jwt_secret()).
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"
os.environ["MUSICGEN_SKIP_REDIS"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-minimum-32-chars"
os.environ["TOKEN_BLACKLIST_FAIL_POLICY"] = "open"


@pytest.fixture(scope="session")
def device():
    """Get test device (CPU for CI/CD compatibility)."""
    _ensure_torch()
    return torch.device("cpu")


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    _ensure_torch()
    sample_rate = 24000
    duration = 1.0
    samples = int(duration * sample_rate)

    # Generate sine wave with some harmonics
    t = torch.linspace(0, duration, samples)
    freq = 440.0  # A4
    audio = (
        torch.sin(2 * np.pi * freq * t) * 0.5
        + torch.sin(2 * np.pi * freq * 2 * t) * 0.3
        + torch.sin(2 * np.pi * freq * 3 * t) * 0.2
    )

    # Add to mono channel
    audio = audio.unsqueeze(0)

    return audio, sample_rate


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    _ensure_torch()
    batch_size = 2
    seq_len = 100
    vocab_size = 256

    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "texts": ["Happy jazz music", "Calm ambient sounds"],
        "genre_ids": torch.randint(0, 10, (batch_size,)),
        "mood_ids": torch.randint(0, 5, (batch_size,)),
        "tempo": torch.tensor([120.0, 90.0]),
    }

    return batch


@pytest.fixture
def sample_texts():
    """Sample text prompts for testing."""
    return [
        "Happy jazz music with piano",
        "Calm ambient music with nature sounds",
        "Energetic electronic dance music",
        "Melancholic classical piano piece",
        "Upbeat rock song with guitar solo",
    ]


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests that test individual components")
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interactions"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests that test complete workflows")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line("markers", "gpu: Tests that require GPU")
    config.addinivalue_line("markers", "model: Tests that require model weights")


# Skip slow tests by default
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("--runslow"):
        # Run slow tests if explicitly requested
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--rungpu", action="store_true", default=False, help="run GPU tests")


@pytest.fixture
def auth_headers():
    """Create authentication headers for tests that require auth."""
    try:
        from musicgen.api.middleware.auth import UserRole, get_auth_middleware

        auth = get_auth_middleware()
        token = auth.create_access_token(
            user_id="test_user_123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER.value],
        )
        return {"Authorization": f"Bearer {token}"}
    except Exception:
        pytest.skip("Auth middleware not available")


@pytest.fixture
def mock_ml_deps():
    """
    Opt-in fixture providing mock ML objects for tests that need them.
    Not autouse — tests must request this fixture explicitly.
    """
    _ensure_torch()

    # Create mock processor and model instances
    processor_instance = MagicMock()
    processor_instance.return_value = {
        "input_ids": torch.zeros((1, 10)),
        "attention_mask": torch.ones((1, 10)),
    }

    model_instance = MagicMock()
    model_instance.to = MagicMock(return_value=model_instance)
    model_instance.config.audio_encoder.sampling_rate = 32000
    model_instance.generate = MagicMock(return_value=torch.randn(1, 1, 32000))

    # Create mock classes
    processor_class = MagicMock()
    processor_class.from_pretrained = MagicMock(return_value=processor_instance)

    model_class = MagicMock()
    model_class.from_pretrained = MagicMock(return_value=model_instance)

    return {
        "processor_class": processor_class,
        "model_class": model_class,
        "processor_instance": processor_instance,
        "model_instance": model_instance,
    }


@pytest.fixture
def mock_musicgen():
    """
    Fixture to mock MusicGenerator for individual tests.
    Use this instead of global mocking to avoid conflicts.
    """
    with (
        patch("transformers.AutoProcessor") as mock_processor_class,
        patch("transformers.MusicgenForConditionalGeneration") as mock_model_class,
    ):

        # Mock processor
        mock_processor = MagicMock()
        _ensure_torch()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_processor_class.from_pretrained.return_value = mock_processor

        # Mock model
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_config.audio_encoder.sampling_rate = 32000
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model
        _ensure_torch()
        mock_model.generate.return_value = torch.randn(1, 1, 32000)
        mock_model_class.from_pretrained.return_value = mock_model

        yield {
            "processor_class": mock_processor_class,
            "model_class": mock_model_class,
            "processor": mock_processor,
            "model": mock_model,
        }


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    # Only set torch seed if torch is already imported
    if torch is not None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    np.random.seed(42)
