# Testing Guide for MusicGen

This document describes the testing strategy, organization, and best practices for the MusicGen project.

## Test Categories

### Unit Tests (`tests/unit/`)
Fast, isolated tests that verify individual components work correctly in isolation.
- **Speed**: < 1 second per test
- **Dependencies**: Mocked
- **Purpose**: Verify function/class logic
- **Marker**: `@pytest.mark.unit`

### Integration Tests (`tests/integration/`)
Tests that verify multiple components work together correctly.
- **Speed**: 1-30 seconds per test
- **Dependencies**: May use real components (not ML models)
- **Purpose**: Verify API workflows, authentication flows
- **Marker**: `@pytest.mark.integration`

### End-to-End Tests (`tests/e2e/`)
Full system tests that simulate real user scenarios.
- **Speed**: May take > 30 seconds
- **Dependencies**: Full system (except ML models in CI)
- **Purpose**: Verify complete user journeys
- **Marker**: `@pytest.mark.slow`

## Running Tests

### All Tests
```bash
pytest
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# By marker
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

### With Coverage
```bash
# Full coverage report
pytest --cov=src/musicgen --cov-report=html

# Quick coverage summary
pytest --cov=src/musicgen --cov-report=term-missing:skip-covered
```

### Skip Model Downloads
For faster CI runs, skip ML model downloads:
```bash
MUSICGEN_SKIP_MODEL_DOWNLOAD=1 pytest
```

## Test Configuration

### pytest.ini
Main test configuration including:
- Coverage thresholds (currently 50%)
- Test timeout (30 seconds)
- Custom markers

### conftest.py
Shared fixtures including:
- `client`: FastAPI TestClient
- `mock_musicgen`: Mocked MusicGenerator
- Environment variable setup

## Writing New Tests

### Test Naming
Use descriptive names that explain what's being tested:
```python
# Good
def test_should_hash_password_with_bcrypt():
def test_login_fails_with_invalid_credentials():

# Bad
def test_password():
def test_login():
```

### Test Structure (Arrange-Act-Assert)
```python
def test_user_registration():
    # Arrange
    user_data = {"username": "testuser", "email": "test@example.com", "password": "secret123"}

    # Act
    response = client.post("/auth/register", json=user_data)

    # Assert
    assert response.status_code == 200
    assert "access_token" in response.json()
```

### Using Fixtures
```python
@pytest.fixture
def authenticated_client(client):
    """Client with valid authentication token."""
    # Register and login
    response = client.post("/auth/register", json={...})
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client

def test_protected_endpoint(authenticated_client):
    response = authenticated_client.get("/auth/me")
    assert response.status_code == 200
```

### Mocking Guidelines
1. **Mock external dependencies** (ML models, external APIs, file system)
2. **Don't mock the code under test**
3. **Use fixtures** from conftest.py when available
4. **Be explicit** about what you're mocking

```python
# Good - mocking external dependency
@patch("musicgen.core.generator.MusicgenForConditionalGeneration")
def test_generator_loads_model(mock_model):
    generator = MusicGenerator()
    mock_model.from_pretrained.assert_called_once()

# Bad - mocking the function you're testing
@patch("musicgen.core.prompt.improve_prompt")  # Don't do this
def test_prompt_improvement(mock_improve):
    mock_improve.return_value = "improved"
    # This tests nothing
```

## Coverage Requirements

- **Minimum**: 50% line coverage (enforced in CI)
- **Target for patches**: 80% coverage on new code
- **Focus on**: Meaningful tests, not coverage metrics

### Coverage Exclusions
The following are excluded from coverage:
- Type checking blocks (`if TYPE_CHECKING:`)
- Abstract methods
- Import error handlers

## Test Markers

Available markers (defined in pytest.ini):
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests taking > 30 seconds
- `@pytest.mark.gpu` - Tests requiring GPU hardware
- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.auth` - Authentication tests
- `@pytest.mark.cli` - CLI tests

## Troubleshooting

### Tests Hang on Model Download
Set environment variable:
```bash
export MUSICGEN_SKIP_MODEL_DOWNLOAD=1
export TRANSFORMERS_OFFLINE=1
```

### Import Errors
Ensure you've installed dev dependencies:
```bash
pip install -e ".[dev]"
```

### Rate Limiting in Tests
Tests may fail due to rate limiting. The test fixtures should reset rate limiter state, but if issues persist:
```python
from musicgen.api.rest.middleware.rate_limiting import RateLimiter
RateLimiter._instance = None  # Reset singleton
```

### Coverage Not Meeting Threshold
If coverage drops below 50%:
1. Don't add coverage-chasing tests
2. Add meaningful tests for untested functionality
3. Consider if the threshold should be adjusted

## CI/CD Integration

Tests run automatically on:
- Push to `main` or `develop`
- Pull requests to `main`

CI runs:
1. Code quality checks (black, isort, flake8, mypy)
2. Unit tests with coverage
3. Integration tests
4. Security scans (bandit, safety)

See `.github/workflows/ci.yml` for full configuration.
