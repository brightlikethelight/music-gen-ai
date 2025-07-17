# MusicGen AI Configuration Quick Reference

## 🚀 Quick Start Commands

```bash
# Load and validate production config
python scripts/config_cli.py validate --environment production

# Generate configuration schema
python scripts/config_cli.py schema --output schema.json

# Create example config
python scripts/config_cli.py example --environment staging

# Health check
python scripts/config_cli.py check --environment production

# List all configurations
python scripts/config_cli.py list --detailed
```

## 📋 Configuration Loading

```python
from music_gen.core.config_manager import load_config

# Basic usage
config = load_config(environment="production")

# With overrides
config = load_config(
    environment="production",
    overrides=["api.workers=8", "model.device=cuda"]
)

# Access values
print(f"Port: {config.api.port}")
print(f"Model: {config.model.default_model}")
```

## 🌍 Environment Variables

### Required for Production
```bash
JWT_SECRET="32-character-minimum-secret-key"
WANDB_API_KEY="your-wandb-api-key"
POSTGRES_HOST="postgres.example.com"
POSTGRES_PASSWORD="secure-password"
REDIS_HOST="redis.example.com"
REDIS_PASSWORD="redis-password"
```

### Optional with Defaults
```bash
REDIS_PORT="6379"
POSTGRES_PORT="5432"
POSTGRES_DB="musicgen_prod"
MODEL_CACHE_DIR="./cache/models"
AUDIO_OUTPUT_DIR="./outputs/audio"
```

## 📁 File Structure

```
configs/
├── app/
│   ├── base.yaml           # Foundation
│   ├── development.yaml    # Dev overrides
│   ├── staging.yaml        # Staging config  
│   └── production.yaml     # Production config
├── model/
│   ├── musicgen_small.yaml
│   ├── musicgen_medium.yaml
│   └── musicgen_large.yaml
└── infrastructure/
    ├── local.yaml
    ├── docker.yaml
    └── kubernetes.yaml
```

## ⚙️ Key Configuration Sections

| Section | Purpose | Key Settings |
|---------|---------|--------------|
| `app` | Basic app info | `environment`, `debug`, `version` |
| `api` | HTTP server | `port`, `workers`, `cors` |
| `auth` | Security | `enabled`, `jwt_secret`, `api_key_required` |
| `model` | AI models | `default_model`, `cache_size`, `device` |
| `generation` | Music generation | `max_duration`, `temperature`, `quality_mode` |
| `audio` | Audio processing | `sample_rate`, `format`, `output_dir` |
| `resources` | System resources | `max_memory_usage_percent`, `gpu_memory_fraction` |
| `features` | Feature flags | `streaming_generation`, `batch_processing` |
| `monitoring` | Observability | `metrics`, `health_check`, `logging` |

## 🔒 Production Requirements

| Setting | Development | Production | Validation |
|---------|-------------|------------|------------|
| `app.debug` | `true` | `false` | Must be false |
| `auth.enabled` | `false` | `true` | Must be true |
| `auth.jwt_secret` | Any | 32+ chars | Length check |
| `auth.jwt_algorithm` | `HS256` | `RS256` | More secure |
| `api.workers` | `1` | `4+` | Performance |
| `limits` | Optional | Required | Resource limits |

## 🎛️ Common Overrides

```bash
# Performance tuning
--override api.workers=8
--override model.cache_size=10
--override generation.max_batch_size=8

# Hardware specific
--override model.device=cuda
--override resources.gpu_memory_fraction=0.9

# Development shortcuts
--override generation.default_duration=5.0
--override model.compile_model=false
--override auth.enabled=false

# Production security
--override auth.jwt_algorithm=RS256
--override validation.strict_mode=true
--override features.experimental_features=false
```

## 🔍 Validation Rules

### API Configuration
- `port`: 1-65535
- `workers`: 1-32
- `host`: Valid IP or hostname

### Model Configuration  
- `cache_size`: 1-20
- `device`: Must be available (cuda/mps check)
- `download_timeout_seconds`: 30-3600

### Generation Settings
- `default_duration`: 0.1-300.0 seconds
- `max_duration`: > `default_duration`
- `temperature`: 0.1-2.0
- `max_batch_size`: 1-32

### Audio Settings
- `sample_rate`: [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]
- `format`: ["wav", "mp3", "flac", "ogg"]
- `bit_depth`: [8, 16, 24, 32]
- `channels`: [1, 2]

### Resource Limits
- All percentages: 10.0-95.0
- `cleanup_threshold_percent` < `max_memory_usage_percent`
- `gpu_memory_fraction`: 0.1-1.0

## 🚨 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Authentication must be enabled in production` | `auth.enabled=false` in prod | Set `auth.enabled=true` |
| `JWT secret must be at least 32 characters` | Weak JWT secret | Use strong 32+ char secret |
| `CUDA device requested but not available` | `device=cuda` without GPU | Use `device=auto` or `device=cpu` |
| `max_duration must be greater than default_duration` | Invalid duration config | Ensure max > default |
| `Cannot create path` | Invalid/missing directories | Check path permissions |

## 📊 Health Check Items

```bash
python scripts/config_cli.py check --environment production
```

Checks:
- ✅ Configuration loads successfully
- ✅ All paths are accessible
- ✅ Environment consistency (prod requirements)
- ✅ Resource settings are reasonable
- ✅ External service configurations
- ✅ Security settings are appropriate

## 🛠️ Development Workflow

1. **Start with base configuration**
   ```bash
   cp configs/app/development.yaml my_config.yaml
   ```

2. **Make changes and validate**
   ```bash
   python scripts/config_cli.py validate --file my_config.yaml
   ```

3. **Test configuration**
   ```bash
   python scripts/config_cli.py load --file my_config.yaml
   ```

4. **Deploy to staging**
   ```bash
   python scripts/config_cli.py validate --environment staging
   ```

5. **Production deployment**
   ```bash
   python scripts/config_cli.py check --environment production
   ```

## 📖 Full Documentation

- [Complete Configuration Guide](configuration.md)
- [API Reference](../music_gen/core/config_models.py)
- [CLI Tool](../scripts/config_cli.py)
- [Example Configurations](../configs/)