# MusicGen AI Configuration System

## Overview

MusicGen AI uses a unified configuration management system built on **Hydra** for YAML configuration files and **Pydantic** for validation and type safety. This system provides:

- ✅ **Environment-specific configurations** (development, staging, production)
- ✅ **Type-safe validation** with comprehensive error reporting
- ✅ **Environment variable substitution** for secrets and deployment flexibility
- ✅ **Schema generation** for documentation and IDE support
- ✅ **CLI tools** for validation and management
- ✅ **Hot-reloading** and dynamic configuration updates

## Quick Start

### Basic Usage

```python
from music_gen.core.config_manager import load_config

# Load development configuration
config = load_config(environment="development")

# Load production configuration with overrides
config = load_config(
    environment="production",
    overrides=["api.workers=8", "model.device=cuda"]
)

# Access configuration values
print(f"API Port: {config.api.port}")
print(f"Model: {config.model.default_model}")
print(f"Environment: {config.app.environment}")
```

### CLI Usage

```bash
# Validate configuration
python scripts/config_cli.py validate --environment production

# Generate schema
python scripts/config_cli.py schema --output config_schema.json

# Check configuration health
python scripts/config_cli.py check --environment staging

# List available configurations
python scripts/config_cli.py list --detailed
```

## Configuration Structure

### Core Configuration Files

```
configs/
├── app/
│   ├── base.yaml           # Foundation configuration
│   ├── development.yaml    # Development overrides
│   ├── staging.yaml        # Staging environment
│   └── production.yaml     # Production environment
├── model/
│   ├── musicgen_small.yaml
│   ├── musicgen_medium.yaml
│   └── musicgen_large.yaml
└── infrastructure/
    ├── local.yaml
    ├── docker.yaml
    └── kubernetes.yaml
```

### Configuration Hierarchy

Configurations use Hydra's composition system:

```yaml
# production.yaml
defaults:
  - base                          # Inherit from base.yaml
  - override /model: musicgen_large  # Use large model
  - override /infrastructure: kubernetes

# Override specific settings
app:
  environment: "production"
  debug: false

auth:
  enabled: true
  jwt_secret: "${JWT_SECRET}"
```

## Configuration Sections

### Application Settings (`app`)

Basic application information and environment settings.

```yaml
app:
  name: "MusicGen AI"
  version: "1.0.0"
  environment: "development"  # development | staging | production
  debug: true
```

**Validation Rules:**
- `name`: Required, non-empty string
- `environment`: Must be one of: development, staging, production
- `debug`: Must be disabled in production

### API Configuration (`api`)

HTTP API server settings including CORS and rate limiting.

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  debug: false
  
  cors:
    enabled: true
    origins:
      - "https://musicgen-ai.com"
      - "https://app.musicgen-ai.com"
    methods: ["GET", "POST", "PUT", "DELETE"]
    credentials: true
  
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_limit: 10
```

**Validation Rules:**
- `port`: 1-65535
- `workers`: 1-32
- `rate_limiting.requests_per_minute`: 1-10000
- `cors.origins`: Valid URL list

### Authentication (`auth`)

Security and authentication configuration.

```yaml
auth:
  enabled: true
  jwt_secret: "${JWT_SECRET}"
  jwt_algorithm: "RS256"        # HS256 | HS384 | HS512 | RS256 | RS384 | RS512
  jwt_expiration_hours: 8
  api_key_required: true
  
  rate_limiting:
    enabled: true
    requests_per_minute: 30
    per_user_limit: 100
```

**Validation Rules:**
- `jwt_secret`: Must be ≥32 characters when auth is enabled
- `jwt_expiration_hours`: 1-8760 (max 1 year)
- Must be enabled in production environment

### Model Configuration (`model`)

AI model settings and caching configuration.

```yaml
model:
  default_model: "facebook/musicgen-large"
  cache_size: 6
  cache_dir: "${MODEL_CACHE_DIR:./cache/models}"
  device: "auto"              # auto | cpu | cuda | mps
  mixed_precision: true
  compile_model: true
  download_timeout_seconds: 180
  preload_models:
    - "facebook/musicgen-medium"
    - "facebook/musicgen-large"
```

**Validation Rules:**
- `cache_size`: 1-20
- `device`: Validates availability (CUDA/MPS)
- `download_timeout_seconds`: 30-3600

### Generation Settings (`generation`)

Music generation parameters and constraints.

```yaml
generation:
  default_duration: 10.0
  max_duration: 60.0
  default_temperature: 0.8
  default_top_k: 250
  default_top_p: 0.0
  max_batch_size: 4
  timeout_seconds: 120
  quality_mode: "high"        # fast | standard | high
  priority_queue_enabled: true
```

**Validation Rules:**
- `default_duration`: 0.1-300.0 seconds
- `max_duration`: Must be > `default_duration`
- `temperature`: 0.1-2.0
- `max_batch_size`: 1-32

### Audio Processing (`audio`)

Audio format and processing settings.

```yaml
audio:
  sample_rate: 24000          # 8000 | 16000 | 22050 | 24000 | 32000 | 44100 | 48000 | 96000
  format: "wav"               # wav | mp3 | flac | ogg
  bit_depth: 16               # 8 | 16 | 24 | 32
  channels: 1                 # 1 | 2
  normalization: "peak"       # peak | rms | lufs | none
  output_dir: "${AUDIO_OUTPUT_DIR:./outputs/audio}"
  temp_dir: "${AUDIO_TEMP_DIR:./tmp/audio}"
  cleanup_temp_files: true
  backup_outputs: true
  backup_retention_days: 7
```

**Validation Rules:**
- `sample_rate`: Must be in supported rates list
- `format`: Must be supported audio format
- `bit_depth`: Must be valid bit depth
- `backup_retention_days`: 1-365

### Resource Management (`resources`)

System resource monitoring and limits.

```yaml
resources:
  max_memory_usage_percent: 75.0
  max_cpu_usage_percent: 80.0
  cleanup_threshold_percent: 70.0
  monitoring_interval_seconds: 60.0
  auto_cleanup_enabled: true
  gpu_memory_fraction: 0.8
  memory_pressure_handling: "aggressive"  # conservative | moderate | aggressive
```

**Validation Rules:**
- All percentages: 10.0-95.0
- `cleanup_threshold_percent` < `max_memory_usage_percent`
- `monitoring_interval_seconds`: 1.0-300.0
- `gpu_memory_fraction`: 0.1-1.0

### External Services (`external_services`)

Integration with external services and databases.

```yaml
external_services:
  wandb:
    enabled: true
    project: "musicgen-ai-production"
    entity: "${WANDB_ENTITY}"
    api_key: "${WANDB_API_KEY}"
    log_frequency: "minimal"   # minimal | normal | verbose
  
  redis:
    enabled: true
    host: "${REDIS_HOST}"
    port: "${REDIS_PORT:6379}"
    password: "${REDIS_PASSWORD}"
    ssl: true
    connection_pool_size: 20
    max_connections: 100
  
  postgresql:
    enabled: true
    host: "${POSTGRES_HOST}"
    port: "${POSTGRES_PORT:5432}"
    database: "${POSTGRES_DB:musicgen_prod}"
    username: "${POSTGRES_USER}"
    password: "${POSTGRES_PASSWORD}"
    ssl_mode: "require"        # disable | allow | prefer | require | verify-ca | verify-full
    pool_size: 20
```

**Validation Rules:**
- Ports: 1-65535
- Pool sizes: 1-100
- SSL modes: Valid PostgreSQL SSL modes

### Feature Flags (`features`)

Toggle experimental and optional features.

```yaml
features:
  streaming_generation: true
  multi_instrument: false
  batch_processing: true
  conditioning: true
  model_switching: true
  audio_effects: false
  midi_export: false
```

### Development Tools (`dev_tools`)

Development and debugging utilities.

```yaml
dev_tools:
  profiling_enabled: false
  debug_sql: false
  mock_slow_operations: false
  test_mode: false
  hot_reload: true
  auto_open_browser: false
  development_server: false
```

### Validation Settings (`validation`)

Input and output validation configuration.

```yaml
validation:
  strict_mode: true
  validate_audio_output: true
  validate_model_outputs: true
  max_prompt_length: 800
  min_duration: 1.0
  max_file_size_mb: 50
```

**Validation Rules:**
- `max_prompt_length`: 1-10000
- `min_duration`: 0.01-10.0
- `max_file_size_mb`: 1-1000

## Environment-Specific Configurations

### Development Environment

Optimized for local development with debugging enabled:

```yaml
# development.yaml
defaults:
  - base
  - override /model: musicgen_small

app:
  environment: "development"
  debug: true

auth:
  enabled: false  # Simplified for development

model:
  device: "cpu"   # Avoid GPU conflicts
  compile_model: false  # Faster startup

generation:
  default_duration: 5.0  # Shorter for testing
  quality_mode: "fast"

dev_tools:
  profiling_enabled: true
  hot_reload: true
```

### Staging Environment

Production-like environment for testing:

```yaml
# staging.yaml
defaults:
  - base
  - override /model: musicgen_medium

app:
  environment: "staging"
  debug: false

auth:
  enabled: true
  jwt_secret: "${JWT_SECRET}"

api:
  workers: 2

features:
  # Enable all features for testing
  streaming_generation: true
  multi_instrument: true
  audio_effects: true
```

### Production Environment

Optimized for high availability and security:

```yaml
# production.yaml
defaults:
  - base
  - override /model: musicgen_large
  - override /infrastructure: kubernetes

app:
  environment: "production"
  debug: false

auth:
  enabled: true
  jwt_secret: "${JWT_SECRET}"
  jwt_algorithm: "RS256"
  api_key_required: true

api:
  workers: 4
  cors:
    origins:
      - "https://musicgen-ai.com"
      - "https://api.musicgen-ai.com"

resources:
  max_memory_usage_percent: 75.0  # Conservative
  memory_pressure_handling: "aggressive"

features:
  # Only stable features in production
  streaming_generation: true
  batch_processing: true
  conditioning: true
  model_switching: true
  # Disable experimental features
  multi_instrument: false
  audio_effects: false

limits:
  max_requests_per_hour: 1000
  max_generation_time_minutes: 5
  max_concurrent_generations: 10
```

## Environment Variables

The configuration system supports environment variable substitution:

### Required Production Variables

```bash
# Security
JWT_SECRET="your-strong-32-character-secret-key"
WANDB_API_KEY="your-wandb-api-key"

# Database
POSTGRES_HOST="postgres.example.com"
POSTGRES_PASSWORD="secure-password"
REDIS_HOST="redis.example.com"
REDIS_PASSWORD="redis-password"

# Storage paths
MODEL_CACHE_DIR="/app/cache/models"
AUDIO_OUTPUT_DIR="/app/outputs/audio"
```

### Optional Variables with Defaults

```bash
# Service configuration
REDIS_PORT="6379"                    # Default: 6379
POSTGRES_PORT="5432"                 # Default: 5432
POSTGRES_DB="musicgen_prod"          # Default: musicgen_prod
POSTGRES_USER="musicgen"             # Default: musicgen

# Monitoring
WANDB_ENTITY="your-org"              # Default: empty
```

### Variable Syntax

```yaml
# Basic substitution
jwt_secret: "${JWT_SECRET}"

# With default value
redis_port: "${REDIS_PORT:6379}"

# Nested in paths
cache_dir: "/app/cache/${CACHE_TYPE:models}"
```

## Validation and Type Safety

### Pydantic Models

The configuration system uses Pydantic models for comprehensive validation:

```python
from music_gen.core.config_models import MusicGenConfig

# Automatic validation on load
config = MusicGenConfig.parse_obj(config_dict)

# Manual validation
try:
    config = MusicGenConfig(**config_dict)
except ValidationError as e:
    print(f"Validation errors: {e.errors()}")
```

### Custom Validation Rules

The system includes custom validators for:

- **Environment consistency**: Production requirements
- **Resource constraints**: Memory/CPU limits
- **Device availability**: CUDA/MPS validation
- **Path accessibility**: Directory creation
- **Service connectivity**: External service validation

### Validation CLI

```bash
# Validate specific environment
python scripts/config_cli.py validate --environment production

# Validate custom file
python scripts/config_cli.py validate --file my_config.yaml

# Health check
python scripts/config_cli.py check --environment production
```

## Schema Generation

### Generate JSON Schema

```python
from music_gen.core.config_models import MusicGenConfig

# Generate complete schema
schema = MusicGenConfig.schema()

# Save to file
with open('config_schema.json', 'w') as f:
    json.dump(schema, f, indent=2)
```

```bash
# CLI command
python scripts/config_cli.py schema --output config_schema.json
```

### IDE Integration

The generated schema can be used for:

- **IDE autocomplete** in YAML files
- **Real-time validation** during editing
- **Documentation generation**
- **API documentation**

## Advanced Usage

### Dynamic Configuration Updates

```python
from music_gen.core.config_manager import get_config_manager

manager = get_config_manager()

# Update configuration at runtime
updates = {
    "api": {"workers": 8},
    "model": {"cache_size": 10}
}
updated_config = manager.update_config(updates)
```

### Configuration Composition

```yaml
# custom.yaml - Compose from multiple sources
defaults:
  - base
  - override /model: musicgen_large
  - override /infrastructure: kubernetes
  - staging  # Include staging overrides

# Additional customizations
custom_settings:
  feature_x: true
  optimization_level: "high"
```

### Programmatic Configuration

```python
from music_gen.core.config_models import MusicGenConfig, ApiConfig

# Create configuration programmatically
config = MusicGenConfig(
    app={"environment": "custom", "debug": False},
    api=ApiConfig(port=9000, workers=6),
    model={"default_model": "facebook/musicgen-large"}
)

# Validate and use
print(config.get_environment_summary())
```

### Context Manager Usage

```python
from music_gen.core.config_manager import ConfigManager

with ConfigManager() as manager:
    config = manager.load_config("production")
    # Configuration automatically cleaned up
```

## Best Practices

### Security

1. **Never commit secrets** to configuration files
2. **Use environment variables** for sensitive data
3. **Enable authentication** in staging and production
4. **Use strong JWT secrets** (≥32 characters)
5. **Enable SSL** for external services in production

### Performance

1. **Use appropriate model sizes** for each environment
2. **Configure resource limits** based on available hardware
3. **Enable model compilation** in production
4. **Use connection pooling** for databases
5. **Monitor resource usage** with appropriate thresholds

### Maintainability

1. **Use environment inheritance** to reduce duplication
2. **Document custom settings** with comments
3. **Validate configurations** before deployment
4. **Use descriptive names** for custom settings
5. **Keep environment differences** minimal and well-documented

### Testing

1. **Validate all environments** before deployment
2. **Test configuration changes** in staging first
3. **Use health checks** to verify settings
4. **Monitor configuration** impact on performance
5. **Have rollback procedures** for configuration changes

## Troubleshooting

### Common Issues

**Configuration not loading:**
```bash
# Check file paths and syntax
python scripts/config_cli.py validate --file configs/app/production.yaml
```

**Environment variables not substituted:**
```bash
# Verify environment variables are set
env | grep MUSICGEN
env | grep JWT_SECRET
```

**Validation errors:**
```bash
# Get detailed validation report
python scripts/config_cli.py check --environment production
```

**Path issues:**
```bash
# Check path permissions and creation
python scripts/config_cli.py check --environment production
```

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from music_gen.core.config_manager import load_config
config = load_config(environment="production")
```

### Error Messages

The configuration system provides detailed error messages:

```
❌ Configuration validation failed!

🚨 Errors (3):
  - auth -> jwt_secret: ensure this value has at least 32 characters
  - api -> port: ensure this value is less than or equal to 65535
  - generation -> max_duration: max_duration must be greater than default_duration

⚠️  Warnings (1):
  - Cannot create path /invalid/path: Permission denied
```

## Migration Guide

### From Previous Versions

If migrating from a previous configuration system:

1. **Export existing configuration:**
   ```bash
   python scripts/config_cli.py load --save current_config.yaml
   ```

2. **Validate against new schema:**
   ```bash
   python scripts/config_cli.py validate --file current_config.yaml
   ```

3. **Update format if needed:**
   ```bash
   python scripts/config_cli.py example --environment production --output new_config.yaml
   ```

4. **Test thoroughly:**
   ```bash
   python scripts/config_cli.py check --environment production
   ```

## API Reference

For complete API documentation, see:
- [`config_models.py`](../music_gen/core/config_models.py) - Pydantic models
- [`config_manager.py`](../music_gen/core/config_manager.py) - Configuration manager
- [`config_cli.py`](../scripts/config_cli.py) - CLI interface

## Examples

See the [`configs/`](../configs/) directory for complete configuration examples for all environments.