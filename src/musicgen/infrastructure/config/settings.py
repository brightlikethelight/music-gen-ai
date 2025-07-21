"""
Configuration management for MusicGen.

Handles loading and validation of configuration from multiple sources:
- YAML configuration files
- Environment variables
- Command line arguments
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "detailed"
    file: Optional[str] = None


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_enabled: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class ModelConfig:
    """Model configuration."""

    default: str = "facebook/musicgen-small"
    cache_dir: str = "~/.cache/musicgen"
    device: str = "auto"
    optimize: bool = True


@dataclass
class GenerationConfig:
    """Generation settings."""

    max_duration: float = 30.0
    default_duration: float = 10.0
    sample_rate: int = 32000
    batch_size: int = 1


@dataclass
class StorageConfig:
    """Storage configuration."""

    output_dir: str = "./outputs"
    temp_dir: str = "/tmp/musicgen"


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    metrics_enabled: bool = True
    metrics_port: int = 9090
    health_check_enabled: bool = True
    health_check_interval: int = 30


@dataclass
class SecurityConfig:
    """Security configuration."""

    api_key_required: bool = False
    rate_limiting_enabled: bool = True
    requests_per_minute: int = 60


@dataclass
class AppConfig:
    """Main application configuration."""

    environment: str = "development"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __post_init__(self):
        """Expand paths and validate configuration."""
        self.storage.output_dir = os.path.expanduser(self.storage.output_dir)
        self.storage.temp_dir = os.path.expanduser(self.storage.temp_dir)
        self.models.cache_dir = os.path.expanduser(self.models.cache_dir)

        # Create directories if they don't exist
        Path(self.storage.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.storage.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models.cache_dir).mkdir(parents=True, exist_ok=True)


def load_config_from_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required to load YAML configuration files")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}

    # Environment detection
    if env_val := os.getenv("MUSICGEN_ENV"):
        config["environment"] = env_val

    # API configuration
    api_config = {}
    if host := os.getenv("MUSICGEN_API_HOST"):
        api_config["host"] = host
    if port := os.getenv("MUSICGEN_API_PORT"):
        api_config["port"] = int(port)
    if workers := os.getenv("MUSICGEN_API_WORKERS"):
        api_config["workers"] = int(workers)
    if api_config:
        config["api"] = api_config

    # Model configuration
    model_config = {}
    if default_model := os.getenv("MUSICGEN_DEFAULT_MODEL"):
        model_config["default"] = default_model
    if cache_dir := os.getenv("MUSICGEN_CACHE_DIR"):
        model_config["cache_dir"] = cache_dir
    if device := os.getenv("MUSICGEN_DEVICE"):
        model_config["device"] = device
    if model_config:
        config["models"] = model_config

    # Storage configuration
    storage_config = {}
    if output_dir := os.getenv("MUSICGEN_OUTPUT_DIR"):
        storage_config["output_dir"] = output_dir
    if temp_dir := os.getenv("MUSICGEN_TEMP_DIR"):
        storage_config["temp_dir"] = temp_dir
    if storage_config:
        config["storage"] = storage_config

    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    merged = {}
    for config in configs:
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged


def dict_to_dataclass(data: Dict[str, Any], cls) -> Any:
    """Convert dictionary to dataclass instance."""
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    kwargs = {}
    for field_name, field_def in cls.__dataclass_fields__.items():
        if field_name in data:
            field_type = field_def.type
            field_value = data[field_name]

            # Handle nested dataclasses
            if hasattr(field_type, "__dataclass_fields__"):
                kwargs[field_name] = dict_to_dataclass(field_value, field_type)
            else:
                kwargs[field_name] = field_value

    return cls(**kwargs)


def load_config(
    config_file: Optional[Union[str, Path]] = None, environment: Optional[str] = None
) -> AppConfig:
    """
    Load application configuration from multiple sources.

    Args:
        config_file: Path to YAML configuration file
        environment: Environment name (development, production, testing)

    Returns:
        AppConfig instance
    """
    configs = []

    # Load from environment
    env_config = load_config_from_env()
    if env_config:
        configs.append(env_config)

    # Auto-detect environment
    if not environment:
        environment = env_config.get("environment", "development")

    # Load from YAML file
    if config_file:
        yaml_config = load_config_from_yaml(config_file)
        configs.append(yaml_config)
    else:
        # Try to auto-detect config file
        project_root = Path(__file__).parent.parent.parent.parent.parent
        config_path = project_root / "configs" / f"{environment}.yaml"
        if config_path.exists():
            yaml_config = load_config_from_yaml(config_path)
            configs.append(yaml_config)

    # Merge all configurations
    if configs:
        merged_config = merge_configs(*configs)
    else:
        merged_config = {"environment": environment}

    # Convert to dataclass
    return dict_to_dataclass(merged_config, AppConfig)


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
