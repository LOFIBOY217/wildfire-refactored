"""
Configuration Management
========================
Loads YAML config files with environment variable expansion and path resolution.

Usage:
    from src.config import load_config
    cfg = load_config()                          # loads configs/default.yaml
    cfg = load_config("configs/paths_windows.yaml")  # loads with overrides

    # Access values
    cfg['paths']['fwi_dir']
    cfg['model']['transformer']['d_model']
"""

import os
import re
import yaml
from pathlib import Path

# Project root: directory containing src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_cached_config = None
_cached_path = None


def _expand_env_vars(value):
    """Expand ${ENV_VAR} references in string values."""
    if not isinstance(value, str):
        return value

    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(r'\$\{(\w+)\}', replacer, value)


def _expand_recursive(obj):
    """Recursively expand environment variables in a config dict."""
    if isinstance(obj, dict):
        return {k: _expand_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return _expand_env_vars(obj)
    return obj


def _deep_merge(base, override):
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path=None):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, loads configs/default.yaml.
                     If a non-default path is given, it's merged on top of defaults.

    Returns:
        dict: Configuration dictionary with env vars expanded.
    """
    global _cached_config, _cached_path

    if config_path is not None and _cached_path == config_path and _cached_config is not None:
        return _cached_config

    # Load defaults
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    if default_path.exists():
        with open(default_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Load overrides if specified
    if config_path is not None:
        override_path = Path(config_path)
        if not override_path.is_absolute():
            override_path = PROJECT_ROOT / override_path

        if override_path.exists() and str(override_path) != str(default_path):
            with open(override_path, 'r') as f:
                overrides = yaml.safe_load(f) or {}
            config = _deep_merge(config, overrides)

    # Expand environment variables
    config = _expand_recursive(config)

    # Cache
    _cached_config = config
    _cached_path = config_path

    return config


def get_path(config, key):
    """
    Get a path from config, resolving relative paths against project root.

    Args:
        config: Config dict from load_config()
        key: Key in config['paths'], e.g. 'fwi_dir'

    Returns:
        str: Resolved absolute path
    """
    path_str = config['paths'][key]
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def add_config_argument(parser):
    """Add --config argument to an argparse parser."""
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: configs/default.yaml)"
    )
