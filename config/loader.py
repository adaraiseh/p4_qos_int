# config/loader.py
"""
YAML configuration loader.

Loads and parses topology configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from .schema import TopologyConfig


class ConfigLoadError(Exception):
    """Error loading configuration file."""
    pass


def load_config(config_path: Union[str, Path]) -> TopologyConfig:
    """
    Load topology configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        TopologyConfig: Validated topology configuration

    Raises:
        ConfigLoadError: If file cannot be loaded or parsed
        ValidationError: If configuration is invalid
    """
    config_path = Path(config_path)

    # Check file exists
    if not config_path.exists():
        raise ConfigLoadError(f"Configuration file not found: {config_path}")

    if not config_path.is_file():
        raise ConfigLoadError(f"Not a file: {config_path}")

    # Check extension
    if config_path.suffix.lower() not in ('.yaml', '.yml'):
        raise ConfigLoadError(f"Expected YAML file, got: {config_path.suffix}")

    # Load YAML
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"YAML parsing error in {config_path}: {e}")
    except IOError as e:
        raise ConfigLoadError(f"Failed to read {config_path}: {e}")

    if raw_config is None:
        raise ConfigLoadError(f"Empty configuration file: {config_path}")

    if not isinstance(raw_config, dict):
        raise ConfigLoadError(f"Configuration must be a dictionary, got: {type(raw_config)}")

    # Parse and validate with Pydantic
    try:
        config = TopologyConfig(**raw_config)
    except ValidationError as e:
        # Re-raise with file context
        raise ConfigLoadError(
            f"Configuration validation failed for {config_path}:\n{e}"
        )

    return config


def load_config_dict(raw_config: dict) -> TopologyConfig:
    """
    Load topology configuration from a dictionary.

    Args:
        raw_config: Configuration dictionary

    Returns:
        TopologyConfig: Validated topology configuration

    Raises:
        ValidationError: If configuration is invalid
    """
    return TopologyConfig(**raw_config)


def find_config(topology_name: str, search_paths: list[Path] = None) -> Path:
    """
    Find a topology configuration file by name.

    Searches in:
    1. Current directory
    2. config/topologies/
    3. Custom search paths

    Args:
        topology_name: Name of topology (with or without .yaml extension)
        search_paths: Additional paths to search

    Returns:
        Path to configuration file

    Raises:
        ConfigLoadError: If configuration not found
    """
    # Ensure .yaml extension
    if not topology_name.endswith(('.yaml', '.yml')):
        topology_name = f"{topology_name}.yaml"

    # Build search paths
    paths = [
        Path.cwd() / topology_name,
        Path.cwd() / 'config' / 'topologies' / topology_name,
        Path(__file__).parent / 'topologies' / topology_name,
    ]

    if search_paths:
        for sp in search_paths:
            paths.append(Path(sp) / topology_name)

    # Search
    for path in paths:
        if path.exists() and path.is_file():
            return path

    # Not found
    searched = '\n  - '.join(str(p) for p in paths)
    raise ConfigLoadError(
        f"Configuration '{topology_name}' not found. Searched:\n  - {searched}"
    )
