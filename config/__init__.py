# config/__init__.py
"""
Configuration package for topology definitions.

Provides YAML-based topology configuration with validation.
"""

from .schema import (
    TopologyType,
    QueueConfig,
    SwitchNaming,
    SwitchDefaults,
    LinkBandwidths,
    IntCollectorConfig,
    TrafficPair,
    TrafficConfig,
    FatTreeConfig,
    LeafSpineConfig,
    ThreeTierConfig,
    TopologyConfig,
    MAX_SWITCHES,
)
from .loader import load_config

# Note: validator is imported separately to avoid circular import warnings
# when running `python3 -m config.validator`
# Use: from config.validator import TopologyValidator, validate_config

__all__ = [
    'TopologyType',
    'QueueConfig',
    'SwitchNaming',
    'SwitchDefaults',
    'LinkBandwidths',
    'IntCollectorConfig',
    'TrafficPair',
    'TrafficConfig',
    'FatTreeConfig',
    'LeafSpineConfig',
    'ThreeTierConfig',
    'TopologyConfig',
    'MAX_SWITCHES',
    'load_config',
]
