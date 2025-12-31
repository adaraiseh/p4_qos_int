# topology/factory.py
"""
Topology factory for creating topology builders.

Provides a factory pattern for instantiating the appropriate
topology builder based on configuration.
"""

from typing import Union
from pathlib import Path

from config.schema import TopologyConfig, TopologyType
from config.loader import load_config
from config.validator import validate_config

from .base import TopologyBuilder


class TopologyFactory:
    """
    Factory for creating topology builders.

    Usage:
        config = load_config('config/topologies/fat_tree_k4.yaml')
        builder = TopologyFactory.create(config)
        builder.build()
    """

    _builders = {}

    @classmethod
    def register(cls, topo_type: TopologyType, builder_class: type):
        """
        Register a builder class for a topology type.

        Args:
            topo_type: Topology type enum value
            builder_class: TopologyBuilder subclass
        """
        cls._builders[topo_type] = builder_class

    @classmethod
    def create(cls, config: TopologyConfig) -> TopologyBuilder:
        """
        Create a topology builder from configuration.

        Args:
            config: Validated topology configuration

        Returns:
            TopologyBuilder instance for the specified topology type

        Raises:
            ValueError: If topology type is not supported
        """
        topo_type = config.topology.type

        if topo_type not in cls._builders:
            # Try lazy import
            cls._lazy_import()

        if topo_type not in cls._builders:
            supported = [t.value for t in cls._builders.keys()]
            raise ValueError(
                f"Unsupported topology type: {topo_type.value}. "
                f"Supported types: {supported}"
            )

        builder_class = cls._builders[topo_type]
        return builder_class(config)

    @classmethod
    def _lazy_import(cls):
        """Lazily import all builder modules to register them."""
        # Import here to avoid circular imports
        from .fat_tree import FatTreeBuilder
        from .leaf_spine import LeafSpineBuilder
        from .three_tier import ThreeTierBuilder

        # Registration happens in each module's import


def create_topology(config_path: Union[str, Path], validate: bool = True) -> TopologyBuilder:
    """
    Convenience function to create a topology from a config file.

    Args:
        config_path: Path to YAML configuration file
        validate: Whether to validate the configuration first

    Returns:
        Built TopologyBuilder instance (with build() already called)

    Raises:
        ConfigLoadError: If configuration cannot be loaded
        ValidationError: If configuration is invalid (when validate=True)
    """
    if validate:
        result = validate_config(config_path)
        if not result.is_valid:
            raise ValueError(f"Configuration validation failed:\n{result}")
        config = result.config
    else:
        config = load_config(config_path)

    builder = TopologyFactory.create(config)
    builder.build()
    return builder


def create_topology_from_config(config: TopologyConfig) -> TopologyBuilder:
    """
    Create a topology from a validated configuration.

    Args:
        config: Validated TopologyConfig

    Returns:
        Built TopologyBuilder instance (with build() already called)
    """
    builder = TopologyFactory.create(config)
    builder.build()
    return builder
