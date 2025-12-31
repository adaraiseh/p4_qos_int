# topology/__init__.py
"""
Topology builder package.

Provides builders for creating network topologies:
- Fat-Tree (k-ary)
- Leaf-Spine
- Three-Tier Hierarchical
"""

from .base import TopologyBuilder, SwitchInfo, HostInfo, LinkInfo, CollectorInfo
from .factory import TopologyFactory, create_topology
from .fat_tree import FatTreeBuilder
from .leaf_spine import LeafSpineBuilder
from .three_tier import ThreeTierBuilder

__all__ = [
    'TopologyBuilder',
    'SwitchInfo',
    'HostInfo',
    'LinkInfo',
    'CollectorInfo',
    'TopologyFactory',
    'create_topology',
    'FatTreeBuilder',
    'LeafSpineBuilder',
    'ThreeTierBuilder',
]
