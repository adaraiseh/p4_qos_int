# config/schema.py
"""
Pydantic models for topology configuration.

Supports three topology types:
- Fat-Tree (k-ary)
- Leaf-Spine
- Three-Tier Hierarchical
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


# Maximum switches supported by RL agent (fixed-size input)
MAX_SWITCHES = 20


class TopologyType(str, Enum):
    """Supported topology types."""
    FAT_TREE = "fat-tree"
    LEAF_SPINE = "leaf-spine"
    THREE_TIER = "three-tier"


class QueueConfig(BaseModel):
    """QoS queue configuration."""
    id: int = Field(..., ge=0, le=7, description="Queue ID (0-7)")
    name: str = Field(..., description="Queue name (e.g., 'voice', 'video')")
    dscp: str = Field(..., description="DSCP value in hex (e.g., '0x2E')")
    sla_ms: int = Field(..., ge=1, description="SLA threshold in milliseconds")


class SwitchNaming(BaseModel):
    """Switch naming conventions."""
    leaf_prefix: str = Field(default="leaf", description="Prefix for leaf/ToR switches")
    spine_prefix: str = Field(default="spine", description="Prefix for spine/agg switches")
    core_prefix: str = Field(default="core", description="Prefix for core switches")
    access_prefix: str = Field(default="access", description="Prefix for access switches (three-tier)")
    distribution_prefix: str = Field(default="dist", description="Prefix for distribution switches")
    host_prefix: str = Field(default="h", description="Prefix for hosts")
    collector_prefix: str = Field(default="h", description="Prefix for collector hosts")
    collector_id_base: int = Field(default=100, ge=100, description="Starting ID for collector hosts")


class SwitchDefaults(BaseModel):
    """Default switch configuration."""
    p4_source: str = Field(default="p4src/int_md.p4", description="P4 source file")
    priority_queues: int = Field(default=8, ge=1, le=16, description="Number of priority queues")
    thrift_port_base: int = Field(default=9200, ge=9000, description="Starting thrift port")


class LinkBandwidths(BaseModel):
    """Link bandwidth configuration in Mbps."""
    host_leaf: int = Field(default=10, ge=1, description="Host to leaf bandwidth")
    leaf_spine: int = Field(default=5, ge=1, description="Leaf to spine bandwidth")
    spine_core: int = Field(default=10, ge=1, description="Spine to core bandwidth")
    # Three-tier specific
    host_access: int = Field(default=10, ge=1, description="Host to access bandwidth")
    access_distribution: int = Field(default=5, ge=1, description="Access to distribution bandwidth")
    distribution_core: int = Field(default=10, ge=1, description="Distribution to core bandwidth")


class IntCollectorConfig(BaseModel):
    """INT collector configuration."""
    ratio: int = Field(default=4, ge=1, le=8, description="One collector per N leaf switches")
    ip_base: str = Field(default="172.16", description="Base IP for collector network")
    port: int = Field(default=10, ge=1, description="Port number on leaf switches for mirroring")


class TrafficPair(BaseModel):
    """A single traffic pair (sender -> receiver)."""
    src: str = Field(..., description="Source host name (e.g., 'h1')")
    dst: str = Field(..., description="Destination host name (e.g., 'h2')")


class TrafficConfig(BaseModel):
    """Traffic generation configuration."""
    # Explicit pairs take precedence
    pairs: List[TrafficPair] = Field(
        default_factory=list,
        description="Explicit list of (src, dst) traffic pairs"
    )
    # Alternative: pattern-based generation
    pattern: Optional[str] = Field(
        default=None,
        description="Traffic pattern: 'cross_pod' (default), 'all_to_all', 'random_N'"
    )
    # Pod size for cross_pod pattern (hosts per pod)
    pod_size: int = Field(default=2, ge=1, le=8, description="Hosts per pod for cross_pod pattern")

    @model_validator(mode='after')
    def validate_traffic_config(self) -> 'TrafficConfig':
        """Ensure either pairs or pattern is specified."""
        if not self.pairs and not self.pattern:
            # Default to cross_pod pattern if nothing specified
            self.pattern = "cross_pod"
        return self


class FatTreeConfig(BaseModel):
    """Fat-Tree topology configuration."""
    k: int = Field(..., ge=2, le=16, description="Fat-tree k-value (must be even)")

    @field_validator('k')
    @classmethod
    def k_must_be_even(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError(f"Fat-tree k={v} must be even")
        return v

    @property
    def num_pods(self) -> int:
        """Number of pods."""
        return self.k

    @property
    def leaves_per_pod(self) -> int:
        """Number of leaf/ToR switches per pod."""
        return self.k // 2

    @property
    def spines_per_pod(self) -> int:
        """Number of spine/agg switches per pod."""
        return self.k // 2

    @property
    def num_cores(self) -> int:
        """Number of core switches."""
        return (self.k // 2) ** 2

    @property
    def hosts_per_leaf(self) -> int:
        """Number of hosts per leaf switch."""
        return self.k // 2

    @property
    def total_leaves(self) -> int:
        """Total number of leaf switches."""
        return self.k * self.leaves_per_pod

    @property
    def total_spines(self) -> int:
        """Total number of spine switches."""
        return self.k * self.spines_per_pod

    @property
    def total_switches(self) -> int:
        """Total number of switches."""
        return self.total_leaves + self.total_spines + self.num_cores

    @property
    def total_hosts(self) -> int:
        """Total number of hosts."""
        return self.total_leaves * self.hosts_per_leaf


class LeafSpineConfig(BaseModel):
    """Leaf-Spine topology configuration."""
    num_leaves: int = Field(..., ge=2, le=20, description="Number of leaf switches")
    num_spines: int = Field(..., ge=1, le=10, description="Number of spine switches")
    hosts_per_leaf: int = Field(default=2, ge=1, le=8, description="Hosts per leaf switch")

    @property
    def total_switches(self) -> int:
        """Total number of switches."""
        return self.num_leaves + self.num_spines

    @property
    def total_hosts(self) -> int:
        """Total number of hosts."""
        return self.num_leaves * self.hosts_per_leaf


class ThreeTierPod(BaseModel):
    """Pod definition for three-tier topology."""
    access: List[int] = Field(..., description="Access switch IDs in this pod")
    distribution: List[int] = Field(..., description="Distribution switch IDs for this pod")


class ThreeTierConfig(BaseModel):
    """Three-Tier topology configuration."""
    access_switches: int = Field(..., ge=2, le=16, description="Number of access switches")
    distribution_switches: int = Field(..., ge=1, le=8, description="Number of distribution switches")
    core_switches: int = Field(..., ge=1, le=4, description="Number of core switches")
    hosts_per_access: int = Field(default=2, ge=1, le=8, description="Hosts per access switch")
    pods: List[ThreeTierPod] = Field(default_factory=list, description="Pod groupings")

    @property
    def total_switches(self) -> int:
        """Total number of switches."""
        return self.access_switches + self.distribution_switches + self.core_switches

    @property
    def total_hosts(self) -> int:
        """Total number of hosts."""
        return self.access_switches * self.hosts_per_access


class TopologyInfo(BaseModel):
    """Topology identification."""
    type: TopologyType = Field(..., description="Topology type")
    name: str = Field(..., description="Topology name for identification")


class TopologyConfig(BaseModel):
    """Complete topology configuration."""
    version: str = Field(default="1.0", description="Configuration version")
    topology: TopologyInfo = Field(..., description="Topology identification")

    # Type-specific configurations (only one should be populated)
    fat_tree: Optional[FatTreeConfig] = Field(default=None, description="Fat-tree config")
    leaf_spine: Optional[LeafSpineConfig] = Field(default=None, description="Leaf-spine config")
    three_tier: Optional[ThreeTierConfig] = Field(default=None, description="Three-tier config")

    # Common configurations with defaults
    switch_naming: SwitchNaming = Field(default_factory=SwitchNaming)
    switch_defaults: SwitchDefaults = Field(default_factory=SwitchDefaults)
    link_bandwidths: LinkBandwidths = Field(default_factory=LinkBandwidths)
    int_collectors: IntCollectorConfig = Field(default_factory=IntCollectorConfig)
    traffic: TrafficConfig = Field(default_factory=TrafficConfig)

    # QoS configuration (fixed 3 queues)
    qos_queues: List[QueueConfig] = Field(
        default_factory=lambda: [
            QueueConfig(id=0, name="voice", dscp="0x2E", sla_ms=100),
            QueueConfig(id=1, name="video", dscp="0x18", sla_ms=150),
            QueueConfig(id=7, name="best_effort", dscp="0x00", sla_ms=200),
        ],
        description="QoS queue definitions"
    )

    @model_validator(mode='after')
    def validate_topology_config_present(self) -> 'TopologyConfig':
        """Ensure the correct type-specific config is present."""
        topo_type = self.topology.type

        if topo_type == TopologyType.FAT_TREE:
            if self.fat_tree is None:
                raise ValueError("fat_tree configuration required for fat-tree topology")
        elif topo_type == TopologyType.LEAF_SPINE:
            if self.leaf_spine is None:
                raise ValueError("leaf_spine configuration required for leaf-spine topology")
        elif topo_type == TopologyType.THREE_TIER:
            if self.three_tier is None:
                raise ValueError("three_tier configuration required for three-tier topology")

        return self

    @property
    def total_switches(self) -> int:
        """Get total switch count for this topology."""
        if self.topology.type == TopologyType.FAT_TREE:
            return self.fat_tree.total_switches
        elif self.topology.type == TopologyType.LEAF_SPINE:
            return self.leaf_spine.total_switches
        elif self.topology.type == TopologyType.THREE_TIER:
            return self.three_tier.total_switches
        return 0

    @property
    def total_hosts(self) -> int:
        """Get total host count for this topology."""
        if self.topology.type == TopologyType.FAT_TREE:
            return self.fat_tree.total_hosts
        elif self.topology.type == TopologyType.LEAF_SPINE:
            return self.leaf_spine.total_hosts
        elif self.topology.type == TopologyType.THREE_TIER:
            return self.three_tier.total_hosts
        return 0

    def get_type_config(self) -> FatTreeConfig | LeafSpineConfig | ThreeTierConfig:
        """Get the type-specific configuration."""
        if self.topology.type == TopologyType.FAT_TREE:
            return self.fat_tree
        elif self.topology.type == TopologyType.LEAF_SPINE:
            return self.leaf_spine
        elif self.topology.type == TopologyType.THREE_TIER:
            return self.three_tier
        raise ValueError(f"Unknown topology type: {self.topology.type}")
