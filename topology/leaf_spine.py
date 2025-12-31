# topology/leaf_spine.py
"""
Leaf-Spine topology builder.

Implements a simple two-tier Leaf-Spine topology:
- N leaf switches
- M spine switches
- Each leaf connects to ALL spines (full mesh)
- H hosts per leaf switch
"""

from typing import Dict, Tuple

from config.schema import TopologyConfig, TopologyType
from .base import TopologyBuilder
from .factory import TopologyFactory


class LeafSpineBuilder(TopologyBuilder):
    """
    Builder for Leaf-Spine topology.

    Two-tier design where every leaf connects to every spine,
    providing multiple equal-cost paths between any two leaves.
    """

    def __init__(self, config: TopologyConfig):
        super().__init__(config)
        self.ls_config = config.leaf_spine

    def build(self) -> None:
        """Build the Leaf-Spine topology."""
        # 1. Create switches
        self._create_spine_switches()
        self._create_leaf_switches()

        # 2. Create hosts
        self._create_hosts()

        # 3. Create links
        self._create_links()

        # 4. Set up INT collectors
        self._setup_collectors()

    def _create_spine_switches(self) -> None:
        """Create spine switches."""
        prefix = self.config.switch_naming.spine_prefix
        num_spines = self.ls_config.num_spines

        for i in range(1, num_spines + 1):
            name = f"{prefix}{i}"
            self._add_switch(name, role='spine')

    def _create_leaf_switches(self) -> None:
        """Create leaf switches."""
        prefix = self.config.switch_naming.leaf_prefix
        num_leaves = self.ls_config.num_leaves

        for i in range(1, num_leaves + 1):
            name = f"{prefix}{i}"
            self._add_switch(name, role='leaf')

    def _create_hosts(self) -> None:
        """Create hosts and connect to leaf switches."""
        host_prefix = self.config.switch_naming.host_prefix
        hosts_per_leaf = self.ls_config.hosts_per_leaf

        # Get all leaf switches sorted by name
        leaves = sorted([
            sw for sw in self.switches.values() if sw.role == 'leaf'
        ], key=lambda s: s.name)

        host_counter = 1
        for leaf in leaves:
            for port in range(1, hosts_per_leaf + 1):
                host_name = f"{host_prefix}{host_counter}"
                self._add_host(host_name, leaf.name, port)
                host_counter += 1

    def _create_links(self) -> None:
        """Create all links in the topology."""
        self._create_host_leaf_links()
        self._create_leaf_spine_links()

    def _create_host_leaf_links(self) -> None:
        """Create links between hosts and leaf switches."""
        bw = self.config.link_bandwidths.host_leaf

        for host in self.hosts.values():
            self._add_link(
                host.name,
                host.connected_switch,
                bw=bw,
                port2=host.port_on_switch
            )

    def _create_leaf_spine_links(self) -> None:
        """Create links between each leaf and all spines."""
        bw = self.config.link_bandwidths.leaf_spine
        hosts_per_leaf = self.ls_config.hosts_per_leaf

        leaves = sorted([
            sw for sw in self.switches.values() if sw.role == 'leaf'
        ], key=lambda s: s.name)

        spines = sorted([
            sw for sw in self.switches.values() if sw.role == 'spine'
        ], key=lambda s: s.name)

        # Each leaf connects to all spines
        for leaf in leaves:
            # Ports on leaf: 1..hosts_per_leaf are for hosts
            # Spine ports start at hosts_per_leaf + 1
            leaf_port = hosts_per_leaf + 1
            for spine in spines:
                self._add_link(
                    leaf.name,
                    spine.name,
                    bw=bw,
                    port1=leaf_port,
                )
                leaf_port += 1

    def get_layout_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute positions for Leaf-Spine visualization.

        Layout:
        - Y=0: Hosts
        - Y=2: Leaf switches
        - Y=4: Spine switches
        """
        pos = {}
        num_leaves = self.ls_config.num_leaves
        num_spines = self.ls_config.num_spines
        width = max(num_leaves, num_spines) * 4.0

        # Spine switches (Y=4)
        spines = sorted([
            sw for sw in self.switches.values() if sw.role == 'spine'
        ], key=lambda s: s.name)

        if spines:
            spine_spacing = width / (len(spines) + 1)
            for i, spine in enumerate(spines):
                pos[spine.name] = ((i + 1) * spine_spacing, 4.0)

        # Leaf switches (Y=2)
        leaves = sorted([
            sw for sw in self.switches.values() if sw.role == 'leaf'
        ], key=lambda s: s.name)

        if leaves:
            leaf_spacing = width / (len(leaves) + 1)
            for i, leaf in enumerate(leaves):
                x = (i + 1) * leaf_spacing
                pos[leaf.name] = (x, 2.0)

                # Hosts under this leaf (Y=0)
                leaf_hosts = sorted([
                    h for h in self.hosts.values()
                    if h.connected_switch == leaf.name
                ], key=lambda h: h.name)

                if leaf_hosts:
                    host_spread = leaf_spacing * 0.8
                    for j, host in enumerate(leaf_hosts):
                        hx = x - host_spread / 2 + (j + 0.5) * host_spread / len(leaf_hosts)
                        pos[host.name] = (hx, 0.0)

        return pos


# Register with factory
TopologyFactory.register(TopologyType.LEAF_SPINE, LeafSpineBuilder)
