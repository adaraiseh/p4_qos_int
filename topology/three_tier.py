# topology/three_tier.py
"""
Three-Tier Hierarchical topology builder.

Implements a classic enterprise three-tier topology:
- Access layer (edge) switches
- Distribution layer switches
- Core layer switches
- Hosts connect to access switches
- Access switches connect to distribution (based on pod grouping)
- Distribution switches connect to all core switches
"""

from typing import Dict, Tuple, List

from config.schema import TopologyConfig, TopologyType
from .base import TopologyBuilder
from .factory import TopologyFactory


class ThreeTierBuilder(TopologyBuilder):
    """
    Builder for Three-Tier Hierarchical topology.

    Classic enterprise design with access, distribution, and core layers.
    Access switches are grouped into pods, each served by distribution switches.
    """

    def __init__(self, config: TopologyConfig):
        super().__init__(config)
        self.tt_config = config.three_tier

    def build(self) -> None:
        """Build the Three-Tier topology."""
        # 1. Create switches
        self._create_core_switches()
        self._create_distribution_switches()
        self._create_access_switches()

        # 2. Create hosts
        self._create_hosts()

        # 3. Create links
        self._create_links()

        # 4. Set up INT collectors
        self._setup_collectors()

    def _create_core_switches(self) -> None:
        """Create core switches."""
        prefix = self.config.switch_naming.core_prefix
        num_cores = self.tt_config.core_switches

        for i in range(1, num_cores + 1):
            name = f"{prefix}{i}"
            self._add_switch(name, role='core')

    def _create_distribution_switches(self) -> None:
        """Create distribution switches."""
        prefix = self.config.switch_naming.distribution_prefix
        num_dist = self.tt_config.distribution_switches

        for i in range(1, num_dist + 1):
            name = f"{prefix}{i}"
            self._add_switch(name, role='distribution')

    def _create_access_switches(self) -> None:
        """Create access switches with pod assignments."""
        prefix = self.config.switch_naming.access_prefix
        num_access = self.tt_config.access_switches

        # If pods are defined, use them for pod_id assignment
        access_to_pod = {}
        if self.tt_config.pods:
            for pod_idx, pod in enumerate(self.tt_config.pods):
                for access_id in pod.access:
                    access_to_pod[access_id] = pod_idx

        for i in range(1, num_access + 1):
            name = f"{prefix}{i}"
            pod_id = access_to_pod.get(i, i - 1)  # Default: one pod per access
            self._add_switch(name, role='access', pod_id=pod_id)

    def _create_hosts(self) -> None:
        """Create hosts and connect to access switches."""
        host_prefix = self.config.switch_naming.host_prefix
        hosts_per_access = self.tt_config.hosts_per_access

        # Get all access switches sorted by name
        access_switches = sorted([
            sw for sw in self.switches.values() if sw.role == 'access'
        ], key=lambda s: s.name)

        host_counter = 1
        for access in access_switches:
            for port in range(1, hosts_per_access + 1):
                host_name = f"{host_prefix}{host_counter}"
                self._add_host(host_name, access.name, port)
                host_counter += 1

    def _create_links(self) -> None:
        """Create all links in the topology."""
        self._create_host_access_links()
        self._create_access_distribution_links()
        self._create_distribution_core_links()

    def _create_host_access_links(self) -> None:
        """Create links between hosts and access switches."""
        bw = self.config.link_bandwidths.host_access

        for host in self.hosts.values():
            self._add_link(
                host.name,
                host.connected_switch,
                bw=bw,
                port2=host.port_on_switch
            )

    def _create_access_distribution_links(self) -> None:
        """Create links between access and distribution switches."""
        bw = self.config.link_bandwidths.access_distribution
        hosts_per_access = self.tt_config.hosts_per_access

        access_switches = sorted([
            sw for sw in self.switches.values() if sw.role == 'access'
        ], key=lambda s: s.name)

        dist_switches = sorted([
            sw for sw in self.switches.values() if sw.role == 'distribution'
        ], key=lambda s: s.name)

        # Build access-to-distribution mapping from pods
        access_to_dist = {}
        if self.tt_config.pods:
            for pod in self.tt_config.pods:
                for access_id in pod.access:
                    access_name = f"{self.config.switch_naming.access_prefix}{access_id}"
                    dist_names = [
                        f"{self.config.switch_naming.distribution_prefix}{d}"
                        for d in pod.distribution
                    ]
                    access_to_dist[access_name] = dist_names
        else:
            # Default: each access connects to all distribution switches
            for access in access_switches:
                access_to_dist[access.name] = [d.name for d in dist_switches]

        # Create links
        for access in access_switches:
            dist_names = access_to_dist.get(access.name, [d.name for d in dist_switches])
            # Ports on access: 1..hosts_per_access are for hosts
            access_port = hosts_per_access + 1
            for dist_name in dist_names:
                self._add_link(
                    access.name,
                    dist_name,
                    bw=bw,
                    port1=access_port,
                )
                access_port += 1

    def _create_distribution_core_links(self) -> None:
        """Create links between distribution and core switches."""
        bw = self.config.link_bandwidths.distribution_core

        dist_switches = sorted([
            sw for sw in self.switches.values() if sw.role == 'distribution'
        ], key=lambda s: s.name)

        core_switches = sorted([
            sw for sw in self.switches.values() if sw.role == 'core'
        ], key=lambda s: s.name)

        # Each distribution connects to all core switches
        for dist in dist_switches:
            for core in core_switches:
                self._add_link(dist.name, core.name, bw=bw)

    def get_layout_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute positions for Three-Tier visualization.

        Layout:
        - Y=0: Hosts
        - Y=2: Access switches
        - Y=4: Distribution switches
        - Y=6: Core switches
        """
        pos = {}
        num_access = self.tt_config.access_switches
        num_dist = self.tt_config.distribution_switches
        num_core = self.tt_config.core_switches
        width = max(num_access, num_dist, num_core) * 4.0

        # Core switches (Y=6)
        cores = sorted([
            sw for sw in self.switches.values() if sw.role == 'core'
        ], key=lambda s: s.name)

        if cores:
            core_spacing = width / (len(cores) + 1)
            for i, core in enumerate(cores):
                pos[core.name] = ((i + 1) * core_spacing, 6.0)

        # Distribution switches (Y=4)
        dists = sorted([
            sw for sw in self.switches.values() if sw.role == 'distribution'
        ], key=lambda s: s.name)

        if dists:
            dist_spacing = width / (len(dists) + 1)
            for i, dist in enumerate(dists):
                pos[dist.name] = ((i + 1) * dist_spacing, 4.0)

        # Access switches (Y=2)
        access_list = sorted([
            sw for sw in self.switches.values() if sw.role == 'access'
        ], key=lambda s: s.name)

        if access_list:
            access_spacing = width / (len(access_list) + 1)
            for i, access in enumerate(access_list):
                x = (i + 1) * access_spacing
                pos[access.name] = (x, 2.0)

                # Hosts under this access switch (Y=0)
                access_hosts = sorted([
                    h for h in self.hosts.values()
                    if h.connected_switch == access.name
                ], key=lambda h: h.name)

                if access_hosts:
                    host_spread = access_spacing * 0.8
                    for j, host in enumerate(access_hosts):
                        hx = x - host_spread / 2 + (j + 0.5) * host_spread / len(access_hosts)
                        pos[host.name] = (hx, 0.0)

        return pos


# Register with factory
TopologyFactory.register(TopologyType.THREE_TIER, ThreeTierBuilder)
