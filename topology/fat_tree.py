# topology/fat_tree.py
"""
Fat-Tree topology builder.

Implements a k-ary Fat-Tree (Clos) topology:
- k pods
- k/2 leaf (ToR) switches per pod
- k/2 spine (aggregation) switches per pod
- (k/2)^2 core switches
- k/2 hosts per leaf switch
"""

from typing import Dict, List, Tuple

from config.schema import TopologyConfig, TopologyType
from .base import TopologyBuilder, SwitchInfo
from .factory import TopologyFactory


class FatTreeBuilder(TopologyBuilder):
    """
    Builder for k-ary Fat-Tree topology.

    Fat-Tree structure:
    - Organized into k pods
    - Each pod has k/2 leaf switches and k/2 spine switches
    - Each leaf connects to all spines in its pod
    - Each spine connects to k/2 core switches
    - Core switches are shared across pods
    """

    def __init__(self, config: TopologyConfig):
        super().__init__(config)
        self.ft_config = config.fat_tree
        self.k = self.ft_config.k

    def build(self) -> None:
        """Build the Fat-Tree topology."""
        # 1. Create switches
        self._create_core_switches()
        self._create_pod_switches()

        # 2. Create hosts
        self._create_hosts()

        # 3. Create links
        self._create_links()

        # 4. Set up INT collectors
        self._setup_collectors()

    def _create_core_switches(self) -> None:
        """Create core switches."""
        prefix = self.config.switch_naming.core_prefix
        num_cores = self.ft_config.num_cores

        for i in range(1, num_cores + 1):
            name = f"{prefix}{i}"
            self._add_switch(name, role='core')

    def _create_pod_switches(self) -> None:
        """Create leaf and spine switches for each pod."""
        leaf_prefix = self.config.switch_naming.leaf_prefix
        spine_prefix = self.config.switch_naming.spine_prefix

        leaves_per_pod = self.ft_config.leaves_per_pod
        spines_per_pod = self.ft_config.spines_per_pod

        leaf_counter = 1
        spine_counter = 1

        for pod_id in range(self.k):
            # Create leaf switches for this pod
            for _ in range(leaves_per_pod):
                name = f"{leaf_prefix}{leaf_counter}"
                self._add_switch(name, role='leaf', pod_id=pod_id)
                leaf_counter += 1

            # Create spine switches for this pod
            for _ in range(spines_per_pod):
                name = f"{spine_prefix}{spine_counter}"
                self._add_switch(name, role='spine', pod_id=pod_id)
                spine_counter += 1

    def _create_hosts(self) -> None:
        """Create hosts and connect to leaf switches."""
        host_prefix = self.config.switch_naming.host_prefix
        hosts_per_leaf = self.ft_config.hosts_per_leaf

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
        self._create_spine_core_links()

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
        """Create links between leaves and spines within each pod."""
        bw = self.config.link_bandwidths.leaf_spine
        hosts_per_leaf = self.ft_config.hosts_per_leaf

        # Group switches by pod
        pods = {}
        for sw in self.switches.values():
            if sw.pod_id is not None:
                if sw.pod_id not in pods:
                    pods[sw.pod_id] = {'leaves': [], 'spines': []}
                if sw.role == 'leaf':
                    pods[sw.pod_id]['leaves'].append(sw)
                elif sw.role == 'spine':
                    pods[sw.pod_id]['spines'].append(sw)

        # Connect each leaf to all spines in the same pod
        for pod_id, switches in pods.items():
            for leaf in switches['leaves']:
                # Ports on leaf: 1..hosts_per_leaf are for hosts
                # Spine ports start at hosts_per_leaf + 1
                leaf_port = hosts_per_leaf + 1
                for spine in switches['spines']:
                    self._add_link(
                        leaf.name,
                        spine.name,
                        bw=bw,
                        port1=leaf_port,
                    )
                    leaf_port += 1

    def _create_spine_core_links(self) -> None:
        """Create links between spines and cores."""
        bw = self.config.link_bandwidths.spine_core

        # Get all core switches
        cores = sorted([
            sw for sw in self.switches.values() if sw.role == 'core'
        ], key=lambda s: s.name)

        # Get all spine switches grouped by pod
        pods = {}
        for sw in self.switches.values():
            if sw.role == 'spine' and sw.pod_id is not None:
                if sw.pod_id not in pods:
                    pods[sw.pod_id] = []
                pods[sw.pod_id].append(sw)

        # In a k-ary Fat-Tree:
        # - There are (k/2)^2 core switches
        # - Each spine in a pod connects to k/2 core switches
        # - Spine j in each pod connects to core switches ((k/2) * (j-1)) to ((k/2) * j - 1)
        half_k = self.k // 2

        for pod_id in sorted(pods.keys()):
            spines = sorted(pods[pod_id], key=lambda s: s.name)
            for spine_idx, spine in enumerate(spines):
                # Each spine connects to half_k cores
                core_start = spine_idx * half_k
                for core_offset in range(half_k):
                    core_idx = core_start + core_offset
                    if core_idx < len(cores):
                        core = cores[core_idx]
                        self._add_link(spine.name, core.name, bw=bw)

    def get_layout_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute positions for Fat-Tree visualization.

        Layout:
        - Y=0: Hosts
        - Y=2: Leaf switches
        - Y=4: Spine switches
        - Y=6: Core switches
        """
        pos = {}
        width = self.k * 6.0  # Scale based on k

        # Core switches at top (Y=6)
        cores = sorted([
            sw for sw in self.switches.values() if sw.role == 'core'
        ], key=lambda s: s.name)

        if cores:
            core_spacing = width / (len(cores) + 1)
            for i, core in enumerate(cores):
                pos[core.name] = ((i + 1) * core_spacing, 6.0)

        # Pod-based layout for leaves and spines
        pod_width = width / self.k

        # Group switches by pod
        pods = {}
        for sw in self.switches.values():
            if sw.pod_id is not None:
                if sw.pod_id not in pods:
                    pods[sw.pod_id] = {'leaves': [], 'spines': []}
                if sw.role == 'leaf':
                    pods[sw.pod_id]['leaves'].append(sw)
                elif sw.role == 'spine':
                    pods[sw.pod_id]['spines'].append(sw)

        for pod_id in sorted(pods.keys()):
            pod_center = pod_width * (pod_id + 0.5)
            leaves = sorted(pods[pod_id]['leaves'], key=lambda s: s.name)
            spines = sorted(pods[pod_id]['spines'], key=lambda s: s.name)

            # Spine switches (Y=4)
            if spines:
                spine_spacing = pod_width / (len(spines) + 1)
                for i, spine in enumerate(spines):
                    x = pod_center - pod_width / 2 + (i + 1) * spine_spacing
                    pos[spine.name] = (x, 4.0)

            # Leaf switches (Y=2)
            if leaves:
                leaf_spacing = pod_width / (len(leaves) + 1)
                for i, leaf in enumerate(leaves):
                    x = pod_center - pod_width / 2 + (i + 1) * leaf_spacing
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
TopologyFactory.register(TopologyType.FAT_TREE, FatTreeBuilder)
