# topology/base.py
"""
Abstract base class for topology builders.

Defines the interface for building network topologies and
provides common utilities for all topology types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from config.schema import TopologyConfig, TopologyType


@dataclass
class SwitchInfo:
    """Switch metadata container."""
    name: str
    role: str  # "leaf", "spine", "core", "access", "distribution"
    switch_id: int
    thrift_port: int
    p4_source: str
    pod_id: Optional[int] = None  # Pod assignment for hierarchical topologies
    connected_hosts: List[str] = field(default_factory=list)
    connected_switches: List[str] = field(default_factory=list)
    is_int_source: bool = False  # Has host-facing ports (INT source)
    is_int_sink: bool = False    # Has host-facing ports (INT sink)
    collector_port: Optional[int] = None  # Port for INT report mirroring


@dataclass
class HostInfo:
    """Host metadata container."""
    name: str
    ip: str
    mac: str
    connected_switch: str
    port_on_switch: int


@dataclass
class LinkInfo:
    """Link metadata container."""
    node1: str
    node2: str
    port1: Optional[int] = None
    port2: Optional[int] = None
    bw: int = 10  # Bandwidth in Mbps


@dataclass
class CollectorInfo:
    """INT collector host metadata."""
    name: str
    connected_switches: List[str] = field(default_factory=list)
    interfaces: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # interfaces: {switch_name: {'collector_ip': ..., 'switch_ip': ..., 'mac': ...}}


class TopologyBuilder(ABC):
    """
    Abstract base class for topology builders.

    Subclasses implement specific topology types (Fat-Tree, Leaf-Spine, etc.)
    and provide the build() method to construct the topology.
    """

    def __init__(self, config: TopologyConfig):
        """
        Initialize the topology builder.

        Args:
            config: Validated topology configuration
        """
        self.config = config
        self.switches: Dict[str, SwitchInfo] = {}
        self.hosts: Dict[str, HostInfo] = {}
        self.links: List[LinkInfo] = []
        self.collectors: Dict[str, CollectorInfo] = {}

        # Counters for ID/port assignment
        self._thrift_port_counter = config.switch_defaults.thrift_port_base
        self._host_id_counter = 1
        self._host_ip_counter = 1

        # Switch ID assignment bases by role
        self._switch_id_bases = {
            'leaf': 1,
            'access': 1,
            'spine': 11,
            'distribution': 11,
            'core': 21,
        }
        self._switch_id_counters = dict(self._switch_id_bases)

    @abstractmethod
    def build(self) -> None:
        """
        Build the topology data structures.

        Must populate:
        - self.switches
        - self.hosts
        - self.links
        - self.collectors
        """
        pass

    @abstractmethod
    def get_layout_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Get node positions for visualization.

        Returns:
            Dict mapping node names to (x, y) coordinates
        """
        pass

    def _next_thrift_port(self) -> int:
        """Get the next available thrift port."""
        port = self._thrift_port_counter
        self._thrift_port_counter += 1
        return port

    def _next_switch_id(self, role: str) -> int:
        """Get the next switch ID for the given role."""
        base_role = self._normalize_role(role)
        sid = self._switch_id_counters.get(base_role, 1)
        self._switch_id_counters[base_role] = sid + 1
        return sid

    def _normalize_role(self, role: str) -> str:
        """Normalize role name to base category."""
        if role in ('leaf', 'tor', 'access'):
            return 'leaf'
        elif role in ('spine', 'agg', 'distribution'):
            return 'spine'
        elif role == 'core':
            return 'core'
        return role

    def _generate_host_ip(self, host_id: int, switch_id: int) -> str:
        """
        Generate IP address for a host.

        Uses scheme: 10.{switch_id}.{host_id}.2/24
        """
        return f"10.{switch_id}.{host_id}.2"

    def _generate_host_mac(self, host_id: int) -> str:
        """
        Generate MAC address for a host.

        Uses scheme: 00:00:00:00:{hi}:{lo}
        """
        hi = (host_id >> 8) & 0xFF
        lo = host_id & 0xFF
        return f"00:00:00:00:{hi:02x}:{lo:02x}"

    def _add_switch(self, name: str, role: str, pod_id: int = None) -> SwitchInfo:
        """
        Add a switch to the topology.

        Args:
            name: Switch name (e.g., "leaf1", "spine2")
            role: Switch role ("leaf", "spine", "core", etc.)
            pod_id: Optional pod assignment

        Returns:
            SwitchInfo for the new switch
        """
        switch_id = self._next_switch_id(role)
        thrift_port = self._next_thrift_port()

        # Leaf/access switches are INT sources and sinks
        is_edge = role in ('leaf', 'access', 'tor')

        sw = SwitchInfo(
            name=name,
            role=role,
            switch_id=switch_id,
            thrift_port=thrift_port,
            p4_source=self.config.switch_defaults.p4_source,
            pod_id=pod_id,
            is_int_source=is_edge,
            is_int_sink=is_edge,
            collector_port=self.config.int_collectors.port if is_edge else None,
        )
        self.switches[name] = sw
        return sw

    def _add_host(self, name: str, connected_switch: str, port_on_switch: int) -> HostInfo:
        """
        Add a host to the topology.

        Args:
            name: Host name (e.g., "h1")
            connected_switch: Name of switch this host connects to
            port_on_switch: Port number on the switch

        Returns:
            HostInfo for the new host
        """
        host_id = self._host_id_counter
        self._host_id_counter += 1

        # Get switch ID for IP generation
        switch_info = self.switches.get(connected_switch)
        switch_id = switch_info.switch_id if switch_info else host_id

        ip = self._generate_host_ip(host_id, switch_id)
        mac = self._generate_host_mac(host_id)

        host = HostInfo(
            name=name,
            ip=ip,
            mac=mac,
            connected_switch=connected_switch,
            port_on_switch=port_on_switch,
        )
        self.hosts[name] = host

        # Update switch's connected hosts
        if connected_switch in self.switches:
            self.switches[connected_switch].connected_hosts.append(name)

        return host

    def _add_link(self, node1: str, node2: str, bw: int = 10,
                  port1: int = None, port2: int = None) -> LinkInfo:
        """
        Add a link between two nodes.

        Args:
            node1: First node name
            node2: Second node name
            bw: Bandwidth in Mbps
            port1: Port number on node1
            port2: Port number on node2

        Returns:
            LinkInfo for the new link
        """
        link = LinkInfo(
            node1=node1,
            node2=node2,
            port1=port1,
            port2=port2,
            bw=bw,
        )
        self.links.append(link)

        # Update switch connections
        if node1 in self.switches and node2 in self.switches:
            self.switches[node1].connected_switches.append(node2)
            self.switches[node2].connected_switches.append(node1)

        return link

    def _setup_collectors(self) -> None:
        """
        Set up INT collector hosts based on configuration.

        Creates collector hosts and assigns them to leaf switches.
        """
        # Get all leaf/access switches
        edge_switches = sorted([
            sw for sw in self.switches.values()
            if sw.role in ('leaf', 'access', 'tor')
        ], key=lambda s: s.name)

        if not edge_switches:
            return

        ratio = self.config.int_collectors.ratio
        ip_base = self.config.int_collectors.ip_base
        collector_port = self.config.int_collectors.port
        collector_id_base = self.config.switch_naming.collector_id_base

        ip_counter = 10  # Start at .10.x subnet

        for i in range(0, len(edge_switches), ratio):
            group = edge_switches[i:i + ratio]
            collector_id = collector_id_base + (i // ratio)
            collector_name = f"{self.config.switch_naming.collector_prefix}{collector_id}"

            collector = CollectorInfo(
                name=collector_name,
                connected_switches=[sw.name for sw in group],
                interfaces={},
            )

            for sw in group:
                collector_ip = f"{ip_base}.{ip_counter}.101"
                switch_ip = f"{ip_base}.{ip_counter}.100"
                collector_mac = f"10:10:10:10:{ip_counter:02x}:11"
                switch_mac = f"10:10:10:10:{ip_counter:02x}:10"

                collector.interfaces[sw.name] = {
                    'collector_ip': collector_ip,
                    'switch_ip': switch_ip,
                    'collector_mac': collector_mac,
                    'switch_mac': switch_mac,
                    'subnet': f"{ip_base}.{ip_counter}.0/24",
                    'port': collector_port,
                }
                ip_counter += 1

            self.collectors[collector_name] = collector

    # -------------------------------------------------------------------------
    # Query methods for controller/RL agent
    # -------------------------------------------------------------------------

    def get_switch_role_map(self) -> Dict[int, str]:
        """Return switch_id -> role mapping for controller."""
        return {sw.switch_id: sw.role for sw in self.switches.values()}

    def get_switch_name_map(self) -> Dict[int, str]:
        """Return switch_id -> name mapping."""
        return {sw.switch_id: sw.name for sw in self.switches.values()}

    def get_switch_id_map(self) -> Dict[str, int]:
        """Return name -> switch_id mapping."""
        return {sw.name: sw.switch_id for sw in self.switches.values()}

    def get_leaf_switches(self) -> List[str]:
        """Return list of leaf/edge switch names."""
        return [sw.name for sw in self.switches.values()
                if sw.role in ('leaf', 'access', 'tor')]

    def get_spine_switches(self) -> List[str]:
        """Return list of spine/aggregation switch names."""
        return [sw.name for sw in self.switches.values()
                if sw.role in ('spine', 'distribution', 'agg')]

    def get_core_switches(self) -> List[str]:
        """Return list of core switch names."""
        return [sw.name for sw in self.switches.values()
                if sw.role == 'core']

    def get_host_ips(self) -> Dict[str, str]:
        """Return host_name -> IP mapping."""
        return {h.name: h.ip for h in self.hosts.values()}

    def get_host_macs(self) -> Dict[str, str]:
        """Return host_name -> MAC mapping."""
        return {h.name: h.mac for h in self.hosts.values()}

    def get_collector_interfaces(self) -> List[str]:
        """Return list of interfaces to sniff for INT reports."""
        interfaces = []
        for collector in self.collectors.values():
            for sw_name, info in collector.interfaces.items():
                port = info['port']
                interfaces.append(f"{sw_name}-eth{port}")
        return interfaces

    def get_traffic_hosts(self) -> List[str]:
        """Return list of traffic-generating host names (excluding collectors)."""
        collector_names = set(self.collectors.keys())
        return [h.name for h in self.hosts.values()
                if h.name not in collector_names]

    def get_total_switches(self) -> int:
        """Return total number of switches."""
        return len(self.switches)

    def get_total_hosts(self) -> int:
        """Return total number of hosts (excluding collectors)."""
        collector_names = set(self.collectors.keys())
        return len([h for h in self.hosts.values() if h.name not in collector_names])
