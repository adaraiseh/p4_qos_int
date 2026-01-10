import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from p4utils.mininetlib.network_API import NetworkAPI

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from logging_config import setup_unified_logging
from topology.factory import create_topology
from topology.base import TopologyBuilder
from config.validator import validate_config
from config.schema import TopologyConfig

log = logging.getLogger(__name__)


# ----------------------------
# Helpers for traffic creation
# ----------------------------

# Fixed mapping: qid -> ToS(DSCP)
QID_TOS = {
    0: 184,   # voice (DSCP EF = 46 << 2)
    1: 96,    # video (DSCP AF41 = 24 << 2)
    7: 0,     # best effort
}

ALL_QUEUES = [0, 1, 7]


def _ensure_dict_per_queue(val, queues):
    """Allow single scalar or per-queue dict; return dict{qid: val}."""
    if isinstance(val, dict):
        return {qid: val.get(qid, list(val.values())[0]) for qid in queues}
    return {qid: val for qid in queues}


def _traffic_dst_port(flow_id: int, qid: int, base: int = 6000) -> int:
    """
    Build dst port so that:
      flow_id = (port // 10) % 100
      expected_queue_id = port % 10
    Using scheme: base + flow_id*10 + qid
    """
    if not (0 <= flow_id <= 99):
        raise ValueError("flow_id must be 0..99")
    if not (0 <= qid <= 9):
        raise ValueError("queue_id must be 0..9")
    return base + flow_id * 10 + qid


def _probe_dst_port(flow_id: int, qid: int, base: int = 5000) -> int:
    """
    Build dst port for probe packets.
    """
    if not (0 <= flow_id <= 99):
        raise ValueError("flow_id must be 0..99")
    if not (0 <= qid <= 9):
        raise ValueError("queue_id must be 0..9")
    return base + flow_id * 10 + qid


def generate_traffic(
    net: NetworkAPI,
    src_host: str,
    dst_host: str,
    flow_id: int,
    queue_id: int | str = "all",
    per_queue_bw: float | int | dict = 2.0,
    per_queue_len: int | dict = 0,
    hosts_ips: dict = None,
):
    """
    Schedule receiver(s) on dst and sender(s) on src according to:
      - src_host, dst_host: e.g., 'h1', 'h8'
      - flow_id: 0..99 (encoded into dst ports)
      - queue_id: int (0..9) or 'all' (== [0,1,7] per your mapping)
      - per_queue_bw: Mbps as float/int or dict{qid: Mbps}
      - per_queue_len: iperf3 -t duration seconds as int or dict{qid: seconds} (0 = continuous)
      - hosts_ips: dict mapping host name to IP address
    """
    if hosts_ips is None:
        raise ValueError("hosts_ips must be provided")

    # exactly the three queues you requested when "all"
    queues = ALL_QUEUES if queue_id == "all" else [int(queue_id)]

    # Per-queue params normalized to dicts
    bw_map = _ensure_dict_per_queue(per_queue_bw, queues)
    len_map = _ensure_dict_per_queue(per_queue_len, queues)

    dst_ip = hosts_ips.get(dst_host)
    if not dst_ip:
        raise ValueError(f"No IP found for host {dst_host}")

    # Start iperf3 servers for each selected queue/port
    for qid in queues:
        traffic_port = _traffic_dst_port(flow_id, qid)
        net.addTask(
            dst_host,
            (
                f"bash -lc '"
                f"while true; do "
                f"  iperf3 -s -p {traffic_port} -i 1 "
                f"    --logfile /tmp/{dst_host}_iperf3_s_{traffic_port}.log ; "
                f"  echo \"[RESTART][$(date +%F_%T)] iperf3 server {traffic_port} exited ($?)\" "
                f"    >> /tmp/{dst_host}_iperf3_s_{traffic_port}.log ; "
                f"  sleep 1 ; "
                f"done'"
            ),
            1, 0, True
        )

    # Send traffic from src -> dst for each selected queue
    for qid in queues:
        tos = QID_TOS.get(qid)
        if tos is None:
            raise ValueError(f"No ToS mapping defined for queue {qid}")

        traffic_port = _traffic_dst_port(flow_id, qid)
        bw_mbps = bw_map[qid]
        length = len_map[qid]

        net.addTask(
            src_host,
            (
                f"bash -lc '"
                f"while true; do "
                f"  iperf3 -c {dst_ip} -p {traffic_port} -u "
                f"         -b {bw_mbps}M -l {length} --tos {tos} "
                f"         -i 1 -t 0 --connect-timeout 5000 "
                f"         >> /tmp/{src_host}_iperf3_c_{traffic_port}.log 2>&1 ; "
                f"  echo \"[RESTART][$(date +%F_%T)] iperf3 client {traffic_port} exited ($?)\" "
                f"    >> /tmp/{src_host}_iperf3_c_{traffic_port}.log ; "
                f"  sleep 1 ; "
                f"done'"
            ),
            2.0, 0, True
        )


class NetworkBuilder:
    """
    Builds a Mininet network from a TopologyBuilder.

    Translates topology data into p4utils NetworkAPI calls.
    """

    def __init__(self, builder: TopologyBuilder, rules_dir: str = None):
        """
        Initialize the network builder.

        Args:
            builder: Built TopologyBuilder instance
            rules_dir: Directory containing P4 rule files (default: rules/<topology_name>)
        """
        self.builder = builder
        self.config = builder.config

        if rules_dir is None:
            topo_name = self.config.topology.name.replace('-', '_')
            rules_dir = str(Path(__file__).parent / 'rules' / topo_name)
        self.rules_dir = rules_dir

        self.net: Optional[NetworkAPI] = None
        self._switch_refs = {}  # name -> p4utils switch reference
        self._host_refs = {}    # name -> p4utils host reference

    def build(self) -> NetworkAPI:
        """
        Build the Mininet network.

        Returns:
            Configured NetworkAPI instance
        """
        self.net = NetworkAPI()

        # Network general options
        self.net.setLogLevel('info')
        self.net.disableCli()

        # 1. Create switches
        self._create_switches()

        # 2. Set P4 source for all switches
        self.net.setP4SourceAll(self.config.switch_defaults.p4_source)

        # 3. Create hosts
        self._create_hosts()

        # 4. Create links
        self._create_links()

        # 5. Set L3 addressing
        self.net.l3()

        # 6. Set up INT collectors
        self._setup_collectors()

        # 7. Enable schedulers on traffic hosts
        self._enable_host_schedulers()

        return self.net

    def _create_switches(self) -> None:
        """Create all P4 switches from topology."""
        for switch_name, sw_info in self.builder.switches.items():
            # Determine max bandwidth for this switch
            max_bw = self._get_switch_max_bw(sw_info.role)

            cli_input = os.path.join(self.rules_dir, f"{switch_name}-commands.txt")

            switch_ref = self.net.addP4Switch(
                switch_name,
                priority_queues_num=self.config.switch_defaults.priority_queues,
                max_link_bw=max_bw,
                thrift_port=sw_info.thrift_port,
                cli_input=cli_input
            )
            self._switch_refs[switch_name] = switch_ref

    def _get_switch_max_bw(self, role: str) -> int:
        """Get max bandwidth for a switch based on its role."""
        bw = self.config.link_bandwidths

        if role in ('leaf', 'access', 'tor'):
            return max(bw.host_leaf, bw.host_access, bw.leaf_spine, bw.access_distribution)
        elif role in ('spine', 'distribution', 'agg'):
            return max(bw.leaf_spine, bw.access_distribution, bw.spine_core, bw.distribution_core)
        elif role == 'core':
            return max(bw.spine_core, bw.distribution_core)
        return 10  # default

    def _create_hosts(self) -> None:
        """Create all hosts from topology."""
        for host_name, host_info in self.builder.hosts.items():
            host_ref = self.net.addHost(host_name)
            self._host_refs[host_name] = host_ref

    def _create_links(self) -> None:
        """Create all links from topology."""
        for link in self.builder.links:
            node1 = link.node1
            node2 = link.node2

            # Get node references
            ref1 = self._host_refs.get(node1) or self._switch_refs.get(node1)
            ref2 = self._host_refs.get(node2) or self._switch_refs.get(node2)

            if ref1 is None or ref2 is None:
                log.warning(f"Skipping link {node1} <-> {node2}: node not found")
                continue

            # Build kwargs for addLink
            kwargs = {'bw': link.bw}
            if link.port1 is not None:
                kwargs['port1'] = link.port1
            if link.port2 is not None:
                kwargs['port2'] = link.port2

            self.net.addLink(ref1, ref2, **kwargs)

    def _setup_collectors(self) -> None:
        """Set up INT collector hosts and interfaces."""
        for collector_name, collector_info in self.builder.collectors.items():
            # Create collector host
            collector_ref = self.net.addHost(collector_name)
            self._host_refs[collector_name] = collector_ref

            # Track port numbers for multi-interface collectors
            collector_port_num = 10

            for sw_name in collector_info.connected_switches:
                sw_ref = self._switch_refs.get(sw_name)
                if sw_ref is None:
                    continue

                iface = collector_info.interfaces.get(sw_name, {})
                collector_ip = iface.get('collector_ip', '0.0.0.0')
                switch_ip = iface.get('switch_ip', '0.0.0.0')
                collector_mac = iface.get('collector_mac', '00:00:00:00:00:00')
                switch_mac = iface.get('switch_mac', '00:00:00:00:00:00')
                port = iface.get('port', 10)

                # Add link: collector port N <-> switch port 10 (collector port)
                self.net.addLink(
                    collector_ref, sw_ref,
                    port1=collector_port_num, port2=port
                )

                # Set IP and MAC addresses
                self.net.setIntfIp(collector_ref, sw_ref, f"{collector_ip}/24")
                self.net.setIntfIp(sw_ref, collector_ref, f"{switch_ip}/24")
                self.net.setIntfMac(collector_ref, sw_ref, collector_mac)
                self.net.setIntfMac(sw_ref, collector_ref, switch_mac)

                collector_port_num += 1

    def _enable_host_schedulers(self) -> None:
        """Enable task schedulers on traffic-generating hosts."""
        traffic_hosts = self.builder.get_traffic_hosts()
        for host_name in traffic_hosts:
            host_ref = self._host_refs.get(host_name)
            if host_ref:
                self.net.enableScheduler(host_ref)

    def get_hosts_ips(self) -> dict:
        """Get host name to IP mapping for traffic generation."""
        return self.builder.get_host_ips()

    def cleanup(self) -> None:
        """
        Clean up network resources.

        Stops the Mininet network and releases associated resources.
        Critical for multi-topology training to prevent resource leaks.
        """
        if self.net is not None:
            try:
                self.net.stopNetwork()
            except Exception as e:
                log.warning(f"Error stopping network: {e}")

    def __enter__(self):
        """Context manager support for automatic cleanup."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.cleanup()
        return False


def config_network(config_path: str, rules_dir: str = None) -> tuple[NetworkAPI, TopologyBuilder]:
    """
    Configure the network from a YAML configuration file.

    Args:
        config_path: Path to YAML topology configuration
        rules_dir: Optional override for rules directory

    Returns:
        Tuple of (NetworkAPI instance, TopologyBuilder instance)
    """
    # Validate configuration
    log.info(f"Validating configuration: {config_path}")
    result = validate_config(config_path)

    if result.warnings:
        for warning in result.warnings:
            log.warning(warning)

    if not result.is_valid:
        log.error("Configuration validation failed:")
        for error in result.errors:
            log.error(f"  - {error}")
        raise ValueError("Invalid configuration")

    log.info(f"Configuration valid: {result.config.topology.name}")

    # Build topology
    builder = create_topology(config_path, validate=False)

    # Build network
    net_builder = NetworkBuilder(builder, rules_dir)
    net = net_builder.build()

    return net, builder


def get_args():
    parser = argparse.ArgumentParser(
        description="Start P4 network with configurable topology"
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to YAML topology configuration file',
        type=str,
        required=False,
        default='config/topologies/fat_tree_k4.yaml'
    )
    parser.add_argument(
        '--rules', '-r',
        help='Directory containing P4 rule files (default: auto from config)',
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        '--p4',
        help='Override P4 source file (default: from config)',
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='debug',
        choices=['debug', 'info', 'warning', 'error'],
        help='File log level (console always shows INFO)'
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Set up unified logging
    setup_unified_logging(module_name="network", log_level=args.log_level)

    # Build network from configuration
    net, builder = config_network(args.config, args.rules)

    # Override P4 source if specified
    if args.p4:
        net.setP4SourceAll(args.p4)

    # Start network
    net.startNetwork()

    # Determine rules directory
    if args.rules:
        rules_dir = args.rules
    else:
        topo_name = builder.config.topology.name.replace('-', '_')
        rules_dir = str(Path(__file__).parent / 'rules' / topo_name)

    # Start the P4 controller with topology info
    from controller import Controller
    controller = Controller(topology_builder=builder, rules_dir=rules_dir)

    log.info("SUMMARY:")
    log.info(f"Topology: {builder.config.topology.name} ({builder.config.topology.type.value})")
    log.info(f"Switches: {len(builder.switches)}")
    log.info(f"Hosts: {len(builder.hosts)}")
    log.info(f"INT Collectors: {len(builder.collectors)}")
    log.info("OSPF Shortest Paths:")
    controller.print_paths()

    # Auto-launch visualization
    import subprocess

    viz_script = os.path.join(os.getcwd(), "visualize_routes.py")
    if os.path.exists(viz_script):
        log.info(f"Auto-launching visualization: {viz_script}")

        cmd = ["python3", viz_script, "--config", args.config]

        # If running as root (sudo), try to launch as the original user
        sudo_user = os.environ.get('SUDO_USER')
        if sudo_user:
            cmd = ["sudo", "-u", sudo_user] + cmd

        subprocess.Popen(cmd, start_new_session=True)

    net.enableCli()
    net.start_net_cli()


if __name__ == '__main__':
    main()
