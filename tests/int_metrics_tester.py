#!/usr/bin/env python3
"""
int_metrics_tester.py - Verify INT metrics collection for all traffic flows.

Tests that all expected INT metrics (switch_latency, tx_utilization,
q_drop_rate_100ms, flow_latency) are present in InfluxDB for the
configured traffic pairs and topology. Also validates that switch IDs
in metrics correspond to the correct switches on the expected flow paths.

Usage:
    sudo python3 int_metrics_tester.py --config config/topologies/fat_tree_k4.yaml

Requirements:
    - Mininet network must be running (sudo python network.py -c <config>)
    - INT collector must be running (sudo python report_collector/main.py)
    - InfluxDB must be accessible
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set

from influxdb_client import InfluxDBClient
from p4utils.utils.helper import load_topo
from p4utils.utils.task_scheduler import Task, TaskClient

# Project imports
from network import _traffic_dst_port, QID_TOS, ALL_QUEUES
from controller import Controller
from topology.factory import create_topology
from config.loader import load_config
from traffic_generator import TrafficManager

log = logging.getLogger(__name__)

# =============================================================================
#                              TIMING CONSTANTS
# =============================================================================
# From rl_agent_4.py (lines 144-154) - use same timers as training agent
WINDOW_SECONDS = 2.0        # Observation window for metric collection (2s captures ~6 samples with 300ms per-flow sampling)
SAFETY_LAG_MS = 100         # Safety lag for InfluxDB query
METRICS_WAIT_TIME = 5.0     # Wait time for metrics to populate (like episode reset)
MIN_POINTS_PER_METRIC = 1   # Require at least 1 data point per metric

# Measurements and queues
MEASUREMENTS = ['switch_latency', 'tx_utilization', 'q_drop_rate_100ms', 'flow_latency']
PER_SWITCH_MEASUREMENTS = ['switch_latency', 'tx_utilization', 'q_drop_rate_100ms']
QUEUES = [0, 1, 7]

# Traffic tag for process identification
TRAFFIC_TAG = "__INT_TESTER__"


# =============================================================================
#                              REPORT DATACLASSES
# =============================================================================
@dataclass
class MissingMetric:
    """Represents a missing expected metric."""
    measurement: str
    src_ip: str
    dst_ip: str
    queue_id: int
    switch_id: Optional[int]  # For per-switch metrics
    flow_id: int
    switch_name: Optional[str] = None


@dataclass
class PathDeviation:
    """Represents a switch that appears in metrics but is NOT on expected path."""
    src_ip: str
    dst_ip: str
    src_host: str
    dst_host: str
    flow_id: int
    queue_id: int
    unexpected_switch_id: int      # Switch in metrics but not on path
    expected_path: List[int]       # Expected switch IDs for this flow
    switch_name: str               # Resolved switch name (e.g., "a2")


@dataclass
class FlowPathValidation:
    """Validation result for a single flow's path."""
    src_host: str
    dst_host: str
    src_ip: str
    dst_ip: str
    flow_id: int
    expected_switches: List[int]      # Switch IDs on expected path
    expected_switch_names: List[str]  # Switch names (e.g., ["t1", "a1", "c1", "a3", "t6"])
    found_switches: Set[int]          # Switch IDs found in metrics
    missing_switches: List[int]       # Expected but not found
    unexpected_switches: List[int]    # Found but not expected (PATH DEVIATION!)
    is_valid: bool                    # True if found == expected


@dataclass
class MetricsReport:
    """Full verification report."""
    total_expected: int
    total_found: int
    missing_metrics: List[MissingMetric]
    by_queue: Dict[int, Dict]         # {0: {'expected': N, 'found': M}, ...}
    by_measurement: Dict[str, Dict]   # {'switch_latency': {'expected': N, 'found': M}, ...}
    by_switch: Dict[int, Dict]        # {switch_id: {'expected': N, 'found': M}, ...}
    # Path validation
    path_validations: List[FlowPathValidation]  # Per-flow path validation
    path_deviations: List[PathDeviation]        # Unexpected switches
    total_flows: int
    valid_paths: int
    invalid_paths: int


# =============================================================================
#                              MAIN TESTER CLASS
# =============================================================================
class INTMetricsTester:
    """Tests INT metrics collection for all traffic flows in a topology."""

    def __init__(self, config_path: str,
                 influx_url: str = "http://192.168.56.1:8086",
                 influx_token: str = None,
                 influx_org: str = "Research",
                 influx_bucket: str = "INT",
                 topology_file: str = "/tmp/topology.json",
                 window_seconds: float = WINDOW_SECONDS):
        """
        Initialize the INT metrics tester.

        Args:
            config_path: Path to topology YAML config
            influx_url: InfluxDB URL
            influx_token: InfluxDB token
            influx_org: InfluxDB organization
            influx_bucket: InfluxDB bucket
            topology_file: Path to topology.json
            window_seconds: Query time window in seconds
        """
        self.config_path = config_path
        self.topology_file = topology_file
        self.bucket = influx_bucket
        self.org = influx_org
        self.window_seconds = window_seconds

        # Resolve InfluxDB token from argument or environment variable
        if influx_token is None:
            influx_token = os.environ.get('INFLUX_TOKEN')
        if not influx_token:
            raise ValueError("InfluxDB token not configured. Set INFLUX_TOKEN environment variable or pass influx_token argument.")

        # Load topology config
        log.info(f"Loading topology config from {config_path}")
        self._topology_config = load_config(config_path)
        self._topology_builder = create_topology(config_path)

        # Initialize InfluxDB client
        log.info(f"Connecting to InfluxDB at {influx_url}")
        self.client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org, timeout=10000)
        self.query_api = self.client.query_api()

        # Initialize Controller (for path info and switch mappings)
        log.info("Initializing controller...")
        self.controller = Controller(
            verbose=False,
            topology_builder=self._topology_builder,
            rules_dir=None
        )

        # Get host IPs
        self.hosts_ips = self._get_hosts_ips()
        log.info(f"Found {len(self.hosts_ips)} traffic hosts")

        # Build traffic pairs from config (matching traffic_generator)
        self.traffic_pairs = self._build_traffic_pairs_from_config()
        log.info(f"Built {len(self.traffic_pairs)} traffic pairs from config")

        # Store expected paths for validation
        self.expected_paths: Dict[Tuple[str, str], List[int]] = {}
        self.expected_path_names: Dict[Tuple[str, str], List[str]] = {}

        # Current traffic loads
        self.current_load: Dict[int, float] = {}

    def _get_hosts_ips(self) -> Dict[str, str]:
        """Get host name to IP mapping from topology.json."""
        hosts_ips = {}
        try:
            with open(self.topology_file, 'r') as f:
                topo = json.load(f)

            for node in topo.get('nodes', []):
                node_id = node.get('id', '')
                if isinstance(node_id, str) and node_id.startswith('h'):
                    try:
                        host_num = int(node_id[1:])
                        if host_num < 100:  # Exclude collectors (h100+)
                            ip = node.get('ip', '')
                            if ip:
                                ip = ip.split('/')[0]
                                hosts_ips[node_id] = ip
                    except ValueError:
                        continue

            if hosts_ips:
                return hosts_ips

        except Exception as e:
            log.warning(f"Error loading topology.json: {e}")

        # Fallback to topology builder
        hosts_ips = self._topology_builder.get_host_ips()
        return {name: ip for name, ip in hosts_ips.items()
                if name.startswith('h') and int(name[1:]) < 100}

    def _build_traffic_pairs_from_config(self) -> List[Tuple[str, str, int]]:
        """Load traffic pairs from config file (same as traffic_generator)."""
        pairs = []
        flow_id = 10  # FLOW_ID_BASE

        # Access Pydantic model attributes
        if not self._topology_config or not self._topology_config.traffic.pairs:
            log.warning("No traffic pairs in config, falling back to algorithmic generation")
            return self._build_all_hosts_traffic_pairs()

        for pair in self._topology_config.traffic.pairs:
            src_name = pair.src
            dst_name = pair.dst
            if src_name in self.hosts_ips and dst_name in self.hosts_ips:
                pairs.append((src_name, dst_name, flow_id))
                flow_id += 1
            else:
                log.warning(f"Skipping pair {src_name}->{dst_name}: host not found")

        if not pairs:
            log.warning("No valid pairs found in config, falling back to algorithmic generation")
            return self._build_all_hosts_traffic_pairs()

        return pairs

    def _build_all_hosts_traffic_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Build traffic pairs so EVERY host sends traffic.
        Each host sends to a destination in a different pod for cross-pod traffic.
        """
        pairs = []
        hosts = sorted([h for h in self.hosts_ips.keys()], key=lambda x: int(x[1:]))
        n_hosts = len(hosts)

        # Build pod structure (groups of hosts sharing the same ToR)
        # For fat-tree k=4: 16 hosts, 4 pods, 4 hosts per pod
        pods = []
        hosts_per_pod = 4 if n_hosts >= 16 else 2
        for i in range(0, n_hosts, hosts_per_pod):
            pods.append(hosts[i:i + hosts_per_pod])

        n_pods = len(pods)
        flow_id = 10  # Start flow_id at 10 like TrafficManager

        for pod_idx, pod in enumerate(pods):
            for host in pod:
                # Find destination in next pod (round-robin)
                dst_pod_idx = (pod_idx + 1) % n_pods
                dst_pod = pods[dst_pod_idx]
                # Pick a host from destination pod
                dst_host = dst_pod[0] if dst_pod else hosts[(hosts.index(host) + n_hosts // 2) % n_hosts]

                if host != dst_host:
                    pairs.append((host, dst_host, flow_id))
                    flow_id += 1

        return pairs

    def _send_task(self, hostname: str, cmd: str, delay: float = 0.0) -> bool:
        """Send a task to the host's TaskServer."""
        socket_path = f"/tmp/{hostname}_socket"
        if not os.path.exists(socket_path):
            log.warning(f"TaskServer socket not found: {socket_path}")
            return False
        try:
            client = TaskClient(socket_path)
            task = Task(cmd, start=time.time() + delay, duration=0)
            client.send([task], retry=False)
            return True
        except PermissionError:
            log.error(f"Permission denied for {hostname}. Run with sudo!")
            return False
        except Exception as e:
            log.warning(f"Failed to send task to {hostname}: {e}")
            return False

    def stop_traffic(self):
        """Stop all traffic processes."""
        import subprocess
        log.info("Stopping all traffic processes...")
        try:
            subprocess.run(['pkill', '-9', '-f', 'iperf3.*-p 6[12]'], capture_output=True)
            subprocess.run(['pkill', '-9', '-f', f'bash.*{TRAFFIC_TAG}'], capture_output=True)
        except Exception as e:
            log.warning(f"Failed to kill traffic: {e}")
        time.sleep(0.5)

    def start_traffic(self, profile: str = 'light_1', packet_len: int = 1250) -> Dict:
        """
        Start traffic from all hosts using the specified profile.

        Args:
            profile: Traffic profile name (default: light_1)
            packet_len: UDP packet length in bytes

        Returns:
            Dict with profile info
        """
        import random
        self.stop_traffic()
        time.sleep(0.3)

        # Get profile
        if profile not in TrafficManager.TRAFFIC_PROFILES:
            raise ValueError(
                f"Unknown traffic profile {profile!r}; valid profiles: "
                f"{', '.join(TrafficManager.TRAFFIC_PROFILES)}"
            )

        profile_ranges = TrafficManager.TRAFFIC_PROFILES[profile]
        self.current_load = {
            qid: random.uniform(*rng) for qid, rng in profile_ranges.items()
        }

        log.info(f"Starting traffic with profile '{profile}'")
        log.info(f"  Loads: Q0={self.current_load[0]:.2f}, Q1={self.current_load[1]:.2f}, Q7={self.current_load[7]:.2f} Mbps")

        # Start servers on receivers
        receivers = set(p[1] for p in self.traffic_pairs)
        for receiver in receivers:
            for _, dst, flow_id in [p for p in self.traffic_pairs if p[1] == receiver]:
                for qid in QUEUES:
                    port = _traffic_dst_port(flow_id, qid)
                    cmd = (
                        f"bash -lc '{TRAFFIC_TAG}=1; "
                        f"while true; do iperf3 -s -p {port} -i 1 "
                        f"--logfile /tmp/{receiver}_iperf3_s_{port}.log; sleep 1; done'"
                    )
                    self._send_task(receiver, cmd)

        time.sleep(1.0)

        # Start clients on senders
        for sender, receiver, flow_id in self.traffic_pairs:
            dst_ip = self.hosts_ips.get(receiver, "0.0.0.0")
            for qid in QUEUES:
                port = _traffic_dst_port(flow_id, qid)
                tos = QID_TOS.get(qid, 0)
                bw = self.current_load.get(qid, 0.2)
                cmd = (
                    f"bash -lc '{TRAFFIC_TAG}=1; "
                    f"while true; do iperf3 -c {dst_ip} -p {port} -u "
                    f"-b {bw}M -l {packet_len} --tos {tos} -i 1 -t 0 "
                    f"--connect-timeout 5000 >> /tmp/{sender}_iperf3_c_{port}.log 2>&1; "
                    f"sleep 1; done'"
                )
                self._send_task(sender, cmd, delay=0.5)

        log.info("Traffic generation started")
        return {
            'profile_name': profile,
            'loads': self.current_load.copy(),
            'num_pairs': len(self.traffic_pairs),
        }

    def wait_for_metrics(self, wait_seconds: float = METRICS_WAIT_TIME):
        """Wait for INT metrics to populate in InfluxDB."""
        log.info(f"Waiting {wait_seconds}s for metrics to populate...")
        time.sleep(wait_seconds)

    def _time_window(self) -> Tuple[str, str]:
        """Calculate time window for InfluxDB query (like rl_agent_4.py)."""
        stop_dt = datetime.utcnow() - timedelta(milliseconds=SAFETY_LAG_MS)
        start_dt = stop_dt - timedelta(seconds=self.window_seconds)
        start = start_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        stop = stop_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        return start, stop

    def compute_expected_metrics(self) -> Dict[str, List[Dict]]:
        """
        Compute all expected metrics based on traffic pairs and paths.

        Returns:
            Dict mapping measurement name to list of expected tag combinations
        """
        expected = {m: [] for m in MEASUREMENTS}
        self.expected_paths.clear()
        self.expected_path_names.clear()

        for src_host, dst_host, flow_id in self.traffic_pairs:
            src_ip = self.hosts_ips.get(src_host, "")
            dst_ip = self.hosts_ips.get(dst_host, "")

            if not src_ip or not dst_ip:
                log.warning(f"Missing IP for {src_host} or {dst_host}")
                continue

            # Get path from controller
            path = self.controller.get_path_by_hosts(src_host, dst_host)
            if not path:
                log.warning(f"No path found for {src_host} -> {dst_host}")
                continue

            # Extract switches from path (exclude hosts)
            switches_on_path = []
            switch_ids_on_path = []
            for node in path:
                if node in self.controller.switch_name_to_id:
                    switches_on_path.append(node)
                    switch_ids_on_path.append(self.controller.switch_name_to_id[node])

            # Store for path validation
            self.expected_paths[(src_ip, dst_ip)] = switch_ids_on_path
            self.expected_path_names[(src_ip, dst_ip)] = switches_on_path

            for qid in QUEUES:
                # flow_latency (end-to-end, no switch_id)
                expected['flow_latency'].append({
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'queue_id': str(qid),
                    'flow_id': str(flow_id),
                })

                # Per-switch metrics
                for i, sw_name in enumerate(switches_on_path):
                    switch_id = self.controller.switch_name_to_id[sw_name]

                    base_tags = {
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'queue_id': str(qid),
                        'flow_id': str(flow_id),
                        'switch_id': str(switch_id),
                    }

                    expected['switch_latency'].append(base_tags.copy())
                    expected['tx_utilization'].append(base_tags.copy())
                    expected['q_drop_rate_100ms'].append(base_tags.copy())

        return expected

    def query_available_metrics(self) -> Dict[str, Set[tuple]]:
        """
        Query InfluxDB for all available INT metrics.

        Returns:
            Dict mapping measurement name to set of tag tuples
        """
        start, stop = self._time_window()
        available = {m: set() for m in MEASUREMENTS}

        for measurement in MEASUREMENTS:
            # Determine group columns
            if measurement == 'flow_latency':
                group_cols = ['src_ip', 'dst_ip', 'queue_id', 'flow_id']
            else:
                group_cols = ['src_ip', 'dst_ip', 'queue_id', 'flow_id', 'switch_id']

            cols_str = ', '.join(f'"{c}"' for c in group_cols)

            flux = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start}, stop: {stop})
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
                |> group(columns: [{cols_str}])
                |> count()
            '''

            try:
                tables = self.query_api.query(query=flux, org=self.org)
                for table in tables or []:
                    for record in table.records:
                        tags = tuple(sorted((col, str(record.values.get(col, ''))) for col in group_cols))
                        if all(v for _, v in tags):  # All tags present
                            available[measurement].add(tags)
            except Exception as e:
                log.error(f"Error querying {measurement}: {e}")

        return available

    def query_switches_per_flow(self) -> Dict[Tuple[str, str], Set[int]]:
        """
        Query unique switch_ids per flow for path validation.

        Returns:
            Dict mapping (src_ip, dst_ip) to set of switch IDs found in metrics
        """
        start, stop = self._time_window()
        switches_per_flow: Dict[Tuple[str, str], Set[int]] = {}

        flux = f'''
        from(bucket: "{self.bucket}")
            |> range(start: {start}, stop: {stop})
            |> filter(fn: (r) => r._measurement == "switch_latency")
            |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
            |> group(columns: ["src_ip", "dst_ip", "switch_id"])
            |> count()
        '''

        try:
            tables = self.query_api.query(query=flux, org=self.org)
            for table in tables or []:
                for record in table.records:
                    src_ip = record.values.get('src_ip', '')
                    dst_ip = record.values.get('dst_ip', '')
                    switch_id_str = record.values.get('switch_id', '')

                    if src_ip and dst_ip and switch_id_str:
                        key = (src_ip, dst_ip)
                        if key not in switches_per_flow:
                            switches_per_flow[key] = set()
                        try:
                            switches_per_flow[key].add(int(switch_id_str))
                        except ValueError:
                            pass
        except Exception as e:
            log.error(f"Error querying switches per flow: {e}")

        return switches_per_flow

    def verify_metrics(self) -> MetricsReport:
        """
        Compare expected vs available metrics and generate report.
        """
        # Compute expected metrics
        expected = self.compute_expected_metrics()
        available = self.query_available_metrics()

        # Convert expected to tuples for comparison
        expected_sets = {m: set() for m in MEASUREMENTS}
        for m, items in expected.items():
            for tags in items:
                expected_sets[m].add(tuple(sorted(tags.items())))

        # Find missing metrics
        missing_metrics = []
        by_queue = {q: {'expected': 0, 'found': 0} for q in QUEUES}
        by_measurement = {m: {'expected': 0, 'found': 0} for m in MEASUREMENTS}
        by_switch = {}

        total_expected = 0
        total_found = 0

        for measurement in MEASUREMENTS:
            exp_set = expected_sets[measurement]
            avail_set = available[measurement]

            for tags_tuple in exp_set:
                total_expected += 1
                tags_dict = dict(tags_tuple)
                qid = int(tags_dict.get('queue_id', 0))
                by_queue[qid]['expected'] += 1
                by_measurement[measurement]['expected'] += 1

                switch_id = None
                if 'switch_id' in tags_dict:
                    switch_id = int(tags_dict['switch_id'])
                    if switch_id not in by_switch:
                        by_switch[switch_id] = {'expected': 0, 'found': 0}
                    by_switch[switch_id]['expected'] += 1

                if tags_tuple in avail_set:
                    total_found += 1
                    by_queue[qid]['found'] += 1
                    by_measurement[measurement]['found'] += 1
                    if switch_id is not None:
                        by_switch[switch_id]['found'] += 1
                else:
                    # Missing metric
                    switch_name = None
                    if switch_id is not None:
                        switch_name = self.controller.switch_id_to_name.get(switch_id, f"sw{switch_id}")

                    missing_metrics.append(MissingMetric(
                        measurement=measurement,
                        src_ip=tags_dict.get('src_ip', ''),
                        dst_ip=tags_dict.get('dst_ip', ''),
                        queue_id=qid,
                        switch_id=switch_id,
                        flow_id=int(tags_dict.get('flow_id', 0)),
                        switch_name=switch_name,
                    ))

        # Path validation
        switches_per_flow = self.query_switches_per_flow()
        path_validations = []
        path_deviations = []

        for (src_ip, dst_ip), expected_switch_ids in self.expected_paths.items():
            found_switches = switches_per_flow.get((src_ip, dst_ip), set())
            expected_set = set(expected_switch_ids)

            missing_switches = list(expected_set - found_switches)
            unexpected_switches = list(found_switches - expected_set)

            # Find src/dst host names
            src_host = None
            dst_host = None
            flow_id = 0
            for s, d, fid in self.traffic_pairs:
                if self.hosts_ips.get(s) == src_ip and self.hosts_ips.get(d) == dst_ip:
                    src_host = s
                    dst_host = d
                    flow_id = fid
                    break

            path_validations.append(FlowPathValidation(
                src_host=src_host or "?",
                dst_host=dst_host or "?",
                src_ip=src_ip,
                dst_ip=dst_ip,
                flow_id=flow_id,
                expected_switches=expected_switch_ids,
                expected_switch_names=self.expected_path_names.get((src_ip, dst_ip), []),
                found_switches=found_switches,
                missing_switches=missing_switches,
                unexpected_switches=unexpected_switches,
                is_valid=(len(missing_switches) == 0 and len(unexpected_switches) == 0),
            ))

            # Record path deviations
            for unexpected_sid in unexpected_switches:
                switch_name = self.controller.switch_id_to_name.get(unexpected_sid, f"sw{unexpected_sid}")
                for qid in QUEUES:
                    path_deviations.append(PathDeviation(
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_host=src_host or "?",
                        dst_host=dst_host or "?",
                        flow_id=flow_id,
                        queue_id=qid,
                        unexpected_switch_id=unexpected_sid,
                        expected_path=expected_switch_ids,
                        switch_name=switch_name,
                    ))

        valid_paths = sum(1 for v in path_validations if v.is_valid)
        invalid_paths = len(path_validations) - valid_paths

        return MetricsReport(
            total_expected=total_expected,
            total_found=total_found,
            missing_metrics=missing_metrics,
            by_queue=by_queue,
            by_measurement=by_measurement,
            by_switch=by_switch,
            path_validations=path_validations,
            path_deviations=path_deviations,
            total_flows=len(path_validations),
            valid_paths=valid_paths,
            invalid_paths=invalid_paths,
        )

    def print_report(self, report: MetricsReport):
        """Print formatted metrics report."""
        print("\n" + "=" * 80)
        print("                        INT METRICS VERIFICATION REPORT")
        print("=" * 80)

        print("\nSUMMARY")
        print("-" * 40)
        print(f"Topology: {self._topology_config.topology.name}")
        print(f"Traffic Hosts: {len(self.hosts_ips)} (all hosts sending)")
        print(f"Traffic Pairs: {len(self.traffic_pairs)}")
        print(f"Expected Metrics: {report.total_expected}")
        print(f"Found Metrics: {report.total_found}")
        print(f"Missing Metrics: {len(report.missing_metrics)}")

        # By queue
        print("\nMISSING METRICS BY QUEUE")
        print("-" * 40)
        queue_names = {0: "voice", 1: "video", 7: "best_effort"}
        for qid in QUEUES:
            stats = report.by_queue[qid]
            missing = stats['expected'] - stats['found']
            print(f"Queue {qid} ({queue_names.get(qid, '?'):12}): {missing:3} missing "
                  f"({stats['found']}/{stats['expected']} found)")

        # By measurement
        print("\nMISSING METRICS BY MEASUREMENT")
        print("-" * 40)
        for m in MEASUREMENTS:
            stats = report.by_measurement[m]
            missing = stats['expected'] - stats['found']
            print(f"{m:22}: {missing:3} missing ({stats['found']}/{stats['expected']} found)")

        # Missing metrics detail (limit to first 20)
        if report.missing_metrics:
            print("\nMISSING METRICS DETAIL" + (" (first 20)" if len(report.missing_metrics) > 20 else ""))
            print("-" * 40)
            for metric in report.missing_metrics[:20]:
                sw_info = f", switch_id: {metric.switch_id} ({metric.switch_name})" if metric.switch_id else ""
                print(f"[MISSING] {metric.measurement}")
                print(f"  src: {metric.src_ip} -> dst: {metric.dst_ip}")
                print(f"  queue_id: {metric.queue_id}, flow_id: {metric.flow_id}{sw_info}")
                print()

        # Path validation report
        print("\n" + "=" * 80)
        print("                          PATH VALIDATION REPORT")
        print("=" * 80)

        print("\nSUMMARY")
        print("-" * 40)
        print(f"Total Flows: {report.total_flows}")
        print(f"Valid Paths: {report.valid_paths} (switch IDs match expected)")
        print(f"Invalid Paths: {report.invalid_paths} (path deviation detected!)")

        print("\nPATH VALIDATION DETAILS")
        print("-" * 40)

        for pv in report.path_validations:
            if pv.is_valid:
                path_str = " -> ".join(f"{n}({sid})" for n, sid in
                                       zip(pv.expected_switch_names, pv.expected_switches))
                print(f"[OK] {pv.src_host} -> {pv.dst_host}")
                print(f"  Expected path: {path_str}")
                print(f"  Found switches: {pv.found_switches} - MATCH")
            else:
                path_str = " -> ".join(f"{n}({sid})" for n, sid in
                                       zip(pv.expected_switch_names, pv.expected_switches))
                print(f"[DEVIATION] {pv.src_host} -> {pv.dst_host}")
                print(f"  Expected path: {path_str}")
                print(f"  Found switches: {pv.found_switches}")
                if pv.missing_switches:
                    missing_names = [self.controller.switch_id_to_name.get(sid, f"sw{sid}")
                                    for sid in pv.missing_switches]
                    print(f"  Missing switches: {pv.missing_switches} ({missing_names})")
                if pv.unexpected_switches:
                    unexpected_names = [self.controller.switch_id_to_name.get(sid, f"sw{sid}")
                                       for sid in pv.unexpected_switches]
                    print(f"  UNEXPECTED switches: {pv.unexpected_switches} ({unexpected_names}) <-- PATH DEVIATION!")
            print()

        # Final result
        print("=" * 80)
        pct = 100.0 * report.total_found / report.total_expected if report.total_expected > 0 else 0
        if report.total_found == report.total_expected and report.invalid_paths == 0:
            print(f"RESULT: SUCCESS ({report.total_found}/{report.total_expected} metrics, "
                  f"{report.valid_paths}/{report.total_flows} valid paths)")
        else:
            print(f"RESULT: PARTIAL ({report.total_found}/{report.total_expected} metrics ({pct:.1f}%), "
                  f"{report.valid_paths}/{report.total_flows} valid paths)")
        print("=" * 80)

    def run_test(self, profile: str = 'light_1',
                 wait_seconds: float = METRICS_WAIT_TIME,
                 stop_after: bool = False,
                 retries: int = 0) -> MetricsReport:
        """
        Run full test cycle: start traffic, wait, verify, report.

        Args:
            profile: Traffic profile to use
            wait_seconds: Seconds to wait for metrics
            stop_after: Stop traffic after test
            retries: Number of retries if metrics missing

        Returns:
            MetricsReport
        """
        # Start traffic
        self.start_traffic(profile=profile)

        # Wait for metrics
        self.wait_for_metrics(wait_seconds=wait_seconds)

        # Verify and report
        report = self.verify_metrics()
        self.print_report(report)

        # Retry if needed
        for retry in range(retries):
            if len(report.missing_metrics) == 0 and report.invalid_paths == 0:
                break
            log.info(f"\nRetry {retry + 1}/{retries}: waiting additional 2s...")
            time.sleep(2.0)
            report = self.verify_metrics()
            self.print_report(report)

        # Stop traffic if requested
        if stop_after:
            self.stop_traffic()

        return report

    def close(self):
        """Clean up resources."""
        self.client.close()


# =============================================================================
#                              CLI INTERFACE
# =============================================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Test INT metrics collection for all traffic flows"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to topology YAML config (e.g., config/topologies/fat_tree_k4.yaml)'
    )
    parser.add_argument(
        '--profile', '-p',
        type=str,
        choices=tuple(TrafficManager.TRAFFIC_PROFILES),
        default='light_1',
        help='Traffic profile to use (default: light_1)'
    )
    parser.add_argument(
        '--wait', '-w',
        type=float,
        default=METRICS_WAIT_TIME,
        help=f'Seconds to wait for metrics (default: {METRICS_WAIT_TIME})'
    )
    parser.add_argument(
        '--influx-url',
        type=str,
        default='http://192.168.56.1:8086',
        help='InfluxDB URL'
    )
    parser.add_argument(
        '--influx-token',
        type=str,
        default=os.environ.get('INFLUX_TOKEN'),
        help='InfluxDB token (or set INFLUX_TOKEN env var)'
    )
    parser.add_argument(
        '--influx-org',
        type=str,
        default='Research',
        help='InfluxDB organization'
    )
    parser.add_argument(
        '--influx-bucket',
        type=str,
        default='INT',
        help='InfluxDB bucket'
    )
    parser.add_argument(
        '--topology-file',
        type=str,
        default='/tmp/topology.json',
        help='Path to topology.json'
    )
    parser.add_argument(
        '--stop-traffic',
        action='store_true',
        help='Stop traffic after test (default: leave running)'
    )
    parser.add_argument(
        '--retry',
        type=int,
        default=0,
        help='Number of retries if metrics missing (with 2s wait between)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--window',
        type=float,
        default=WINDOW_SECONDS,
        help=f'Query time window in seconds (default: {WINDOW_SECONDS})'
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check root
    if os.geteuid() != 0:
        print("ERROR: Must run with sudo!")
        print("Usage: sudo python3 int_metrics_tester.py --config <config.yaml>")
        sys.exit(1)

    # Create tester
    tester = INTMetricsTester(
        config_path=args.config,
        influx_url=args.influx_url,
        influx_token=args.influx_token,
        influx_org=args.influx_org,
        influx_bucket=args.influx_bucket,
        topology_file=args.topology_file,
        window_seconds=args.window,
    )

    try:
        # Run test
        report = tester.run_test(
            profile=args.profile,
            wait_seconds=args.wait,
            stop_after=args.stop_traffic,
            retries=args.retry,
        )

        # JSON output if requested
        if args.json:
            result = {
                'total_expected': report.total_expected,
                'total_found': report.total_found,
                'missing_count': len(report.missing_metrics),
                'total_flows': report.total_flows,
                'valid_paths': report.valid_paths,
                'invalid_paths': report.invalid_paths,
                'by_queue': report.by_queue,
                'by_measurement': report.by_measurement,
                'missing_metrics': [asdict(m) for m in report.missing_metrics],
                'path_deviations': [asdict(d) for d in report.path_deviations],
            }
            print("\n--- JSON OUTPUT ---")
            print(json.dumps(result, indent=2))

        # Exit with error code if metrics missing or paths invalid
        if len(report.missing_metrics) > 0 or report.invalid_paths > 0:
            sys.exit(1)

    finally:
        tester.close()


if __name__ == "__main__":
    main()
