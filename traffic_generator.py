#!/usr/bin/env python3
"""Resilient, reproducible traffic generation for routing experiments.

Traffic endpoints are started through P4Utils TaskServers and verified exactly.
Benchmark profiles use verified sender HTB caps. Steady profiles use fixed
ten-step schedules; bursty profiles use seeded random ten-step schedules,
allowing RL, ECMP, and OSPF to receive the same offered workload.
"""

import os
import sys
import json
import glob
import time
import random
import hashlib
import math
import threading
import subprocess
import logging
import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from p4utils.utils.task_scheduler import Task, TaskClient

# Import helpers from network.py
from network import _traffic_dst_port, QID_TOS, ALL_QUEUES

# Import unified logging
from logging_config import setup_unified_logging

log = logging.getLogger(__name__)


PROFILE_CYCLE_STEPS = 10
MEASUREMENT_SETTLE_SECONDS = 0.0
COMMON_QUEUE_WEIGHTS = {0: 0.22327, 1: 0.34591, 7: 0.43082}


def _weighted_load(total_mbps: float, weights: Dict[int, float]) -> Dict[int, float]:
    """Split an aggregate per-demand load using normalized queue weights."""
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("Traffic weights must have a positive sum")
    return {
        qid: total_mbps * weight / weight_total
        for qid, weight in weights.items()
    }


def _source_load_for(stages: Dict[str, Dict[int, float]]) -> Dict[int, float]:
    """Offer slightly above the largest shaped stage for every queue."""
    return {
        qid: max(stages["low"][qid], stages["high"][qid]) * 1.03
        for qid in ALL_QUEUES
    }


def _queue_biased_stages(weights: Dict[int, float]) -> Dict[str, Dict[int, float]]:
    stages = {
        "low": _weighted_load(0.5, weights),
        "high": _weighted_load(3.2, weights),
    }
    stages["source"] = _source_load_for(stages)
    return stages


def _steady_stages(
    low_total_mbps: float,
    high_total_mbps: Optional[float] = None,
) -> Dict[str, Dict[int, float]]:
    low = _weighted_load(low_total_mbps, COMMON_QUEUE_WEIGHTS)
    high = _weighted_load(
        low_total_mbps if high_total_mbps is None else high_total_mbps,
        COMMON_QUEUE_WEIGHTS,
    )
    stages = {"low": low, "high": high}
    stages["source"] = _source_load_for(stages)
    return stages


STEADY_PROFILE_SPECS = {
    # Near-knee stress points are intentionally close because fat_tree_k4 has
    # a sharp ECMP congestion transition. Light profiles are steady. Medium
    # and high profiles use deterministic high/low cycles around the knee so
    # they exercise partial SLA behavior without unbounded queue buildup.
    "light_1": {
        "low": 1.50,
        "high": 1.50,
        "pattern": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    "light_2": {
        "low": 1.62,
        "high": 1.62,
        "pattern": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    "medium_1": {
        "low": 1.50,
        "high": 1.80,
        "pattern": (0, 1, 0, 0, 1, 0, 0, 1, 0, 0),
    },
    "medium_2": {
        "low": 1.50,
        "high": 1.92,
        "pattern": (0, 1, 0, 1, 0, 0, 1, 0, 1, 0),
    },
    "high_1": {
        "low": 1.50,
        "high": 1.85,
        "pattern": (0, 1, 1, 0, 1, 1, 0, 1, 1, 0),
    },
    "high_2": {
        "low": 1.85,
        "high": 1.85,
        "pattern": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
}

BURST_QUEUE_WEIGHTS = {
    "vo": {0: 0.72, 1: 0.11, 7: 0.17},
    "vi": {0: 0.12, 1: 0.70, 7: 0.18},
    "be": {0: 0.10, 1: 0.15, 7: 0.75},
}
BURST_HIGH_STEPS = {
    1: 3,
    2: 6,
}
BURSTY_PROFILE_SPECS = {
    f"bursty_{traffic_class}_{tier}": {
        "weights": weights,
        "high_steps": high_steps,
    }
    for traffic_class, weights in BURST_QUEUE_WEIGHTS.items()
    for tier, high_steps in BURST_HIGH_STEPS.items()
}


BURSTY_PROFILE_HIGH_STEPS = {
    name: spec["high_steps"]
    for name, spec in BURSTY_PROFILE_SPECS.items()
}


def _contiguous_burst_pattern(high_steps: int, start_step: int) -> Tuple[int, ...]:
    """Return a ten-step burst window with wrap-around support."""
    if not 1 <= high_steps <= PROFILE_CYCLE_STEPS:
        raise ValueError(f"Unsupported burst duration: {high_steps} steps")
    if not 1 <= start_step <= PROFILE_CYCLE_STEPS:
        raise ValueError(f"Unsupported burst start step: {start_step}")
    high_indexes = {
        (start_step - 1 + offset) % PROFILE_CYCLE_STEPS
        for offset in range(high_steps)
    }
    return tuple(
        int(index in high_indexes)
        for index in range(PROFILE_CYCLE_STEPS)
    )


SHAPED_PROFILE_STAGES = {
    name: _steady_stages(spec["low"], spec["high"])
    for name, spec in STEADY_PROFILE_SPECS.items()
}
SHAPED_PROFILE_STAGES.update({
    name: _queue_biased_stages(spec["weights"])
    for name, spec in BURSTY_PROFILE_SPECS.items()
})

SHAPED_PROFILE_PATTERNS = {
    name: tuple(spec["pattern"])
    for name, spec in STEADY_PROFILE_SPECS.items()
}

PROFILE_GROUPS = {
    category: tuple(
        name
        for name in SHAPED_PROFILE_STAGES
        if name.startswith(f"{category}_")
    )
    for category in ("light", "medium", "high", "bursty")
}


def get_hosts_from_config(config_path: str, topology_file: str = "topology.json") -> Dict[str, str]:
    """
    Get host name to IP mapping from topology.json (runtime IPs) or config.

    Priority:
    1. topology.json - has actual Mininet-assigned IPs
    2. topology builder - fallback for pre-network planning

    Args:
        config_path: Path to YAML configuration file
        topology_file: Path to topology.json (default: topology.json)

    Returns:
        Dict mapping host names to IPs (e.g., {'h1': '10.13.1.2', 'h2': '10.13.2.2'})
    """
    # First, try to load from topology.json (has actual runtime IPs)
    try:
        import json
        with open(topology_file, 'r') as f:
            topo = json.load(f)

        hosts_ips = {}
        for node in topo.get('nodes', []):
            node_id = node.get('id', '')
            if isinstance(node_id, str) and node_id.startswith('h'):
                try:
                    host_num = int(node_id[1:])
                    if host_num < 100:  # Exclude collectors (h100+)
                        ip = node.get('ip', '')
                        if ip:
                            # Remove CIDR suffix if present
                            ip = ip.split('/')[0]
                            hosts_ips[node_id] = ip
                except ValueError:
                    continue

        if hosts_ips:
            log.info(f"Loaded {len(hosts_ips)} traffic hosts from {topology_file}")
            return hosts_ips

    except FileNotFoundError:
        log.debug(f"{topology_file} not found, falling back to config")
    except Exception as e:
        log.warning(f"Error loading {topology_file}: {e}")

    # Fallback: load from topology builder (only if config_path provided)
    if config_path:
        try:
            from topology.factory import create_topology

            builder = create_topology(config_path)
            hosts_ips = builder.get_host_ips()

            if hosts_ips:
                # Filter out collector hosts (h100+)
                traffic_hosts = {
                    name: ip for name, ip in hosts_ips.items()
                    if name.startswith('h') and int(name[1:]) < 100
                }
                log.info(f"Discovered {len(traffic_hosts)} traffic hosts from topology config")
                return traffic_hosts
            else:
                log.warning("No hosts found in topology config")
                return {}

        except Exception as e:
            log.warning(f"Error loading topology: {e}")
            return {}

    return {}


class TrafficManager:
    """Manages iperf3 traffic generation via P4Utils TaskClient.
    
    Uses TaskClient to send Task objects to TaskServer running on each host.
    This enables runtime traffic control without needing the Mininet net object.
    
    Explicit profiles and weighted profile categories share one canonical
    registry. Default training selection balances the steady profiles.
    """
    
    # Unique marker for identifying traffic processes (used by pkill)
    TRAFFIC_TAG = "__RL_TRAFFIC__"
    
    SHAPED_PROFILE_PATTERNS = SHAPED_PROFILE_PATTERNS
    PROFILE_STAGE_LOADS = SHAPED_PROFILE_STAGES
    BURSTY_PROFILE_HIGH_STEPS = BURSTY_PROFILE_HIGH_STEPS
    PROFILE_GROUPS = PROFILE_GROUPS
    MEASUREMENT_SETTLE_SECONDS = MEASUREMENT_SETTLE_SECONDS
    TRAFFIC_PROFILES = {
        name: {
            qid: (
                min(stages["low"][qid], stages["high"][qid]),
                max(stages["low"][qid], stages["high"][qid]),
            )
            for qid in stages["low"]
        }
        for name, stages in SHAPED_PROFILE_STAGES.items()
    }
    
    # Default host IP mapping (legacy Fat-Tree k=4 topology)
    DEFAULT_HOSTS_IPS = {
        "h1": "10.7.1.2",
        "h2": "10.7.2.2",
        "h3": "10.8.3.2",
        "h4": "10.8.4.2",
        "h5": "10.9.5.2",
        "h6": "10.9.6.2",
        "h7": "10.10.7.2",
        "h8": "10.10.8.2",
    }

    def __init__(self, topology_file: str = "/tmp/topology.json",
                 config_path: str = None, seed: int = None):
        """Initialize TrafficManager.

        Args:
            topology_file: Path to topology.json for host discovery (legacy)
            config_path: Path to YAML topology configuration for dynamic host/IP discovery
            seed: Optional deterministic seed for reproducible RL/ECMP comparisons
        """
        if os.geteuid() != 0:
            log.warning("TrafficManager: Not running as root. TaskClient may fail.")

        # Keep training variability by default, while allowing benchmark runs
        # to use the exact same profile scaling and burst schedule.
        self._seed_material = str(time.time_ns() if seed is None else int(seed))
        self._rng = random.Random(int(self._seed_material))

        self.topology_file = topology_file
        self.config_path = config_path
        self._topology_config = None

        # Load topology config if available
        if config_path:
            try:
                from config.loader import load_config
                self._topology_config = load_config(config_path)
            except Exception as e:
                log.warning(f"Failed to load topology config: {e}")

        # Load host IPs: try topology_file first (has runtime IPs), then config, then defaults
        self.hosts_ips = get_hosts_from_config(config_path, topology_file)
        if not self.hosts_ips:
            log.warning("No hosts found, falling back to defaults")
            self.hosts_ips = self.DEFAULT_HOSTS_IPS.copy()

        # Discover traffic hosts (h1-h8, excluding h100+)
        self.traffic_hosts = self._discover_traffic_hosts()
        log.info(f"TrafficManager: Found traffic hosts: {self.traffic_hosts}")

        # Build sender/receiver pairs from config or fallback to default logic
        self.traffic_pairs = self._build_traffic_pairs_from_config()

        # Extract unique senders and receivers from pairs
        self.senders = list(set(p[0] for p in self.traffic_pairs))
        self.receivers = list(set(p[1] for p in self.traffic_pairs))

        log.info(f"TrafficManager: {len(self.traffic_pairs)} traffic pairs configured")

        # Validate flow_id range
        max_flow_id = max(p[2] for p in self.traffic_pairs) if self.traffic_pairs else 0
        if max_flow_id > 99:
            raise ValueError(
                f"Maximum flow_id {max_flow_id} exceeds limit of 99 (has {len(self.traffic_pairs)} pairs). "
                f"Config path: {self.config_path}. "
                f"Check traffic pair configuration or increase flow_id limit in network.py"
            )
        log.info(f"Flow ID range: 10-{max_flow_id}")
        
        # Current profile state
        self.current_load: Dict[int, float] = {}
        self._source_load: Dict[int, float] = {}
        self.current_profile_name: str = ""
        self.current_profile_category: str = ""
        self._tc_original_classes: Dict[str, Dict[str, str]] = {}
        self._tc_shape_report: Dict = {}
        self._shaped_stage_high: Optional[bool] = None
        self._profile_step_offset: int = 0
        self._bursty_cycle_patterns: Dict[int, Tuple[int, ...]] = {}

        # iPerf log cleanup thread (CPU-efficient, runs every 120s)
        self._log_cleanup_stop = threading.Event()
        self._log_cleanup_thread = None

        # Balanced default selection covers steady profiles. Bursty profiles
        # are selected through the optional "bursty" category weight.
        self._profile_names = tuple(
            name
            for category in ("light", "medium", "high")
            for name in self.PROFILE_GROUPS[category]
        )
        self._episode_count = 0
        self._window_size = 60  # Rebalance weights every 60 episodes
        self._usage_counts = {name: 0 for name in self._profile_names}
        
        # Traffic transition tracking (for telemetry stability)
        self._in_transition = False
        self._transition_start_time = 0.0
        self._transition_stabilization_time = 3.0  # Seconds to wait for traffic to stabilize

        # Health monitoring thread (auto-restart crashed iperf processes)
        self._health_monitor_stop = threading.Event()
        self._health_monitor_thread = None
        self._traffic_active = False  # True when traffic should be running
        self._last_packet_len = 1250  # Remember packet length for restarts
        self._health_check_interval = 10.0  # Check every 10 seconds
        self._restart_count = 0  # Track restarts for logging
        self._startup_recovery_count = 0
        self._last_start_report = None
        self._traffic_failed = False
        self._last_health_error = None

        # Traffic logging - stores traffic configurations to CSV for analysis
        self._traffic_log_dir = Path("log")
        self._traffic_log_dir.mkdir(exist_ok=True)
        self._traffic_log_file = self._traffic_log_dir / "traffic_log.csv"
        self._traffic_log_initialized = False

    def _discover_traffic_hosts(self) -> List[str]:
        """Discover traffic hosts from hosts_ips dict or topology.json (hosts with id < 100)."""
        # If we loaded from config, use those host names
        if self.hosts_ips and self.hosts_ips != self.DEFAULT_HOSTS_IPS:
            hosts = [
                name for name in self.hosts_ips.keys()
                if name.startswith('h') and int(name[1:]) < 100
            ]
            if hosts:
                return sorted(hosts, key=lambda x: int(x[1:]))

        # Fallback: discover from topology.json
        hosts = []
        try:
            with open(self.topology_file, 'r') as f:
                topo = json.load(f)
            for node in topo.get('nodes', []):
                node_id = node.get('id', '')
                if isinstance(node_id, str) and node_id.startswith('h'):
                    try:
                        if int(node_id[1:]) < 100:
                            hosts.append(node_id)
                    except ValueError:
                        continue
        except Exception as e:
            log.warning(f"Failed to load topology: {e}, using hosts from hosts_ips")
            # Use hosts from hosts_ips dict
            hosts = [
                name for name in self.hosts_ips.keys()
                if name.startswith('h') and int(name[1:]) < 100
            ]
        return sorted(hosts, key=lambda x: int(x[1:]))

    def _log_traffic_config(self, event: str = "start", extra_info: Dict = None):
        """Log traffic configuration to CSV file for analysis.

        Args:
            event: Event type such as start, stop, restart, or a failure event.
            extra_info: Optional dict with additional info to log
        """
        try:
            # Initialize CSV with header if needed
            if not self._traffic_log_initialized:
                if not self._traffic_log_file.exists():
                    with open(self._traffic_log_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            'timestamp', 'event', 'profile_name', 'profile_category',
                            'load_q0', 'load_q1', 'load_q7', 'is_bursty',
                            'baseline_profile', 'num_traffic_pairs', 'extra_info'
                        ])
                self._traffic_log_initialized = True

            # Write traffic configuration row
            with open(self._traffic_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    event,
                    self.current_profile_name,
                    self.current_profile_category,
                    f"{self.current_load.get(0, 0):.3f}",
                    f"{self.current_load.get(1, 0):.3f}",
                    f"{self.current_load.get(7, 0):.3f}",
                    self.current_profile_category == "bursty",
                    '',
                    len(self.traffic_pairs),
                    json.dumps(extra_info) if extra_info else ''
                ])
        except Exception as e:
            log.warning(f"Failed to log traffic config: {e}")

    def _build_pods(self) -> List[List[str]]:
        """Build pod structure (pairs of hosts per ToR)."""
        pods = []
        sorted_hosts = sorted(self.traffic_hosts, key=lambda x: int(x[1:]))
        for i in range(0, len(sorted_hosts), 2):
            if i + 1 < len(sorted_hosts):
                pods.append([sorted_hosts[i], sorted_hosts[i+1]])
        return pods

    def _build_traffic_pairs_from_config(self) -> List[Tuple[str, str, int]]:
        """Build traffic pairs from YAML config or fallback to cross_pod pattern."""
        pairs = []
        next_flow_id = 10

        # Try to load explicit pairs from topology config
        if self._topology_config and self._topology_config.traffic.pairs:
            log.info("Loading traffic pairs from topology config")
            for pair in self._topology_config.traffic.pairs:
                src, dst = pair.src, pair.dst
                # Validate hosts exist
                if src not in self.hosts_ips:
                    log.warning(f"Traffic pair source '{src}' not found in hosts, skipping")
                    continue
                if dst not in self.hosts_ips:
                    log.warning(f"Traffic pair destination '{dst}' not found in hosts, skipping")
                    continue
                pairs.append((src, dst, next_flow_id))
                next_flow_id += 1
            if pairs:
                return pairs

        # Check for pattern-based generation
        if self._topology_config and self._topology_config.traffic.pattern:
            pattern = self._topology_config.traffic.pattern
            pod_size = self._topology_config.traffic.pod_size
            log.info(f"Generating traffic pairs with pattern: {pattern}")
            return self._generate_pairs_by_pattern(pattern, pod_size)

        # Fallback: use default cross_pod pattern
        log.info("Using default cross_pod traffic pattern")
        return self._build_traffic_pairs_cross_pod()

    def _generate_pairs_by_pattern(self, pattern: str, pod_size: int) -> List[Tuple[str, str, int]]:
        """Generate traffic pairs based on pattern."""
        pairs = []
        next_flow_id = 10
        sorted_hosts = sorted(self.traffic_hosts, key=lambda x: int(x[1:]))

        if pattern == "cross_pod":
            return self._build_traffic_pairs_cross_pod()

        elif pattern == "all_to_all":
            # Every host sends to every other host
            for src in sorted_hosts:
                for dst in sorted_hosts:
                    if src != dst:
                        pairs.append((src, dst, next_flow_id))
                        next_flow_id += 1

        elif pattern.startswith("random_"):
            # random_N: each host sends to N random other hosts
            try:
                n = int(pattern.split("_")[1])
            except (IndexError, ValueError):
                n = 3  # default
            for src in sorted_hosts:
                others = [h for h in sorted_hosts if h != src]
                targets = self._rng.sample(others, min(n, len(others)))
                for dst in targets:
                    pairs.append((src, dst, next_flow_id))
                    next_flow_id += 1

        else:
            log.warning(f"Unknown traffic pattern '{pattern}', falling back to cross_pod")
            return self._build_traffic_pairs_cross_pod()

        return pairs

    def _build_traffic_pairs_cross_pod(self) -> List[Tuple[str, str, int]]:
        """Build cross-pod traffic pairs (default pattern)."""
        pairs = []
        next_flow_id = 10
        pods = self._build_pods()
        senders = [pod[0] for pod in pods]

        for i, sender in enumerate(senders):
            for j, pod in enumerate(pods):
                if j == i:
                    continue
                pairs.append((sender, pod[1], next_flow_id))
                next_flow_id += 1
        return pairs

    def _build_traffic_pairs(self) -> List[Tuple[str, str, int]]:
        """Build (src, dst, flow_id) pairs - legacy method, use _build_traffic_pairs_from_config."""
        return self._build_traffic_pairs_cross_pod()
    
    def _host_to_ip(self, hostname: str) -> str:
        """Get IP for hostname from the hosts_ips mapping."""
        ip = self.hosts_ips.get(hostname)
        if ip is None:
            log.warning(f"No IP found for host {hostname}, using placeholder")
            return "0.0.0.0"
        return ip
    
    def _send_task(self, hostname: str, cmd: str, delay: float = 0.0,
                   max_retries: int = 3) -> bool:
        """Send a task to the host's TaskServer with retry.

        Args:
            hostname: Target host name
            cmd: Command to execute
            delay: Delay before task starts (seconds)
            max_retries: Maximum retry attempts on failure

        Returns:
            True if task was sent successfully, False otherwise
        """
        socket_path = f"/tmp/{hostname}_socket"
        if not os.path.exists(socket_path):
            log.warning(f"[TaskSend] Socket not found: {socket_path}")
            return False

        for attempt in range(max_retries):
            try:
                t0 = time.monotonic()
                client = TaskClient(socket_path)
                task = Task(cmd, start=time.time() + delay, duration=0)
                client.send([task], retry=False)
                elapsed_ms = (time.monotonic() - t0) * 1000
                if elapsed_ms > 100:  # Log slow sends
                    log.debug(f"[TaskSend] {hostname} slow: {elapsed_ms:.0f}ms")
                return True
            except PermissionError:
                log.error(f"[TaskSend] Permission denied for {hostname}. Run with sudo!")
                return False
            except BrokenPipeError as e:
                log.warning(f"[TaskSend] {hostname} broken pipe (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    backoff = 0.1 * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    return False
            except ConnectionRefusedError as e:
                log.warning(f"[TaskSend] {hostname} connection refused (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    backoff = 0.1 * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s
                    log.debug(f"[TaskSend] {hostname} retry {attempt+1}/{max_retries}: {e}")
                    time.sleep(backoff)
                else:
                    log.warning(f"[TaskSend] {hostname} failed after {max_retries} attempts: {e}")
                    return False
        return False

    def _send_tasks_batched(self, hostname: str, tasks: List[Task], max_retries: int = 3) -> bool:
        """Send multiple tasks to a host's TaskServer in a single connection.

        This reduces connection overhead from N connections to 1 connection per host,
        significantly reducing thread accumulation in TaskServer over long runs.

        Args:
            hostname: Target host name
            tasks: List of Task objects to send
            max_retries: Maximum retry attempts on failure

        Returns:
            True if tasks were sent successfully, False otherwise
        """
        if not tasks:
            return True

        socket_path = f"/tmp/{hostname}_socket"
        if not os.path.exists(socket_path):
            log.warning(f"[TaskSend] Socket not found for batch: {socket_path}")
            return False

        for attempt in range(max_retries):
            try:
                t0 = time.monotonic()
                client = TaskClient(socket_path)
                client.send(tasks, retry=False)
                elapsed_ms = (time.monotonic() - t0) * 1000
                if elapsed_ms > 200:  # Log slow batch sends
                    log.warning(f"[TaskSend] {hostname} batch slow: {len(tasks)} tasks in {elapsed_ms:.0f}ms")
                else:
                    log.debug(f"[TaskSend] {hostname} batch: {len(tasks)} tasks in {elapsed_ms:.0f}ms")
                return True
            except PermissionError:
                log.error(f"[TaskSend] Permission denied for {hostname}. Run with sudo!")
                return False
            except BrokenPipeError as e:
                log.warning(f"[TaskSend] {hostname} batch broken pipe (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    backoff = 0.1 * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    return False
            except ConnectionRefusedError as e:
                log.warning(f"[TaskSend] {hostname} batch connection refused (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    backoff = 0.1 * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = 0.1 * (2 ** attempt)
                    log.debug(f"[TaskSend] {hostname} batch retry {attempt+1}/{max_retries}: {e}")
                    time.sleep(backoff)
                else:
                    log.warning(f"[TaskSend] {hostname} batch failed after {max_retries} attempts: {e}")
                    return False
        return False

    def stop_traffic(self):
        """Stop all traffic processes using pkill."""
        # Stop health monitoring first to prevent auto-restart during stop
        self._traffic_active = False
        self._stop_health_monitor()
        self._stop_log_cleanup_thread()
        log.info("Stopping all traffic processes...")
        try:
            # Kill iperf3 processes on our ports (6xxx range: 6100-6657)
            # Port scheme: 6000 + flow_id*10 + qid, where flow_id in 10-65, qid in 0,1,7
            # This matches ports 6100-6657
            subprocess.run(['pkill', '-9', '-f', 'iperf3.*-p 6[1-6][0-9][0-9]'], capture_output=True)
            # Kill bash wrapper loops with our tag
            result = subprocess.run(
                ['pkill', '-9', '-f', f'bash.*{self.TRAFFIC_TAG}'],
                capture_output=True
            )
            if result.returncode == 0:
                log.info("Killed traffic processes")
            elif result.returncode == 1:
                log.info("No traffic processes to kill")
        except Exception as e:
            log.warning(f"Failed to kill traffic: {e}")

        self._restore_sender_rate_caps()

        # Log traffic stop event
        self._log_traffic_config(event="stop")

        # Clean up zombie processes to prevent process table exhaustion
        self._cleanup_zombie_processes()
        time.sleep(0.5)

    def _cleanup_zombie_processes(self):
        """Reap zombie child processes to prevent process table exhaustion.

        After many start/stop cycles, zombie bash wrapper processes can accumulate
        if the parent process doesn't wait() for them. This method reaps any
        zombie children.
        """
        reaped = 0
        try:
            while True:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break  # No more zombies to reap
                reaped += 1
        except ChildProcessError:
            pass  # No children to reap (normal case)
        except Exception as e:
            log.debug(f"[Cleanup] Error reaping zombies: {e}")

        if reaped > 0:
            log.debug(f"[Cleanup] Reaped {reaped} zombie processes")

    def _start_log_cleanup_thread(self):
        """Start background thread to trim iperf logs (CPU-efficient)."""
        if self._log_cleanup_thread is not None:
            return  # Already running
        self._log_cleanup_stop.clear()
        self._log_cleanup_thread = threading.Thread(
            target=self._log_cleanup_loop,
            daemon=True,
            name="iperf-log-cleanup"
        )
        self._log_cleanup_thread.start()

    def _stop_log_cleanup_thread(self):
        """Stop the log cleanup thread."""
        if self._log_cleanup_thread is None:
            return
        self._log_cleanup_stop.set()
        self._log_cleanup_thread.join(timeout=2)
        self._log_cleanup_thread = None

    def _log_cleanup_loop(self):
        """Periodically trim iperf log files (every 120s for low CPU usage)."""
        while not self._log_cleanup_stop.wait(timeout=120):
            self._trim_iperf_logs()

    def _trim_iperf_logs(self):
        """Trim iperf log files to ~10 min of data (size-based, CPU-efficient)."""
        max_size = 5 * 1024 * 1024   # 5 MB threshold
        keep_size = 4 * 1024 * 1024  # Keep last 4 MB

        for log_path in glob.glob("/tmp/*_iperf3_*.log"):
            try:
                # Only check size (cheap stat call), skip if under threshold
                if os.path.getsize(log_path) <= max_size:
                    continue
                # Only read/write when trimming is needed
                with open(log_path, 'rb') as f:
                    f.seek(-keep_size, 2)
                    f.readline()  # Skip partial line
                    data = f.read()
                with open(log_path, 'wb') as f:
                    f.write(data)
                log.debug(f"Trimmed iperf log: {log_path}")
            except Exception:
                pass  # Ignore errors (file may be in use)

    def _start_health_monitor(self):
        """Start background thread to monitor and restart crashed iperf processes."""
        if self._health_monitor_thread is not None:
            return  # Already running
        self._health_monitor_stop.clear()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="iperf-health-monitor"
        )
        self._health_monitor_thread.start()
        log.debug("[Health Monitor] Started")

    def _stop_health_monitor(self):
        """Stop the health monitor thread."""
        if self._health_monitor_thread is None:
            return
        self._health_monitor_stop.set()
        self._health_monitor_thread.join(timeout=2)
        self._health_monitor_thread = None
        log.debug("[Health Monitor] Stopped")

    def _health_monitor_loop(self):
        """Periodically check and restart missing iperf processes."""
        while not self._health_monitor_stop.wait(timeout=self._health_check_interval):
            if not self._traffic_active:
                continue
            try:
                self._check_and_restart_traffic()
            except Exception as e:
                log.warning(f"[Health Monitor] Error: {e}")

    def _check_and_restart_traffic(self):
        """Require the exact role/port endpoint inventory and recover atomically."""
        report = self.verify_exact_processes(
            # Client tasks are bash loops around iperf3. Under burst/reroute
            # stress an iperf3 child can exit and be relaunched by its wrapper
            # after a short sleep. Require a sustained mismatch before doing an
            # expensive whole-traffic restart, otherwise benchmark runs can be
            # marked invalid even though the wrapper self-healed.
            timeout=3.0,
            raise_on_error=False,
        )
        if report["verified"]:
            log.debug(
                "[Health Monitor] OK: exact traffic endpoint inventory verified"
            )
            return

        self._restart_count += 1
        self._in_transition = True
        self._transition_start_time = time.monotonic()
        log.warning(
            "[Health Monitor] Traffic endpoint mismatch; restarting complete "
            f"traffic set (restart #{self._restart_count}): "
            + "; ".join(report["errors"])
        )

        started_at = time.monotonic()
        try:
            recovery = self._ensure_complete_traffic(
                self._last_packet_len,
                max_attempts=2,
                context="health_monitor",
            )
            self._traffic_failed = False
            self._last_health_error = None
            self._log_traffic_config(
                event="restart",
                extra_info={
                    "reason": "health_monitor",
                    "restart_count": self._restart_count,
                    "runtime_ms": (
                        time.monotonic() - started_at
                    ) * 1000.0,
                    "verification": recovery,
                },
            )
            log.info(
                "[Health Monitor] Exact endpoint inventory restored: "
                f"{recovery['observed_total']}/"
                f"{recovery['expected_total']} processes"
            )
        except Exception as exc:
            self._traffic_failed = True
            self._last_health_error = str(exc)
            self._log_traffic_config(
                event="restart_failed",
                extra_info={
                    "reason": "health_monitor",
                    "restart_count": self._restart_count,
                    "error": str(exc),
                },
            )
            log.error(f"[Health Monitor] Traffic recovery failed: {exc}")

    def _actual_iperf_inventory(
        self,
    ) -> Tuple[int, Counter, Dict[int, int], List[str]]:
        """Inventory actual iperf3 role/port endpoints.

        Returns total processes, a Counter keyed by ``(role, port)``, per-queue
        process counts, and any unparseable command lines. Bash wrapper loops
        are excluded by matching the executable name exactly.
        """
        inventory = Counter()
        queue_counts = {qid: 0 for qid in ALL_QUEUES}
        unparseable = []
        try:
            result = subprocess.run(
                ['pgrep', '-a', '-x', 'iperf3'],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return 0, inventory, queue_counts, unparseable

        if result.returncode != 0:
            return 0, inventory, queue_counts, unparseable

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        for line in lines:
            match = re.search(r"(?:^|\s)-p\s+(\d+)(?:\s|$)", line)
            if not match:
                unparseable.append(line)
                continue
            port = int(match.group(1))
            if re.search(r"(?:^|\s)-s(?:\s|$)", line):
                role = "server"
            elif re.search(r"(?:^|\s)-c\s+\S+", line):
                role = "client"
            else:
                unparseable.append(line)
                continue

            inventory[(role, port)] += 1
            qid = port % 10
            if qid in queue_counts:
                queue_counts[qid] += 1
        return len(lines), inventory, queue_counts, unparseable

    def _actual_iperf_counts(self) -> Tuple[int, Dict[int, int]]:
        """Backward-compatible aggregate view of the endpoint inventory."""
        total, _, queue_counts, _ = self._actual_iperf_inventory()
        return total, queue_counts

    def verify_exact_processes(
        self,
        timeout: float = 8.0,
        require_no_restarts: bool = False,
        raise_on_error: bool = True,
    ) -> Dict:
        """Require the exact benchmark traffic process population."""
        expected_endpoints = Counter()
        for _, _, flow_id in self.traffic_pairs:
            for qid in ALL_QUEUES:
                port = _traffic_dst_port(flow_id, qid)
                expected_endpoints[("server", port)] += 1
                expected_endpoints[("client", port)] += 1

        expected_per_queue = len(self.traffic_pairs) * 2
        expected_total = sum(expected_endpoints.values())
        deadline = time.monotonic() + timeout
        total = 0
        inventory = Counter()
        queue_counts = {qid: 0 for qid in ALL_QUEUES}
        unparseable = []

        while time.monotonic() < deadline:
            (
                total,
                inventory,
                queue_counts,
                unparseable,
            ) = self._actual_iperf_inventory()
            if (
                total == expected_total
                and inventory == expected_endpoints
                and not unparseable
                and all(
                    queue_counts[qid] == expected_per_queue
                    for qid in ALL_QUEUES
                )
            ):
                break
            time.sleep(0.25)

        errors = []
        if total != expected_total:
            errors.append(
                f"total iperf3 processes expected {expected_total}, observed {total}"
            )
        for qid in ALL_QUEUES:
            if queue_counts[qid] != expected_per_queue:
                errors.append(
                    f"Q{qid} processes expected {expected_per_queue}, "
                    f"observed {queue_counts[qid]}"
                )
        missing_endpoints = []
        duplicate_endpoints = []
        unexpected_endpoints = []
        for endpoint, expected_count in expected_endpoints.items():
            observed_count = inventory.get(endpoint, 0)
            if observed_count < expected_count:
                missing_endpoints.append(
                    f"{endpoint[0]}:{endpoint[1]} "
                    f"expected={expected_count} observed={observed_count}"
                )
            elif observed_count > expected_count:
                duplicate_endpoints.append(
                    f"{endpoint[0]}:{endpoint[1]} "
                    f"expected={expected_count} observed={observed_count}"
                )
        for endpoint, observed_count in inventory.items():
            if endpoint not in expected_endpoints:
                unexpected_endpoints.append(
                    f"{endpoint[0]}:{endpoint[1]} observed={observed_count}"
                )
        if missing_endpoints:
            errors.append(
                "missing endpoints: " + ", ".join(missing_endpoints)
            )
        if duplicate_endpoints:
            errors.append(
                "duplicate endpoints: " + ", ".join(duplicate_endpoints)
            )
        if unexpected_endpoints:
            errors.append(
                "unexpected endpoints: " + ", ".join(unexpected_endpoints)
            )
        if unparseable:
            errors.append(
                f"{len(unparseable)} unparseable iperf3 command(s)"
            )
        restart_count = int(getattr(self, "_restart_count", 0))
        if require_no_restarts and restart_count != 0:
            errors.append(
                f"traffic health monitor performed {restart_count} "
                "restart(s) during the measured run"
            )

        report = {
            "verified": not errors,
            "expected_total": expected_total,
            "observed_total": total,
            "expected_per_queue": expected_per_queue,
            "observed_per_queue": queue_counts,
            "expected_endpoint_count": len(expected_endpoints),
            "observed_endpoint_count": len(inventory),
            "missing_endpoints": missing_endpoints,
            "duplicate_endpoints": duplicate_endpoints,
            "unexpected_endpoints": unexpected_endpoints,
            "unparseable_commands": unparseable,
            "restart_count": restart_count,
            "errors": errors,
        }
        if errors and raise_on_error:
            raise RuntimeError(
                "Traffic process verification failed: "
                + "; ".join(errors)
            )
        return report

    def _log_per_queue_status(self) -> Dict[int, int]:
        """Log process counts per queue for debugging.

        Returns:
            Dict mapping qid to running process count
        """
        expected = len(self.traffic_pairs) * 2  # servers + clients per queue
        _, queue_counts = self._actual_iperf_counts()
        status_parts = []

        for qid in ALL_QUEUES:
            status_parts.append(f"Q{qid}:{queue_counts[qid]}/{expected}")

        log.info(f"[Traffic] Per-queue status: {', '.join(status_parts)}")
        return queue_counts

    def _stop_all_iperf(self):
        """Stop all iperf processes without clearing traffic state."""
        try:
            subprocess.run(['pkill', '-9', '-f', 'iperf3.*-p 6[1-6][0-9][0-9]'], capture_output=True)
            subprocess.run(['pkill', '-9', '-f', f'bash.*{self.TRAFFIC_TAG}'], capture_output=True)
        except Exception as e:
            log.debug(f"Error stopping iperf: {e}")

    def _ensure_complete_traffic(
        self,
        packet_len: int,
        max_attempts: int = 3,
        context: str = "startup",
    ) -> Dict:
        """Start the complete endpoint set or fail closed.

        Each attempt begins by removing all old iperf processes, then launches
        every server and client and verifies the exact role/port inventory. If
        an attempt fails, all TaskServers are restarted before retrying.
        Partial traffic is never returned to the caller as a successful start.
        """
        attempts = []
        last_report = None

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._startup_recovery_count += 1
                log.warning(
                    f"[Traffic] {context} recovery attempt "
                    f"{attempt}/{max_attempts}: restarting TaskServers"
                )
                if not self._restart_all_taskservers():
                    attempts.append(
                        {
                            "attempt": attempt,
                            "taskservers_restarted": False,
                            "errors": ["TaskServer restart failed"],
                        }
                    )
                    continue
                time.sleep(1.0)

            self._stop_all_iperf()
            time.sleep(0.5)
            self._start_servers()
            time.sleep(1.0)
            self._start_clients(packet_len)

            report = self.verify_exact_processes(
                timeout=8.0,
                raise_on_error=False,
            )
            last_report = report
            attempts.append(
                {
                    "attempt": attempt,
                    "taskservers_restarted": attempt > 1,
                    "verified": report["verified"],
                    "observed_total": report["observed_total"],
                    "observed_per_queue": report["observed_per_queue"],
                    "errors": report["errors"],
                }
            )
            if report["verified"]:
                report = {
                    **report,
                    "context": context,
                    "attempts_used": attempt,
                    "attempt_history": attempts,
                    "startup_recovery_count": self._startup_recovery_count,
                }
                self._last_start_report = report
                self._traffic_failed = False
                log.info(
                    "[Traffic] Exact endpoint inventory verified: "
                    f"{report['observed_total']}/{report['expected_total']} "
                    f"processes; Q0={report['observed_per_queue'][0]}, "
                    f"Q1={report['observed_per_queue'][1]}, "
                    f"Q7={report['observed_per_queue'][7]}"
                )
                return report

            log.warning(
                f"[Traffic] {context} attempt {attempt}/{max_attempts} "
                "did not create the complete endpoint set: "
                + "; ".join(report["errors"])
            )

        self._stop_all_iperf()
        self._traffic_active = False
        self._traffic_failed = True
        failure = {
            "verified": False,
            "context": context,
            "attempts_used": len(attempts),
            "attempt_history": attempts,
            "last_report": last_report,
            "startup_recovery_count": self._startup_recovery_count,
        }
        self._last_start_report = failure
        raise RuntimeError(
            f"Unable to establish complete traffic after {max_attempts} "
            f"attempts ({context})"
            + (
                ": " + "; ".join(last_report["errors"])
                if last_report
                else ""
            )
        )

    @classmethod
    def profile_category(cls, profile_name: str) -> str:
        """Return the single category containing ``profile_name``."""
        for category, profiles in cls.PROFILE_GROUPS.items():
            if profile_name in profiles:
                return category
        raise ValueError(
            f"Unknown traffic profile {profile_name!r}; valid profiles: "
            f"{', '.join(cls.TRAFFIC_PROFILES)}"
        )

    def _choose_profile_name(
        self,
        profile_name: Optional[str],
        category_weights: Optional[Dict[str, float]],
    ) -> str:
        """Select and validate a profile from the canonical registry."""
        if profile_name is not None:
            if profile_name not in self.TRAFFIC_PROFILES:
                raise ValueError(
                    f"Unknown traffic profile {profile_name!r}; valid profiles: "
                    f"{', '.join(self.TRAFFIC_PROFILES)}"
                )
            return profile_name

        if category_weights is not None:
            weighted_categories = []
            weights = []
            for category, raw_weight in category_weights.items():
                if category not in self.PROFILE_GROUPS:
                    raise ValueError(
                        f"Unknown traffic category {category!r}; valid categories: "
                        f"{', '.join(self.PROFILE_GROUPS)}"
                    )
                weight = float(raw_weight)
                if not math.isfinite(weight):
                    raise ValueError(
                        f"Traffic category weight for {category!r} must be finite"
                    )
                if weight < 0:
                    raise ValueError(
                        f"Traffic category weight for {category!r} cannot be negative"
                    )
                if weight > 0:
                    weighted_categories.append(category)
                    weights.append(weight)
            if not weighted_categories:
                raise ValueError("At least one traffic category weight must be positive")
            category = self._rng.choices(
                weighted_categories,
                weights=weights,
                k=1,
            )[0]
            return self._rng.choice(self.PROFILE_GROUPS[category])

        self._episode_count += 1
        if self._episode_count % self._window_size == 1:
            self._usage_counts = {name: 0 for name in self._profile_names}

        max_usage = max(self._usage_counts.values(), default=0)
        weights = [
            max_usage + 1 - self._usage_counts[name]
            for name in self._profile_names
        ]
        selected = self._rng.choices(
            self._profile_names,
            weights=weights,
            k=1,
        )[0]
        self._usage_counts[selected] += 1
        return selected

    @classmethod
    def _stage_loads(cls, profile_name: str, stage_high: bool) -> Dict[int, float]:
        stage = "high" if stage_high else "low"
        return cls.PROFILE_STAGE_LOADS[profile_name][stage].copy()

    def _bursty_pattern_for_cycle(self, cycle_index: int) -> Tuple[int, ...]:
        """Return the seeded random burst pattern for a zero-based cycle."""
        if cycle_index < 0:
            raise ValueError("cycle_index must be non-negative")
        cached = self._bursty_cycle_patterns.get(cycle_index)
        if cached is not None:
            return cached

        profile_name = self.current_profile_name
        high_steps = self.BURSTY_PROFILE_HIGH_STEPS[profile_name]
        seed_material = (
            f"{getattr(self, '_seed_material', 'unseeded')}:"
            f"{profile_name}:{cycle_index}"
        )
        seed_int = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
            "big",
        )
        rng = random.Random(seed_int)
        start_step = rng.randint(1, PROFILE_CYCLE_STEPS)
        pattern = _contiguous_burst_pattern(high_steps, start_step)
        self._bursty_cycle_patterns[cycle_index] = pattern
        return pattern

    def _scheduled_stage_high(self, profile_step: int) -> bool:
        """Return whether the active profile should be in its high stage."""
        if profile_step < 1:
            raise ValueError("profile_step must be at least 1")
        if self.current_profile_name in self.BURSTY_PROFILE_HIGH_STEPS:
            cycle_index = (profile_step - 1) // PROFILE_CYCLE_STEPS
            cycle_position = (profile_step - 1) % PROFILE_CYCLE_STEPS
            return bool(
                self._bursty_pattern_for_cycle(cycle_index)[cycle_position]
            )

        pattern = self.SHAPED_PROFILE_PATTERNS.get(self.current_profile_name)
        if not pattern:
            raise ValueError(
                f"No shaped traffic pattern for profile {self.current_profile_name!r}"
            )
        return bool(pattern[(profile_step - 1) % len(pattern)])

    @staticmethod
    def _parse_root_htb_class(output: str) -> Optional[Dict[str, str]]:
        """Parse Mininet's root HTB class from ``tc class show`` output."""
        match = re.search(
            r"class htb (?P<classid>\S+) root .*?"
            r"rate (?P<rate>\S+) ceil (?P<ceil>\S+) .*?"
            r"burst (?P<burst>\S+) cburst (?P<cburst>\S+)",
            output,
        )
        return match.groupdict() if match else None

    @staticmethod
    def _tc_rate_to_mbps(value: str) -> float:
        """Convert tc rate strings such as ``4740Kbit`` to Mbps."""
        match = re.fullmatch(
            r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>[KMG]?bit)",
            value,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(f"Unsupported tc rate value: {value!r}")
        amount = float(match.group("value"))
        factor = {
            "bit": 1e-6,
            "kbit": 1e-3,
            "mbit": 1.0,
            "gbit": 1e3,
        }[match.group("unit").lower()]
        return amount * factor

    def _set_sender_rate_cap(self, hostname: str, rate_mbps: float) -> Dict:
        """Set and verify the existing Mininet root HTB class rate."""
        host_pid = self._get_host_pid(hostname)
        if host_pid is None:
            raise RuntimeError(f"Cannot find Mininet PID for sender {hostname}")

        interface = f"{hostname}-eth0"
        show_cmd = [
            "mnexec", "-a", str(host_pid),
            "tc", "class", "show", "dev", interface,
        ]
        before = subprocess.run(
            show_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        parsed = self._parse_root_htb_class(before)
        if parsed is None:
            raise RuntimeError(
                f"Unable to identify root HTB class on {interface}: {before!r}"
            )
        self._tc_original_classes.setdefault(
            hostname,
            {
                **parsed,
                "pid": str(host_pid),
                "interface": interface,
            },
        )

        rate_kbit = max(1, int(round(rate_mbps * 1000.0)))
        subprocess.run(
            [
                "mnexec", "-a", str(host_pid),
                "tc", "class", "change", "dev", interface,
                "classid", parsed["classid"], "htb",
                "rate", f"{rate_kbit}kbit",
                "ceil", f"{rate_kbit}kbit",
                "burst", parsed["burst"],
                "cburst", parsed["cburst"],
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        after = subprocess.run(
            show_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        verified = self._parse_root_htb_class(after)
        if verified is None or verified["classid"] != parsed["classid"]:
            raise RuntimeError(
                f"Unable to verify sender cap on {interface}: {after!r}"
            )
        observed_rate = self._tc_rate_to_mbps(verified["rate"])
        observed_ceil = self._tc_rate_to_mbps(verified["ceil"])
        tolerance = max(0.002, rate_mbps * 0.01)
        if (
            abs(observed_rate - rate_mbps) > tolerance
            or abs(observed_ceil - rate_mbps) > tolerance
        ):
            raise RuntimeError(
                f"Sender cap verification mismatch on {interface}: "
                f"target={rate_mbps:.6f} Mbps, "
                f"rate={observed_rate:.6f}, ceil={observed_ceil:.6f}"
            )
        return {
            "host": hostname,
            "interface": interface,
            "classid": parsed["classid"],
            "target_rate_mbps": rate_mbps,
            "observed_rate_mbps": observed_rate,
            "observed_ceil_mbps": observed_ceil,
            "tc_state": verified,
        }

    def _apply_sender_caps(self, profile_name: str) -> Dict:
        """Shape each sender to the profile's exact aggregate offered load."""
        pair_counts = Counter(sender for sender, _, _ in self.traffic_pairs)
        per_pair_total = sum(self.current_load.values())
        reports = []
        try:
            for sender in sorted(pair_counts):
                reports.append(
                    self._set_sender_rate_cap(
                        sender,
                        per_pair_total * pair_counts[sender],
                    )
                )
        except Exception:
            self._restore_sender_rate_caps()
            raise

        self._tc_shape_report = {
            "verified": True,
            "profile": profile_name,
            "stage": (
                "high"
                if self._shaped_stage_high is True
                else (
                    "low"
                    if self._shaped_stage_high is False
                    else "startup"
                )
            ),
            "per_pair_total_mbps": per_pair_total,
            "senders": reports,
        }
        log.info(
            f"[Traffic] Applied verified sender HTB caps for {profile_name}: "
            f"{per_pair_total:.4f} Mbps per demand"
        )
        return self._tc_shape_report

    def apply_step_profile(
        self,
        current_step: int,
        use_offset: bool = True,
    ) -> Optional[Dict]:
        """Apply the scheduled stage for a shaped benchmark profile."""
        has_pattern = (
            self.current_profile_name in self.SHAPED_PROFILE_PATTERNS
            or self.current_profile_name in self.BURSTY_PROFILE_HIGH_STEPS
        )
        if not has_pattern:
            return None
        if current_step < 1:
            raise ValueError("current_step must be at least 1")

        profile_step = current_step + (
            getattr(self, "_profile_step_offset", 0) if use_offset else 0
        )
        stage_high = self._scheduled_stage_high(profile_step)
        if stage_high == self._shaped_stage_high:
            return None

        self._shaped_stage_high = stage_high
        self.current_load = self._stage_loads(
            self.current_profile_name,
            stage_high,
        )
        report = self._apply_sender_caps(
            self.current_profile_name
        )
        report["step"] = current_step
        report["profile_step"] = profile_step
        pattern_note = ""
        if self.current_profile_name in self.BURSTY_PROFILE_HIGH_STEPS:
            cycle_index = (profile_step - 1) // PROFILE_CYCLE_STEPS
            cycle_pattern = "".join(
                str(bit) for bit in self._bursty_pattern_for_cycle(cycle_index)
            )
            report["profile_cycle"] = cycle_index + 1
            report["cycle_pattern"] = cycle_pattern
            pattern_note = f" cycle={cycle_index + 1} pattern={cycle_pattern}"
        log.info(
            f"[Traffic] Step {current_step}: "
            f"profile_step={profile_step} "
            f"{self.current_profile_name} -> "
            f"{'HIGH' if stage_high else 'LOW'} "
            f"({sum(self.current_load.values()):.3f} Mbps per demand)"
            f"{pattern_note}"
        )
        return report

    def warm_profile_for(
        self,
        duration_seconds: float,
        step_interval_seconds: float = 1.0,
    ) -> Dict:
        """Advance the active profile during an excluded warmup window.

        ``start_traffic()`` activates profile step 1 immediately. This method
        advances later profile steps during warmup so measured step 1 continues
        from a warmed-up deterministic traffic pattern instead of replaying the
        profile from a cold queue state.
        """
        duration = max(0.0, float(duration_seconds))
        interval = max(0.1, float(step_interval_seconds))
        transitions = int(duration // interval)
        applied = []

        for transition in range(1, transitions + 1):
            time.sleep(interval)
            report = self.apply_step_profile(
                transition + 1,
                use_offset=False,
            )
            if report:
                applied.append(report)

        remaining = duration - transitions * interval
        if remaining > 0:
            time.sleep(remaining)

        self._profile_step_offset = transitions
        log.info(
            f"[Traffic] Warmed profile {self.current_profile_name} for "
            f"{duration:.1f}s; measurement starts at profile_step="
            f"{self._profile_step_offset + 1}"
        )
        return {
            "duration_seconds": duration,
            "step_interval_seconds": interval,
            "profile_step_offset": self._profile_step_offset,
            "stage_changes": applied,
        }

    def telemetry_coverage_window_seconds(
        self,
        minimum_seconds: float = 5.0,
    ) -> float:
        """Return the audit window needed to observe all profile flows.

        Bursty profiles intentionally spend much of each cycle at very low
        rates for non-dominant queues. A short exact-flow audit can therefore
        land in an off-burst slice and falsely report missing flows even when
        the measured run had valid telemetry at every step. Use a longer
        window for bursty workloads so coverage audits span at least one
        burst/recovery cycle while still requiring exact flow IDs.
        """
        minimum = max(0.0, float(minimum_seconds))
        if self.current_profile_category == "bursty":
            return max(minimum, 30.0)
        return minimum

    def begin_measurement(
        self,
        settle_seconds: float = MEASUREMENT_SETTLE_SECONDS,
    ) -> Optional[Dict]:
        """Ensure the first measurement stage is active.

        The target stage is normally already active because ``start_traffic()``
        sets sender caps before launching iperf clients. This method remains
        as an idempotent compatibility hook for runners that call it before
        entering their measured loop.
        """
        pattern = self.SHAPED_PROFILE_PATTERNS.get(
            self.current_profile_name
        )
        has_pattern = (
            bool(pattern)
            or self.current_profile_name in self.BURSTY_PROFILE_HIGH_STEPS
        )
        if not has_pattern:
            return None

        profile_step = getattr(self, "_profile_step_offset", 0) + 1
        stage_high = self._scheduled_stage_high(profile_step)
        target_load = self._stage_loads(
            self.current_profile_name,
            stage_high,
        )
        already_active = (
            self._shaped_stage_high == stage_high
            and self.current_load == target_load
            and bool(getattr(self, "_tc_shape_report", {}).get("verified"))
        )
        if already_active:
            report = dict(getattr(self, "_tc_shape_report", {}))
            report["phase"] = "measurement_start"
            report["already_active"] = True
            report["settle_seconds"] = 0.0
            log.info(
                f"[Traffic] Measurement load already active for "
                f"{self.current_profile_name}: "
                f"{sum(self.current_load.values()):.3f} Mbps per demand"
            )
            return report

        self._shaped_stage_high = stage_high
        self.current_load = target_load
        report = self._apply_sender_caps(self.current_profile_name)
        report["phase"] = "measurement_start"
        report["already_active"] = False
        report["settle_seconds"] = float(settle_seconds)
        if settle_seconds > 0:
            time.sleep(float(settle_seconds))
        log.info(
            f"[Traffic] Measurement load ready for "
            f"{self.current_profile_name}: "
            f"{sum(self.current_load.values()):.3f} Mbps per demand"
        )
        return report

    def _restore_sender_rate_caps(self) -> None:
        """Restore Mininet's original sender HTB classes after shaped traffic."""
        states = getattr(self, "_tc_original_classes", {})
        for hostname, state in list(states.items()):
            try:
                subprocess.run(
                    [
                        "mnexec", "-a", state["pid"],
                        "tc", "class", "change", "dev", state["interface"],
                        "classid", state["classid"], "htb",
                        "rate", state["rate"],
                        "ceil", state["ceil"],
                        "burst", state["burst"],
                        "cburst", state["cburst"],
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
            except Exception as exc:
                log.warning(
                    f"[Traffic] Failed restoring sender cap for {hostname}: {exc}"
                )
        states.clear()
        self._tc_shape_report = {}

    def start_traffic(
        self,
        packet_len: int = 1250,
        category_weights: Optional[Dict[str, float]] = None,
        profile_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start traffic with a selected profile.

        Args:
            packet_len: UDP packet length in bytes
            category_weights: Optional weights for keys in ``PROFILE_GROUPS``.
                A category is selected first, followed by one of its profiles.
            profile_name: Optional name from ``TRAFFIC_PROFILES``. An explicit
                profile takes precedence over category weights.

        Returns:
            Profile, offered-load, and exact startup verification metadata.
        """
        self.stop_traffic()
        time.sleep(0.3)
        self._start_log_cleanup_thread()
        self._startup_recovery_count = 0
        self._last_start_report = None

        # Mark traffic as transitioning - telemetry will be unstable during ramp-up
        self._in_transition = True
        self._transition_start_time = time.monotonic()

        self.current_profile_name = self._choose_profile_name(
            profile_name,
            category_weights,
        )
        self.current_profile_category = self.profile_category(
            self.current_profile_name
        )
        is_bursty = self.current_profile_category == "bursty"
        self._bursty_cycle_patterns = {}

        # Apply the profile's first scheduled stage immediately. This keeps the
        # warmup and telemetry-coverage windows on the same offered workload as
        # the measured episode instead of starting every profile at a benign
        # low-rate validation load.
        stages = self.PROFILE_STAGE_LOADS[self.current_profile_name]
        self._source_load = stages["source"].copy()
        self._profile_step_offset = 0
        self._shaped_stage_high = self._scheduled_stage_high(1)
        self.current_load = self._stage_loads(
            self.current_profile_name,
            self._shaped_stage_high,
        )
        
        log.info(f"Starting profile '{self.current_profile_name}' ({self.current_profile_category})")
        log.info(f"  Loads: Q0={self.current_load[0]:.2f}, Q1={self.current_load[1]:.2f}, Q7={self.current_load[7]:.2f} Mbps")
        if is_bursty:
            initial_pattern = "".join(
                str(bit) for bit in self._bursty_pattern_for_cycle(0)
            )
            log.info(
                f"  Bursty cycle 1 pattern: {initial_pattern} "
                f"({sum(self._bursty_pattern_for_cycle(0))}/"
                f"{PROFILE_CYCLE_STEPS} high steps)"
            )

        # Track state for health monitoring
        self._last_packet_len = packet_len
        self._traffic_active = True
        self._traffic_failed = False
        self._last_health_error = None

        shaping_report = self._apply_sender_caps(
            self.current_profile_name
        )

        try:
            start_report = self._ensure_complete_traffic(
                packet_len,
                max_attempts=3,
                context="start_traffic",
            )
        except Exception as exc:
            self._restore_sender_rate_caps()
            self._log_traffic_config(
                event="start_failed",
                extra_info={
                    "packet_len": packet_len,
                    "reason": "complete_endpoint_verification_failed",
                    "error": str(exc),
                    "verification": self._last_start_report,
                },
            )
            raise

        # Start health monitoring to auto-restart crashed processes
        self._start_health_monitor()

        # Log traffic configuration to CSV
        self._log_traffic_config(
            event="start",
            extra_info={
                "packet_len": packet_len,
                "verification": start_report,
            },
        )

        log.info("Traffic generation started")
        return {
            'profile_name': self.current_profile_name,
            'profile_category': self.current_profile_category,
            'loads': self.current_load.copy(),
            'measurement_loads': self.current_load.copy(),
            'is_bursty': is_bursty,
            'traffic_verified': True,
            'start_verification': start_report,
            'traffic_shaping': shaping_report,
        }
    
    def _start_servers(self):
        """Start iperf3 servers on receiver hosts with batched task sending.

        Uses batched sending to reduce socket connections from N tasks to N hosts,
        significantly reducing thread accumulation in TaskServer over long runs.
        """
        t0 = time.monotonic()

        # Group tasks by host for batched sending
        host_tasks: Dict[str, List[Task]] = {}
        total_tasks = 0

        for _, receiver, flow_id in self.traffic_pairs:
            if receiver not in host_tasks:
                host_tasks[receiver] = []

            for idx, qid in enumerate(ALL_QUEUES):
                port = _traffic_dst_port(flow_id, qid)
                cmd = (
                    f"bash -lc '{self.TRAFFIC_TAG}=1; "
                    f"while true; do iperf3 -s -p {port} -i 1 "
                    f"--logfile /tmp/{receiver}_iperf3_s_{port}.log; sleep 1; done'"
                )
                # Start all servers immediately (no stagger) - iperf3 handles concurrent starts
                # NOTE: Previously used stagger (Q0=0.0s, Q1=0.2s, Q7=0.4s) which caused race
                # conditions where Q1's 0.2s delay was in the "collision zone" after batched
                # task transmission delays (50-200ms).
                task = Task(cmd, start=0, duration=0)
                host_tasks[receiver].append(task)
                total_tasks += 1

        # Send batched tasks to each host (one connection per host)
        failed_hosts = []
        for hostname, tasks in host_tasks.items():
            if not self._send_tasks_batched(hostname, tasks):
                failed_hosts.append(hostname)

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Retry failed hosts after a brief pause
        if failed_hosts:
            log.warning(f"[Traffic] Server batch failed for {len(failed_hosts)} hosts: {failed_hosts}")
            time.sleep(0.5)
            retry_failed = []
            for hostname in failed_hosts:
                if not self._send_tasks_batched(hostname, host_tasks[hostname]):
                    retry_failed.append(hostname)
            if retry_failed:
                log.error(f"[Traffic] Server retry failed for hosts: {retry_failed}")
            else:
                log.info(f"[Traffic] Server retry succeeded for all {len(failed_hosts)} hosts")

        # Log summary
        num_hosts = len(host_tasks)
        final_failed = len(retry_failed) if failed_hosts else 0
        if final_failed > 0:
            log.warning(f"[Traffic] Servers: {num_hosts - final_failed}/{num_hosts} hosts "
                       f"({total_tasks} tasks) in {elapsed_ms:.0f}ms")
        else:
            log.debug(f"[Traffic] Servers: {num_hosts} hosts ({total_tasks} tasks) in {elapsed_ms:.0f}ms")
    
    def _start_clients(self, packet_len: int):
        """Start iperf3 clients on sender hosts with batched task sending.

        Uses batched sending to reduce socket connections from N tasks to N hosts,
        significantly reducing thread accumulation in TaskServer over long runs.
        """
        t0 = time.monotonic()

        # Group tasks by host for batched sending
        host_tasks: Dict[str, List[Task]] = {}
        total_tasks = 0

        for sender, receiver, flow_id in self.traffic_pairs:
            if sender not in host_tasks:
                host_tasks[sender] = []

            dst_ip = self._host_to_ip(receiver)
            for idx, qid in enumerate(ALL_QUEUES):
                port = _traffic_dst_port(flow_id, qid)
                tos = QID_TOS.get(qid, 0)
                bw = self._source_load.get(
                    qid,
                    self.current_load.get(qid, 0.2),
                )
                cmd = (
                    f"bash -lc '{self.TRAFFIC_TAG}=1; "
                    f"while true; do iperf3 -c {dst_ip} -p {port} -u "
                    f"-b {bw}M -l {packet_len} --tos {tos} -i 1 -t 0 "
                    f"--connect-timeout 5000 >> /tmp/{sender}_iperf3_c_{port}.log 2>&1; "
                    f"sleep 1; done'"
                )
                # Start all clients immediately (no per-queue stagger)
                # NOTE: Previously used stagger (Q0=0.5s, Q1=0.6s, Q7=0.7s) which caused timing
                # race conditions. Clients rely on the bash loop's retry mechanism if server
                # isn't ready yet. The 1-second sleep between iperf3 attempts handles this.
                task = Task(cmd, start=0, duration=0)
                host_tasks[sender].append(task)
                total_tasks += 1

        # Send batched tasks to each host (one connection per host)
        failed_hosts = []
        for hostname, tasks in host_tasks.items():
            if not self._send_tasks_batched(hostname, tasks):
                failed_hosts.append(hostname)

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Retry failed hosts after a brief pause
        if failed_hosts:
            log.warning(f"[Traffic] Client batch failed for {len(failed_hosts)} hosts: {failed_hosts}")
            time.sleep(1.0)
            retry_failed = []
            for hostname in failed_hosts:
                if not self._send_tasks_batched(hostname, host_tasks[hostname]):
                    retry_failed.append(hostname)
            if retry_failed:
                log.error(f"[Traffic] Client retry failed for hosts: {retry_failed}")
            else:
                log.info(f"[Traffic] Client retry succeeded for all {len(failed_hosts)} hosts")

        # Log summary
        num_hosts = len(host_tasks)
        final_failed = len(retry_failed) if failed_hosts else 0
        if final_failed > 0:
            log.warning(f"[Traffic] Clients: {num_hosts - final_failed}/{num_hosts} hosts "
                       f"({total_tasks} tasks) in {elapsed_ms:.0f}ms")
        else:
            log.debug(f"[Traffic] Clients: {num_hosts} hosts ({total_tasks} tasks) in {elapsed_ms:.0f}ms")

    def _verify_traffic_started(
        self,
        timeout: float = 8.0,
        min_pct: float = 1.0,
    ) -> bool:
        """Compatibility wrapper; startup verification is now exact.

        ``min_pct`` is retained for callers but values below 1.0 are ignored.
        A start is successful only when every expected server/client port is
        present exactly once.
        """
        if min_pct < 1.0:
            log.debug(
                "[Traffic] Ignoring permissive min_pct=%s; exact endpoint "
                "verification is mandatory",
                min_pct,
            )
        return self.verify_exact_processes(
            timeout=timeout,
            raise_on_error=False,
        )["verified"]

    def _get_host_pid(self, hostname: str) -> Optional[int]:
        """Get the PID of a Mininet host's bash process.

        Used for running commands in host namespace via mnexec.

        Args:
            hostname: Host name (e.g., 'h1')

        Returns:
            PID of the host's bash process, or None if not found
        """
        try:
            result = subprocess.run(
                ['pgrep', '-f', f'mininet:{hostname}$'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                pid = int(result.stdout.strip().split('\n')[0])
                return pid
        except Exception as e:
            log.warning(f"[TaskServer] Error finding PID for {hostname}: {e}")
        return None

    def _restart_taskserver(self, hostname: str, timeout: float = 10.0) -> bool:
        """Restart TaskServer on a specific host using mnexec.

        Kills any existing TaskServer for the host and starts a new one.

        Args:
            hostname: Host name (e.g., 'h1')
            timeout: Max time to wait for socket to appear

        Returns:
            True if TaskServer restarted successfully, False otherwise
        """
        socket_path = f"/tmp/{hostname}_socket"

        host_pid = self._get_host_pid(hostname)
        if host_pid is None:
            log.error(f"[TaskServer] Cannot find PID for host {hostname}")
            return False

        # Kill existing TaskServer
        try:
            subprocess.run(
                ['pkill', '-9', '-f', f'task_scheduler.*{hostname}_socket'],
                capture_output=True, timeout=10
            )
            if os.path.exists(socket_path):
                os.remove(socket_path)
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"[TaskServer] Error killing old TaskServer for {hostname}: {e}")

        # Start new TaskServer using mnexec
        try:
            subprocess.Popen(
                ['mnexec', '-a', str(host_pid), 'python3', '-m', 'p4utils.utils.task_scheduler', socket_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
            )

            # Wait for socket to appear
            start_time = time.monotonic()
            while time.monotonic() - start_time < timeout:
                if os.path.exists(socket_path):
                    time.sleep(0.2)  # Brief delay for socket to be ready
                    log.info(f"[TaskServer] Restarted TaskServer for {hostname}")
                    return True
                time.sleep(0.2)

            log.error(f"[TaskServer] Timeout waiting for {hostname} socket")
            return False
        except Exception as e:
            log.error(f"[TaskServer] Failed to restart TaskServer for {hostname}: {e}")
            return False

    def _restart_all_taskservers(self) -> bool:
        """Restart TaskServers on all traffic hosts.

        Used as a recovery mechanism when traffic fails to start due to
        TaskServer becoming unresponsive from thread accumulation.

        Returns:
            True if all TaskServers restarted successfully, False if any failed
        """
        log.warning("[TaskServer] Restarting all TaskServers...")

        # Get unique hosts from traffic pairs
        hosts = set()
        for sender, receiver, _ in self.traffic_pairs:
            hosts.add(sender)
            hosts.add(receiver)

        # Restart each host's TaskServer
        failed = []
        for hostname in sorted(hosts):
            if not self._restart_taskserver(hostname):
                failed.append(hostname)

        if failed:
            log.error(f"[TaskServer] Failed to restart: {failed}")
            return False

        log.info(f"[TaskServer] All {len(hosts)} TaskServers restarted successfully")
        return True

    def is_traffic_stable(self) -> bool:
        """Check if traffic has had time to stabilize after a transition.

        After start_traffic() or _restore_baseline_traffic() is called, traffic
        needs time to ramp up before telemetry data is reliable. This method
        returns False during the ramp-up period.

        Returns:
            True if traffic is stable and telemetry can be trusted,
            False if still transitioning (within stabilization window).
        """
        if not self._in_transition:
            return True
        elapsed = time.monotonic() - self._transition_start_time
        if elapsed >= self._transition_stabilization_time:
            self._in_transition = False
            return True
        return False

    def get_transition_elapsed(self) -> float:
        """Get elapsed time since traffic transition started.

        Returns:
            Seconds since transition started, or 0.0 if not in transition.
        """
        if not self._in_transition:
            return 0.0
        return time.monotonic() - self._transition_start_time


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dynamic traffic generation for RL training episodes"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML topology configuration for dynamic host/IP discovery'
    )
    parser.add_argument(
        '--topology-file',
        type=str,
        default='/tmp/topology.json',
        help='Path to topology.json for host discovery (legacy)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run a 10-second traffic test'
    )
    parser.add_argument(
        '--profile',
        type=str,
        choices=tuple(TrafficManager.TRAFFIC_PROFILES),
        default=None,
        help='Specific traffic profile from TrafficManager.TRAFFIC_PROFILES'
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Use unified logging when run standalone
    setup_unified_logging(__name__, log_level="debug")

    if os.geteuid() != 0:
        log.error("Must run with sudo!")
        log.error("Usage: sudo python3 traffic_generator.py [--config CONFIG] [--test] [--profile PROFILE]")
        sys.exit(1)

    args = get_args()

    tm = TrafficManager(
        topology_file=args.topology_file,
        config_path=args.config
    )

    log.info(f"Hosts: {tm.traffic_hosts}")
    log.info(f"Host IPs: {tm.hosts_ips}")
    log.info(f"Senders: {tm.senders} -> Receivers: {tm.receivers}")
    log.info(f"Traffic pairs: {len(tm.traffic_pairs)}")

    if args.test:
        log.info("Starting traffic test (10 seconds)...")
        info = tm.start_traffic(profile_name=args.profile)
        log.info(f"Profile: {info['profile_name']} ({info['profile_category']})")
        time.sleep(10)
        log.info("Stopping traffic...")
        tm.stop_traffic()
        log.info("Done")
