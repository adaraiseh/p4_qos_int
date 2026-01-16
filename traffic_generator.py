#!/usr/bin/env python3
"""
traffic_generator.py - Dynamic traffic generation for RL training episodes.

Uses P4Utils TaskClient to send iperf3 tasks to running TaskServers on mininet hosts.
Supports per-episode traffic profile selection from 6 predefined profiles.

NOTE: Must be run with sudo since mininet TaskServers run as root.

Traffic Profiles (network bottleneck: ToR-Agg at 5 Mbps):
┌──────────┬──────────┬─────────────┬─────────────┬─────────────┬───────────────┐
│ Profile  │ Category │  Voice (Q0) │  Video (Q1) │    BE (Q7)  │ Per-Sender    │
├──────────┼──────────┼─────────────┼─────────────┼─────────────┼───────────────┤
│ light_1  │ light    │ 0.22-0.34   │ 0.37-0.52   │ 0.39-0.65   │ ~1.0-1.5 Mbps │
│ light_2  │ light    │ 0.29-0.42   │ 0.47-0.63   │ 0.55-0.82   │ ~1.3-1.9 Mbps │ 10%
│ medium_1 │ medium   │ 0.38-0.52   │ 0.47-0.65   │ 0.72-1.03   │ ~1.6-2.2 Mbps │
│ medium_2 │ medium   │ 0.45-0.62   │ 0.56-0.80   │ 0.86-1.16   │ ~1.9-2.6 Mbps │ 15%
│ high_1   │ high     │ 0.48-0.70   │ 0.61-0.85   │ 0.94-1.26   │ ~2.0-2.8 Mbps │
│ high_2   │ high     │ 0.54-0.77   │ 0.68-0.97   │ 1.06-1.44   │ ~2.3-3.2 Mbps │ 35%
└──────────┴──────────┴─────────────┴─────────────┴─────────────┴───────────────┘
bursty profiles: 40%
base medium_1 then pump traffic of one the following profiles:
    'test_be_heavy_1': {0: (0.05, 0.10), 1: (0.10, 0.20), 7: (1.25, 1.75)},
    'test_be_heavy_2': {0: (0.08, 0.15), 1: (0.15, 0.25), 7: (1.50, 2.00)},
    
    # === Video-heavy scenarios (high video, low voice/BE) ===
    'test_video_heavy_1': {0: (0.08, 0.15), 1: (1.25, 1.75), 7: (0.20, 0.35)},
    'test_video_heavy_2': {0: (0.10, 0.18), 1: (1.50, 2.00), 7: (0.25, 0.40)},
    
    # === Voice-heavy scenarios (high voice, low video/BE) ===
    'test_voice_heavy_1': {0: (1.25, 1.75), 1: (0.10, 0.20), 7: (0.20, 0.35)},
    'test_voice_heavy_2': {0: (1.50, 2.00), 1: (0.15, 0.25), 7: (0.25, 0.40)},

Key features:
- Round-robin profile selection for equal distribution across episodes
- Uses unique TRAFFIC_TAG for reliable process termination
- Bash while-loops for auto-restart if iperf crashes
- Profile logged to InfluxDB (tags) and CSV for analysis
"""

import os
import sys
import json
import glob
import time
import random
import threading
import subprocess
import logging
import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from p4utils.utils.task_scheduler import Task, TaskClient

# Import helpers from network.py
from network import _traffic_dst_port, QID_TOS, ALL_QUEUES

# Import unified logging
from logging_config import setup_unified_logging

log = logging.getLogger(__name__)


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
    
    Profiles are selected round-robin for equal distribution across training episodes.
    """
    
    # Unique marker for identifying traffic processes (used by pkill)
    TRAFFIC_TAG = "__RL_TRAFFIC__"
    
    # 6 Training profiles: 2 light, 2 medium, 2 high
    # Format: {qid: (min_mbps, max_mbps)} where qid 0=Voice, 1=Video, 7=BE
    # Network: ToR-Agg bottleneck at 5 Mbps, each sender has 3 flows
    TRAFFIC_PROFILES = {
        # Light traffic (+5%)
        'light_1': {0: (0.22, 0.34), 1: (0.37, 0.52), 7: (0.39, 0.65)},
        'light_2': {0: (0.29, 0.42), 1: (0.47, 0.63), 7: (0.55, 0.82)},
        # Medium traffic (+5%)
        'medium_1': {0: (0.38, 0.52), 1: (0.47, 0.65), 7: (0.72, 1.03)},
        'medium_2': {0: (0.45, 0.62), 1: (0.56, 0.80), 7: (0.86, 1.16)},
        # High traffic (high_1 unchanged from baseline, high_2 unchanged)
        'high_1': {0: (0.48, 0.70), 1: (0.61, 0.85), 7: (0.94, 1.26)},
        'high_2': {0: (0.54, 0.77), 1: (0.68, 0.97), 7: (1.06, 1.44)},
    }
    
    # TEST profiles for production - NOT used in training
    # These provide varied workload patterns to test agent robustness
    TEST_TRAFFIC_PROFILES = {
        # === BE-heavy scenarios (high BE, low voice/video) (+25% from previous) ===
        'test_be_heavy_1': {0: (0.09, 0.20), 1: (0.20, 0.40), 7: (2.41, 3.39)},
        'test_be_heavy_2': {0: (0.16, 0.29), 1: (0.29, 0.49), 7: (2.91, 3.89)},

        # === Video-heavy scenarios (high video, low voice/BE) (+15% from previous) ===
        'test_video_heavy_1': {0: (0.13, 0.23), 1: (1.93, 2.71), 7: (0.32, 0.55)},
        'test_video_heavy_2': {0: (0.16, 0.29), 1: (2.33, 3.11), 7: (0.39, 0.62)},

        # === Voice-heavy scenarios (high voice, low video/BE) (+15% from previous) ===
        'test_voice_heavy_1': {0: (1.93, 2.71), 1: (0.16, 0.32), 7: (0.32, 0.55)},
        'test_voice_heavy_2': {0: (2.33, 3.11), 1: (0.23, 0.39), 7: (0.39, 0.62)},

        # === Minimal load (near idle) (+15% from previous) ===
        'test_idle_1': {0: (0.05, 0.07), 1: (0.06, 0.13), 7: (0.07, 0.20)},
        'test_idle_2': {0: (0.07, 0.16), 1: (0.13, 0.23), 7: (0.16, 0.32)},
    }
    
    # Bursty training profiles - step-based bursts during training
    # Format: {burst_start_min, burst_start_max, burst_duration_min, burst_duration_max, burst_profile}
    # Baseline is randomly selected from light/medium profiles
    # Burst starts at random step between burst_start_min and burst_start_max
    BURSTY_PROFILES = {
        # BE bursts - short and long variants
        'bursty_be_1': {'burst_start_min': 2, 'burst_start_max': 84, 'burst_duration_min': 5, 'burst_duration_max': 15, 'burst_profile': 'test_be_heavy_1'},
        'bursty_be_2': {'burst_start_min': 2, 'burst_start_max': 49, 'burst_duration_min': 25, 'burst_duration_max': 50, 'burst_profile': 'test_be_heavy_1'},
        # Video bursts
        'bursty_vi_1': {'burst_start_min': 2, 'burst_start_max': 84, 'burst_duration_min': 5, 'burst_duration_max': 15, 'burst_profile': 'test_video_heavy_1'},
        'bursty_vi_2': {'burst_start_min': 2, 'burst_start_max': 49, 'burst_duration_min': 25, 'burst_duration_max': 50, 'burst_profile': 'test_video_heavy_1'},
        # Voice bursts  
        'bursty_vo_1': {'burst_start_min': 2, 'burst_start_max': 84, 'burst_duration_min': 5, 'burst_duration_max': 15, 'burst_profile': 'test_voice_heavy_1'},
        'bursty_vo_2': {'burst_start_min': 2, 'burst_start_max': 49, 'burst_duration_min': 25, 'burst_duration_max': 50, 'burst_profile': 'test_voice_heavy_1'},
    }
    
    # Combined profiles for lookup (training + test + bursty metadata)
    ALL_PROFILES = {**TRAFFIC_PROFILES, **TEST_TRAFFIC_PROFILES}
    
    # Profile categories for logging
    PROFILE_CATEGORIES = {
        # Training profiles
        'light_1': 'light', 'light_2': 'light',
        'medium_1': 'medium', 'medium_2': 'medium',
        'high_1': 'high', 'high_2': 'high',
        # Bursty profiles
        'bursty_be_1': 'bursty', 'bursty_be_2': 'bursty',
        'bursty_vi_1': 'bursty', 'bursty_vi_2': 'bursty',
        'bursty_vo_1': 'bursty', 'bursty_vo_2': 'bursty',
        # Test profiles
        'test_be_heavy_1': 'test_be', 'test_be_heavy_2': 'test_be',
        'test_video_heavy_1': 'test_video', 'test_video_heavy_2': 'test_video',
        'test_voice_heavy_1': 'test_voice', 'test_voice_heavy_2': 'test_voice',
        'test_idle_1': 'test_idle', 'test_idle_2': 'test_idle',
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

    def __init__(self, topology_file: str = "/tmp/topology.json", config_path: str = None):
        """Initialize TrafficManager.

        Args:
            topology_file: Path to topology.json for host discovery (legacy)
            config_path: Path to YAML topology configuration for dynamic host/IP discovery
        """
        if os.geteuid() != 0:
            log.warning("TrafficManager: Not running as root. TaskClient may fail.")

        # Use a dedicated random generator seeded with time
        # This ensures traffic variability even when global random is seeded for reproducibility
        self._rng = random.Random(time.time())

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
        self.current_profile_name: str = ""
        self.current_profile_category: str = ""

        # iPerf log cleanup thread (CPU-efficient, runs every 120s)
        self._log_cleanup_stop = threading.Event()
        self._log_cleanup_thread = None

        # Balanced random selection with 60-episode windows
        self._profile_names = list(self.TRAFFIC_PROFILES.keys())
        self._episode_count = 0
        self._window_size = 60  # Rebalance weights every 60 episodes
        self._usage_counts = {name: 0 for name in self._profile_names}
        
        # Burst state tracking (time-based)
        self._burst_active = False
        self._burst_end_time = 0.0
        self._next_burst_time = 0.0
        
        # Step-based burst tracking (for training)
        self._step_burst_profile = None   # Current bursty profile name (e.g., 'bursty_be_1')
        self._step_burst_active = False
        self._step_burst_start_step = 0   # When the burst will start
        self._step_burst_end_step = 0     # When the burst will end
        self._step_burst_count = 0        # Track burst duration for logging
        self._step_burst_baseline = None  # Baseline profile used for this episode
        self._step_burst_baseline_loads = None  # Exact baseline loads to restore after burst

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
            event: Event type (start, stop, burst_start, burst_end, restart)
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
                    getattr(self, '_step_burst_profile', None) is not None,
                    getattr(self, '_step_burst_baseline', ''),
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

    def _get_expected_ports(self) -> Dict[str, set]:
        """Get expected iperf ports per host.

        Returns:
            Dict mapping hostname to set of expected ports
        """
        expected = {}

        # Servers: receivers need server ports
        for _, receiver, flow_id in self.traffic_pairs:
            if receiver not in expected:
                expected[receiver] = {'servers': set(), 'clients': set()}
            for qid in ALL_QUEUES:
                port = _traffic_dst_port(flow_id, qid)
                expected[receiver]['servers'].add(port)

        # Clients: senders need client connections
        for sender, _, flow_id in self.traffic_pairs:
            if sender not in expected:
                expected[sender] = {'servers': set(), 'clients': set()}
            for qid in ALL_QUEUES:
                port = _traffic_dst_port(flow_id, qid)
                expected[sender]['clients'].add(port)

        return expected

    def _check_and_restart_traffic(self):
        """Check if any iperf processes are missing and restart them.

        Performs both aggregate and per-queue checks:
        - Aggregate check: Total processes below 70% triggers restart
        - Per-queue check: Any single queue below 50% triggers restart

        This ensures Q1-specific failures are detected even if aggregate is OK.
        """
        expected = self._get_expected_ports()

        # Count expected total processes
        expected_servers = sum(len(v['servers']) for v in expected.values())
        expected_clients = sum(len(v['clients']) for v in expected.values())
        expected_total = expected_servers + expected_clients

        # Count running processes (quick pgrep count)
        try:
            result = subprocess.run(
                ['pgrep', '-c', 'iperf3'],
                capture_output=True, text=True, timeout=10
            )
            running_count = int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception as e:
            log.debug(f"[Health Monitor] pgrep error: {e}")
            running_count = 0

        # Check 1: Aggregate threshold (70%)
        threshold = int(expected_total * 0.7)
        aggregate_failed = running_count < threshold

        # Check 2: Per-queue threshold (50%) - detect queue-specific failures
        expected_per_queue = len(self.traffic_pairs) * 2  # servers + clients per queue
        per_queue_threshold = int(expected_per_queue * 0.5)
        per_queue_failed = False
        failed_queues = []

        for qid in ALL_QUEUES:
            port_pattern = f'iperf3.*-p 6[1-6][0-9]{qid}'
            try:
                result = subprocess.run(
                    ['pgrep', '-c', '-f', port_pattern],
                    capture_output=True, text=True, timeout=10
                )
                queue_count = int(result.stdout.strip()) if result.returncode == 0 else 0
            except Exception:
                queue_count = 0

            if queue_count < per_queue_threshold:
                per_queue_failed = True
                failed_queues.append((qid, queue_count, expected_per_queue))

        # Restart if either check fails
        if aggregate_failed or per_queue_failed:
            if per_queue_failed and not aggregate_failed:
                # Queue-specific failure with aggregate OK - this is the Q1 bug scenario
                log.warning(f"[Health Monitor] Per-queue failure detected (aggregate OK): "
                           f"queues {[(f'Q{q}:{c}/{e}') for q, c, e in failed_queues]}")
            elif aggregate_failed:
                log.warning(f"[Health Monitor] Aggregate failure: {running_count}/{expected_total} "
                           f"(threshold: {threshold})")

        if aggregate_failed or per_queue_failed:
            self._restart_count += 1
            log.warning(f"[Health Monitor] Only {running_count}/{expected_total} iperf processes running "
                       f"(threshold: {threshold}). Restarting traffic... (restart #{self._restart_count})")

            # Check TaskServer socket health before restart
            missing_sockets = []
            for host in self.traffic_hosts:
                socket_path = f"/tmp/{host}_socket"
                if not os.path.exists(socket_path):
                    missing_sockets.append(host)
            if missing_sockets:
                log.error(f"[Health Monitor] Missing TaskServer sockets: {missing_sockets}")

            # Mark transition for telemetry stability
            self._in_transition = True
            self._transition_start_time = time.monotonic()

            # Restart all traffic (stop then start)
            t0 = time.monotonic()
            self._stop_all_iperf()
            time.sleep(0.5)
            self._start_servers()
            time.sleep(1.0)
            self._start_clients(self._last_packet_len)
            restart_ms = (time.monotonic() - t0) * 1000

            # Verify restart success
            time.sleep(2.0)  # Wait for processes to start
            try:
                result = subprocess.run(
                    ['pgrep', '-c', 'iperf3'],
                    capture_output=True, text=True, timeout=10
                )
                new_count = int(result.stdout.strip()) if result.returncode == 0 else 0
            except Exception:
                new_count = 0

            # Log restart event with verification
            self._log_traffic_config(event="restart", extra_info={
                'reason': 'health_monitor',
                'running_count_before': running_count,
                'running_count_after': new_count,
                'expected_total': expected_total,
                'restart_count': self._restart_count,
                'restart_ms': restart_ms,
                'missing_sockets': missing_sockets
            })

            if new_count >= threshold:
                log.info(f"[Health Monitor] Restart OK: {new_count}/{expected_total} processes "
                        f"(was {running_count}, took {restart_ms:.0f}ms)")
                # Also verify per-queue status after restart
                self._log_per_queue_status()
            else:
                log.error(f"[Health Monitor] Restart FAILED: only {new_count}/{expected_total} processes "
                         f"after restart (was {running_count})")

                # If restart failed, TaskServer may be unresponsive - try restarting TaskServers
                log.warning("[Health Monitor] Attempting TaskServer restart due to failed traffic restart...")
                if self._restart_all_taskservers():
                    time.sleep(1.0)
                    self._start_servers()
                    time.sleep(1.0)
                    self._start_clients(self._last_packet_len)

                    # Verify TaskServer restart helped
                    time.sleep(2.0)
                    try:
                        result = subprocess.run(
                            ['pgrep', '-c', 'iperf3'],
                            capture_output=True, text=True, timeout=10
                        )
                        final_count = int(result.stdout.strip()) if result.returncode == 0 else 0
                    except Exception:
                        final_count = 0

                    if final_count >= threshold:
                        log.info(f"[Health Monitor] TaskServer restart SUCCESS: {final_count}/{expected_total} processes")
                    else:
                        log.error(f"[Health Monitor] TaskServer restart FAILED: still only {final_count}/{expected_total} processes")
                else:
                    log.error("[Health Monitor] TaskServer restart failed!")
        else:
            # Log health status periodically at debug level
            log.debug(f"[Health Monitor] OK: {running_count}/{expected_total} iperf processes running")

    def _log_per_queue_status(self) -> Dict[int, int]:
        """Log process counts per queue for debugging.

        Returns:
            Dict mapping qid to running process count
        """
        expected = len(self.traffic_pairs) * 2  # servers + clients per queue
        queue_counts = {}
        status_parts = []

        for qid in ALL_QUEUES:
            # Pattern matches ports ending in qid digit (6101, 6111, ... for Q1)
            port_pattern = f'iperf3.*-p 6[1-6][0-9]{qid}'
            try:
                result = subprocess.run(
                    ['pgrep', '-c', '-f', port_pattern],
                    capture_output=True, text=True, timeout=5
                )
                count = int(result.stdout.strip()) if result.returncode == 0 else 0
            except Exception:
                count = 0
            queue_counts[qid] = count
            status_parts.append(f"Q{qid}:{count}/{expected}")

        log.info(f"[Traffic] Per-queue status: {', '.join(status_parts)}")
        return queue_counts

    def _verify_per_queue(self, min_pct: float = 0.7) -> Tuple[bool, Dict[int, int]]:
        """Verify iperf processes per queue, not just aggregate.

        Args:
            min_pct: Minimum percentage of expected processes required per queue

        Returns:
            Tuple of (all_ok, counts_dict) where counts_dict[qid] = running_count
        """
        expected_per_queue = len(self.traffic_pairs) * 2  # servers + clients
        threshold_per_queue = int(expected_per_queue * min_pct)

        queue_counts = {}
        all_ok = True
        failed_queues = []

        for qid in ALL_QUEUES:
            # Pattern matches ports ending in qid digit (6101, 6111, ... for Q1)
            port_pattern = f'iperf3.*-p 6[1-6][0-9]{qid}'
            try:
                result = subprocess.run(
                    ['pgrep', '-c', '-f', port_pattern],
                    capture_output=True, text=True, timeout=10
                )
                running = int(result.stdout.strip()) if result.returncode == 0 else 0
            except Exception:
                running = 0

            queue_counts[qid] = running

            if running < threshold_per_queue:
                log.warning(f"[Traffic] Q{qid} only has {running}/{expected_per_queue} processes (need {threshold_per_queue})!")
                all_ok = False
                failed_queues.append(qid)

        if failed_queues:
            log.warning(f"[Traffic] Per-queue verification FAILED for queues: {failed_queues}")

        return all_ok, queue_counts

    def _stop_all_iperf(self):
        """Stop all iperf processes without clearing traffic state."""
        try:
            subprocess.run(['pkill', '-9', '-f', 'iperf3.*-p 6[1-6][0-9][0-9]'], capture_output=True)
            subprocess.run(['pkill', '-9', '-f', f'bash.*{self.TRAFFIC_TAG}'], capture_output=True)
        except Exception as e:
            log.debug(f"Error stopping iperf: {e}")

    def start_traffic(self, packet_len: int = 1250,
                       category_weights: Dict[str, float] = None,
                       profile_name: str = None) -> Dict[str, any]:
        """Start traffic with a selected profile.
        
        Args:
            packet_len: UDP packet length in bytes
            category_weights: Optional category weights, e.g. {'light': 0.1, 'medium': 0.2, 'high': 0.3, 'bursty': 0.4}
                             If provided, selects category first then random profile within category.
                             For 'bursty' category, returns a bursty profile name (caller should use check_step_burst).
            profile_name: Optional specific profile name (e.g., 'high_1', 'medium_2', 'bursty_be_1').
                         If provided, uses this profile directly (overrides category_weights).
            
        Returns:
            Dict with profile info for logging. For bursty profiles, also includes 'is_bursty': True
        """
        self.stop_traffic()
        time.sleep(0.3)
        self._start_log_cleanup_thread()

        # Mark traffic as transitioning - telemetry will be unstable during ramp-up
        self._in_transition = True
        self._transition_start_time = time.monotonic()

        is_bursty = False
        
        if profile_name:
            # Use specific profile by name (supports training, test, and bursty profiles)
            if profile_name in self.BURSTY_PROFILES:
                is_bursty = True
                # Select random baseline from light/medium profiles
                baseline_options = ['light_1', 'light_2', 'medium_1', 'medium_2']
                baseline_profile = self._rng.choice(baseline_options)
                self._step_burst_baseline = baseline_profile
                self.current_profile_name = profile_name
                self.current_profile_category = 'bursty'
                profile_ranges = self.ALL_PROFILES[baseline_profile]
                # Randomize baseline loads now with correlation, save them for restoration after burst
                # Use correlated randomization (same as main traffic start)
                midpoints = {qid: (rng[0] + rng[1]) / 2.0 for qid, rng in profile_ranges.items()}
                scale = self._rng.uniform(0.85, 1.15)  # ±15% variation
                self._step_burst_baseline_loads = {
                    qid: midpoints[qid] * scale for qid in profile_ranges.keys()
                }
                log.info(f"[BURSTY] Episode baseline: {baseline_profile}")
            elif profile_name in self.ALL_PROFILES:
                self.current_profile_name = profile_name
                profile_ranges = self.ALL_PROFILES[profile_name]
            else:
                log.warning(f"Unknown profile '{profile_name}', using high_1")
                profile_name = 'high_1'
                self.current_profile_name = profile_name
                profile_ranges = self.ALL_PROFILES[profile_name]
        elif category_weights:
            # Select category based on weights, then random profile in that category
            category = self._rng.choices(
                list(category_weights.keys()),
                weights=list(category_weights.values())
            )[0]
            
            if category == 'bursty':
                # Select a random bursty profile
                bursty_names = list(self.BURSTY_PROFILES.keys())
                self.current_profile_name = self._rng.choice(bursty_names)
                self.current_profile_category = 'bursty'
                is_bursty = True
                # Select random baseline from light/medium profiles
                baseline_options = ['light_1', 'light_2', 'medium_1', 'medium_2']
                baseline_profile = self._rng.choice(baseline_options)
                self._step_burst_baseline = baseline_profile
                profile_ranges = self.ALL_PROFILES[baseline_profile]
                # Randomize baseline loads now with correlation, save them for restoration after burst
                midpoints = {qid: (rng[0] + rng[1]) / 2.0 for qid, rng in profile_ranges.items()}
                scale = self._rng.uniform(0.85, 1.15)  # ±15% variation
                self._step_burst_baseline_loads = {
                    qid: midpoints[qid] * scale for qid in profile_ranges.keys()
                }
                log.info(f"[BURSTY] Episode baseline: {baseline_profile}")
            else:
                # Get all profiles in this category (training profiles only)
                profiles_in_category = [p for p in self._profile_names 
                                        if self.PROFILE_CATEGORIES.get(p) == category]
                if not profiles_in_category:
                    log.warning(f"No profiles for category '{category}', using medium_1")
                    self.current_profile_name = 'medium_1'
                else:
                    self.current_profile_name = self._rng.choice(profiles_in_category)
                profile_ranges = self.ALL_PROFILES[self.current_profile_name]
        else:
            # Balanced selection with rebalancing every 60 episodes
            self._episode_count += 1
            
            if self._episode_count % self._window_size == 1:
                self._usage_counts = {name: 0 for name in self._profile_names}
            
            max_usage = max(self._usage_counts.values()) if any(self._usage_counts.values()) else 0
            weights = [max_usage + 1 - self._usage_counts[name] for name in self._profile_names]
            
            self.current_profile_name = self._rng.choices(self._profile_names, weights=weights, k=1)[0]
            self._usage_counts[self.current_profile_name] += 1
            profile_ranges = self.ALL_PROFILES[self.current_profile_name]
        
        if not is_bursty:
            self.current_profile_category = self.PROFILE_CATEGORIES.get(self.current_profile_name, 'unknown')
        
        # Set load - use saved baseline loads for bursty profiles, otherwise randomize
        # PHASE 1.3: Use correlated randomization to reduce variance
        # Instead of independent per-queue randomization (40% total load variance),
        # use a single scale factor applied to all queues (maintains queue ratios)
        if is_bursty and self._step_burst_baseline_loads:
            self.current_load = self._step_burst_baseline_loads.copy()
        else:
            # Compute midpoint for each queue
            midpoints = {qid: (rng[0] + rng[1]) / 2.0 for qid, rng in profile_ranges.items()}
            # Compute range for scaling (average of per-queue ranges)
            avg_min = sum(rng[0] for rng in profile_ranges.values()) / len(profile_ranges)
            avg_max = sum(rng[1] for rng in profile_ranges.values()) / len(profile_ranges)
            avg_mid = (avg_min + avg_max) / 2.0

            # Pick a single scale factor for the entire profile
            # This maintains relative queue ratios while allowing load variation
            scale = self._rng.uniform(0.85, 1.15)  # ±15% variation (was ±20-40% per queue)

            self.current_load = {
                qid: midpoints[qid] * scale for qid in profile_ranges.keys()
            }
        
        log.info(f"Starting profile '{self.current_profile_name}' ({self.current_profile_category})")
        log.info(f"  Loads: Q0={self.current_load[0]:.2f}, Q1={self.current_load[1]:.2f}, Q7={self.current_load[7]:.2f} Mbps")

        # Track state for health monitoring
        self._last_packet_len = packet_len
        self._traffic_active = True

        # Start servers then clients
        self._start_servers()
        time.sleep(1.0)
        self._start_clients(packet_len)

        # Verify traffic actually started (detects unresponsive TaskServer)
        if not self._verify_traffic_started():
            log.warning("[Traffic] Startup verification failed, restarting TaskServers...")

            # Attempt TaskServer restart and retry traffic
            if self._restart_all_taskservers():
                time.sleep(1.0)
                self._start_servers()
                time.sleep(1.0)
                self._start_clients(packet_len)

                if not self._verify_traffic_started():
                    log.error("[Traffic] Startup failed even after TaskServer restart!")
                    # Log failure but continue - health monitor may recover later
                    self._log_traffic_config(event="start_failed", extra_info={
                        'packet_len': packet_len,
                        'reason': 'verification_failed_after_taskserver_restart'
                    })
            else:
                log.error("[Traffic] TaskServer restart failed!")
                self._log_traffic_config(event="start_failed", extra_info={
                    'packet_len': packet_len,
                    'reason': 'taskserver_restart_failed'
                })

        # Start health monitoring to auto-restart crashed processes
        self._start_health_monitor()

        # Log traffic configuration to CSV
        self._log_traffic_config(event="start", extra_info={'packet_len': packet_len})

        log.info("Traffic generation started")
        return {
            'profile_name': self.current_profile_name,
            'profile_category': self.current_profile_category,
            'loads': self.current_load.copy(),
            'is_bursty': is_bursty,
        }
    
    def check_burst(self, burst_interval: float = 60.0, min_duration: float = 10.0, 
                    max_duration: float = 300.0, base_profile: str = 'medium_1',
                    burst_profile: str = 'test_be_heavy_1') -> Optional[str]:
        """Check and manage periodic burst traffic.
        
        Call this regularly from the production loop. It will:
        - Start a burst, run for random duration, then wait `burst_interval` seconds
        - Repeat indefinitely
        
        Args:
            burst_interval: Seconds to wait AFTER burst ends before starting next burst
            min_duration: Minimum burst duration in seconds
            max_duration: Maximum burst duration in seconds
            base_profile: Traffic profile to use during normal operation
            burst_profile: Traffic profile to use during bursts
            
        Returns:
            Status message if state changed, None otherwise
        """
        current_time = time.time()
        
        # Initialize: schedule first burst after burst_interval
        if self._next_burst_time == 0.0:
            self._next_burst_time = current_time + burst_interval
            log.info(f"[BURST] Initialized - first burst in {burst_interval:.0f}s")
            return None
        
        # STATE: Burst is active - check if it should end
        if self._burst_active:
            if current_time >= self._burst_end_time:
                self._burst_active = False
                # Schedule next burst: wait burst_interval AFTER this burst ends
                self._next_burst_time = current_time + burst_interval
                # Restart with base profile
                self.start_traffic(profile_name=base_profile)
                log.info(f"[BURST] Next burst in {burst_interval:.0f}s")
                return f"BURST ENDED - Returning to {base_profile}"
            # Burst still running
            return None
        
        # STATE: Not in burst - check if we should start one
        if current_time >= self._next_burst_time:
            self._burst_active = True
            burst_duration = self._rng.uniform(min_duration, max_duration)
            self._burst_end_time = current_time + burst_duration
            
            # Start burst traffic
            self.start_traffic(profile_name=burst_profile)
            
            # Log burst info
            mins = int(burst_duration // 60)
            secs = int(burst_duration % 60)
            duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            return f"BURST STARTED - {burst_profile} for {duration_str}"
        
        return None
    
    def check_step_burst(self, current_step: int, bursty_profile: str = None) -> Optional[str]:
        """Check and manage step-based burst traffic for training.
        
        Call this regularly from the training loop with a bursty profile active.
        Burst starts at a random step (15-25), runs for configured duration, then ends.
        Only ONE burst per episode (no cycling).
        
        Args:
            current_step: Current training step within the episode
            bursty_profile: Name of bursty profile (e.g., 'bursty_be_1')
            
        Returns:
            Status message if state changed, None otherwise
        """
        if bursty_profile is None:
            return None
            
        cfg = self.BURSTY_PROFILES.get(bursty_profile)
        if cfg is None:
            return None
        
        # Initialize or re-initialize for new episode
        # We detect a new episode by checking if current_step == 1 (first step after reset)
        # OR if the profile changed (different bursty profile selected)
        is_new_episode = (current_step == 1) or (self._step_burst_profile != bursty_profile)
        
        if is_new_episode:
            self._step_burst_profile = bursty_profile
            self._step_burst_active = False
            self._step_burst_count = 0
            # Schedule burst start at random step between min and max
            self._step_burst_start_step = self._rng.randint(
                cfg['burst_start_min'], cfg['burst_start_max']
            )
            # Schedule burst duration
            burst_duration = self._rng.randint(
                cfg['burst_duration_min'], cfg['burst_duration_max']
            )
            self._step_burst_end_step = self._step_burst_start_step + burst_duration
            baseline = self._step_burst_baseline or 'medium_1'
            log.info(f"[BURST SCHEDULE] Profile={bursty_profile} (base={baseline}): "
                     f"burst starts at step {self._step_burst_start_step}, "
                     f"duration={burst_duration} steps, ends at step {self._step_burst_end_step}")
        
        result = None
        
        # Check if burst should end
        if self._step_burst_active and current_step >= self._step_burst_end_step:
            actual_duration = self._step_burst_count
            self._step_burst_active = False
            self._step_burst_count = 0
            # Return to baseline with original loads
            baseline = self._step_burst_baseline or 'medium_1'
            self._restore_baseline_traffic(baseline)
            # Log burst end event
            self._log_traffic_config(event="burst_end", extra_info={
                'bursty_profile': bursty_profile,
                'actual_duration_steps': actual_duration,
                'baseline_profile': baseline
            })
            log.info(f"[BURST END] Ended after {actual_duration} steps. Returning to {baseline}.")
            result = "BURST ENDED"
        
        # Check if burst should start
        elif not self._step_burst_active and current_step >= self._step_burst_start_step and current_step < self._step_burst_end_step:
            self._step_burst_active = True
            self._step_burst_count = 0
            
            # Switch to burst profile
            self.start_traffic(profile_name=cfg['burst_profile'])
            remaining = self._step_burst_end_step - current_step
            # Log burst start event (note: start_traffic already logs "start", this adds burst context)
            self._log_traffic_config(event="burst_start", extra_info={
                'bursty_profile': bursty_profile,
                'burst_profile': cfg['burst_profile'],
                'duration_steps': remaining,
                'current_step': current_step
            })
            log.info(f"[BURST START] Profile={cfg['burst_profile']}, "
                     f"duration={remaining} steps (ends at step {self._step_burst_end_step})")
            result = f"BURST STARTED: {cfg['burst_profile']} for {remaining} steps"
        
        # Track burst duration
        if self._step_burst_active:
            self._step_burst_count += 1
        
        return result
    
    def _restore_baseline_traffic(self, baseline_profile: str):
        """Restore baseline traffic with exact saved loads (no re-randomization).

        Used when returning from a burst to maintain the same baseline load
        that was active at episode start.

        PHASE 3.1 TODO: Current implementation uses hard reset (stop+start traffic)
        which creates discontinuous traffic change. Future improvement: implement
        gradual ramp-down over 5-10 steps by updating iperf3 bandwidth without restart.
        This requires either:
        1. Using iperf3 TCP control connection to update bandwidth
        2. Or adding burst_active flag to state (requires retraining)
        """
        self.stop_traffic()
        time.sleep(0.3)

        # Mark traffic as transitioning - telemetry will be unstable during ramp-up
        self._in_transition = True
        self._transition_start_time = time.monotonic()

        # Restore exact loads saved at episode start
        if self._step_burst_baseline_loads:
            self.current_load = self._step_burst_baseline_loads.copy()
        else:
            # Fallback: randomize if no saved loads (use correlated randomization)
            profile_ranges = self.ALL_PROFILES.get(baseline_profile, self.ALL_PROFILES['medium_1'])
            midpoints = {qid: (rng[0] + rng[1]) / 2.0 for qid, rng in profile_ranges.items()}
            scale = self._rng.uniform(0.85, 1.15)  # ±15% variation
            self.current_load = {
                qid: midpoints[qid] * scale for qid in profile_ranges.keys()
            }
        
        self.current_profile_name = baseline_profile
        self.current_profile_category = self.PROFILE_CATEGORIES.get(baseline_profile, 'medium')
        
        log.info(f"Restoring profile '{baseline_profile}' ({self.current_profile_category})")
        log.info(f"  Loads: Q0={self.current_load[0]:.2f}, Q1={self.current_load[1]:.2f}, Q7={self.current_load[7]:.2f} Mbps")

        # Track state for health monitoring
        self._last_packet_len = 1250
        self._traffic_active = True

        self._start_servers()
        time.sleep(1.0)
        self._start_clients(packet_len=1250)

        # Verify traffic actually started (matches start_traffic behavior)
        if not self._verify_traffic_started():
            log.warning("[Traffic] Baseline restore verification failed, restarting TaskServers...")

            if self._restart_all_taskservers():
                time.sleep(1.0)
                self._start_servers()
                time.sleep(1.0)
                self._start_clients(packet_len=1250)

                if not self._verify_traffic_started():
                    log.error("[Traffic] Baseline restore failed even after TaskServer restart!")
            else:
                log.error("[Traffic] TaskServer restart failed during baseline restore!")

        # Start health monitoring (was missing - caused step 474 failure)
        self._start_health_monitor()

        # Wait for traffic to stabilize before returning
        # This ensures InfluxDB has fresh data when RL agent queries after burst transitions
        stabilization_wait = 3.0
        log.info(f"[Traffic] Waiting {stabilization_wait}s for traffic to stabilize...")
        time.sleep(stabilization_wait)

        log.info("Traffic generation started")
    
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
                bw = self.current_load.get(qid, 0.2)
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

    def _verify_traffic_started(self, timeout: float = 8.0, min_pct: float = 0.7) -> bool:
        """Verify iperf processes actually started after traffic commands sent.

        This detects when TaskServer becomes unresponsive (tasks are received but
        never executed). If verification fails, caller should restart TaskServers.

        Also performs per-queue verification to detect queue-specific failures
        (e.g., Q1 failing while Q0/Q7 succeed).

        Args:
            timeout: Max time to wait for processes to appear
            min_pct: Minimum percentage of expected processes required (0.0-1.0)

        Returns:
            True if sufficient processes started, False otherwise
        """
        expected = len(self.traffic_pairs) * len(ALL_QUEUES) * 2  # servers + clients
        threshold = int(expected * min_pct)

        start_time = time.monotonic()
        last_count = 0

        while time.monotonic() - start_time < timeout:
            try:
                result = subprocess.run(
                    ['pgrep', '-c', 'iperf3'],
                    capture_output=True, text=True, timeout=10
                )
                running = int(result.stdout.strip()) if result.returncode == 0 else 0

                if running >= threshold:
                    log.info(f"[Traffic] Verified {running}/{expected} processes started")
                    # Also log per-queue status for diagnostics
                    self._log_per_queue_status()
                    # Check per-queue verification (warn but don't fail aggregate check)
                    per_queue_ok, queue_counts = self._verify_per_queue(min_pct)
                    if not per_queue_ok:
                        log.warning(f"[Traffic] Per-queue verification failed despite aggregate OK - some queues may have issues")
                    return True

                if running != last_count:
                    log.debug(f"[Traffic] Waiting for processes: {running}/{expected} (need {threshold})")
                    last_count = running

                time.sleep(0.5)
            except Exception as e:
                log.warning(f"[Traffic] Verification error: {e}")
                time.sleep(1.0)  # Longer sleep on error to let system recover

        # Log per-queue status on failure for debugging
        self._log_per_queue_status()
        log.error(f"[Traffic] Only {last_count}/{expected} processes after {timeout}s - startup failed!")
        return False

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
        default=None,
        help='Specific traffic profile to use (e.g., high_1, medium_2)'
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
