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
│ light_1  │ light    │ 0.05-0.10   │ 0.05-0.15   │ 0.10-0.20   │ ~0.6-1.4 Mbps │
│ light_2  │ light    │ 0.08-0.12   │ 0.10-0.18   │ 0.15-0.25   │ ~1.0-1.7 Mbps │
│ medium_1 │ medium   │ 0.12-0.20   │ 0.15-0.25   │ 0.25-0.40   │ ~1.6-2.6 Mbps │
│ medium_2 │ medium   │ 0.15-0.25   │ 0.20-0.30   │ 0.30-0.50   │ ~2.0-3.2 Mbps │
│ high_1   │ high     │ 0.25-0.35   │ 0.30-0.45   │ 0.50-0.80   │ ~3.2-4.8 Mbps │
│ high_2   │ high     │ 0.30-0.45   │ 0.35-0.55   │ 0.70-1.00   │ ~4.0-6.0 Mbps │
└──────────┴──────────┴─────────────┴─────────────┴─────────────┴───────────────┘

Key features:
- Round-robin profile selection for equal distribution across episodes
- Uses unique TRAFFIC_TAG for reliable process termination
- Bash while-loops for auto-restart if iperf crashes
- Profile logged to InfluxDB (tags) and CSV for analysis
"""

import os
import sys
import json
import time
import random
import subprocess
import logging
from typing import Dict, List, Tuple

from p4utils.utils.task_scheduler import Task, TaskClient

# Import helpers from network.py
from network import _traffic_dst_port, QID_TOS, ALL_QUEUES

log = logging.getLogger(__name__)


class TrafficManager:
    """Manages iperf3 traffic generation via P4Utils TaskClient.
    
    Uses TaskClient to send Task objects to TaskServer running on each host.
    This enables runtime traffic control without needing the Mininet net object.
    
    Profiles are selected round-robin for equal distribution across training episodes.
    """
    
    # Unique marker for identifying traffic processes (used by pkill)
    TRAFFIC_TAG = "__RL_TRAFFIC__"
    
    # 6 Traffic profiles: 2 light, 2 medium, 2 high
    # Format: {qid: (min_mbps, max_mbps)}
    # Network: ToR-Agg bottleneck at 5 Mbps, each sender has 3 flows
    TRAFFIC_PROFILES = {
        # Light traffic (~10-35% of bottleneck capacity)
        'light_1': {0: (0.05, 0.10), 1: (0.05, 0.15), 7: (0.10, 0.20)},
        'light_2': {0: (0.08, 0.12), 1: (0.10, 0.18), 7: (0.15, 0.25)},
        # Medium traffic (~30-65% of bottleneck capacity)
        'medium_1': {0: (0.12, 0.20), 1: (0.15, 0.25), 7: (0.25, 0.40)},
        'medium_2': {0: (0.15, 0.25), 1: (0.20, 0.30), 7: (0.30, 0.50)},
        # High traffic (~65-120% of bottleneck - causes congestion)
        'high_1': {0: (0.25, 0.35), 1: (0.30, 0.45), 7: (0.50, 0.80)},
        'high_2': {0: (0.30, 0.45), 1: (0.35, 0.55), 7: (0.70, 1.00)},
    }
    
    # Profile categories for logging
    PROFILE_CATEGORIES = {
        'light_1': 'light', 'light_2': 'light',
        'medium_1': 'medium', 'medium_2': 'medium',
        'high_1': 'high', 'high_2': 'high',
    }
    
    # Host IP mapping (h1-h8)
    HOSTS_IPS = [
        "0",          # dummy index 0
        "10.7.1.2",   # h1
        "10.7.2.2",   # h2
        "10.8.3.2",   # h3
        "10.8.4.2",   # h4
        "10.9.5.2",   # h5
        "10.9.6.2",   # h6
        "10.10.7.2",  # h7
        "10.10.8.2",  # h8
    ]
    
    def __init__(self, topology_file: str = "/tmp/topology.json"):
        """Initialize TrafficManager.
        
        Args:
            topology_file: Path to topology.json for host discovery
        """
        if os.geteuid() != 0:
            log.warning("TrafficManager: Not running as root. TaskClient may fail.")
        
        self.topology_file = topology_file
        
        # Discover traffic hosts (h1-h8, excluding h100+)
        self.traffic_hosts = self._discover_traffic_hosts()
        log.info(f"TrafficManager: Found traffic hosts: {self.traffic_hosts}")
        
        # Build sender/receiver pairs
        self.pods = self._build_pods()
        self.senders = [pod[0] for pod in self.pods]    # h1, h3, h5, h7
        self.receivers = [pod[1] for pod in self.pods]  # h2, h4, h6, h8
        self.traffic_pairs = self._build_traffic_pairs()
        
        log.info(f"TrafficManager: {len(self.traffic_pairs)} traffic pairs configured")
        
        # Current profile state
        self.current_load: Dict[int, float] = {}
        self.current_profile_name: str = ""
        self.current_profile_category: str = ""
        
        # Round-robin index for equal profile distribution
        self._profile_index = 0
        self._profile_names = list(self.TRAFFIC_PROFILES.keys())
        random.shuffle(self._profile_names)
    
    def _discover_traffic_hosts(self) -> List[str]:
        """Discover traffic hosts from topology (hosts with id < 100)."""
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
            log.warning(f"Failed to load topology: {e}, using default h1-h8")
            hosts = [f'h{i}' for i in range(1, 9)]
        return sorted(hosts, key=lambda x: int(x[1:]))
    
    def _build_pods(self) -> List[List[str]]:
        """Build pod structure (pairs of hosts per ToR)."""
        pods = []
        sorted_hosts = sorted(self.traffic_hosts, key=lambda x: int(x[1:]))
        for i in range(0, len(sorted_hosts), 2):
            if i + 1 < len(sorted_hosts):
                pods.append([sorted_hosts[i], sorted_hosts[i+1]])
        return pods
    
    def _build_traffic_pairs(self) -> List[Tuple[str, str, int]]:
        """Build (src, dst, flow_id) pairs - senders to receivers in OTHER pods."""
        pairs = []
        next_flow_id = 10
        for i, sender in enumerate(self.senders):
            for j, pod in enumerate(self.pods):
                if j == i:
                    continue
                pairs.append((sender, pod[1], next_flow_id))
                next_flow_id += 1
        return pairs
    
    def _host_to_ip(self, hostname: str) -> str:
        """Get IP for hostname."""
        idx = int(hostname[1:])
        return self.HOSTS_IPS[idx]
    
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
        """Stop all traffic processes using pkill."""
        log.info("Stopping all traffic processes...")
        try:
            # Kill iperf3 processes on our ports (61xx, 62xx)
            subprocess.run(['pkill', '-9', '-f', 'iperf3.*-p 6[12]'], capture_output=True)
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
        time.sleep(0.5)
    
    def start_traffic(self, packet_len: int = 1250) -> Dict[str, any]:
        """Start traffic with a randomly selected profile (round-robin).
        
        Args:
            packet_len: UDP packet length in bytes
            
        Returns:
            Dict with profile info for logging
        """
        self.stop_traffic()
        time.sleep(0.3)
        
        # Select next profile (round-robin)
        self.current_profile_name = self._profile_names[self._profile_index]
        self._profile_index = (self._profile_index + 1) % len(self._profile_names)
        self.current_profile_category = self.PROFILE_CATEGORIES[self.current_profile_name]
        
        # Randomize load within profile ranges
        profile_ranges = self.TRAFFIC_PROFILES[self.current_profile_name]
        self.current_load = {
            qid: random.uniform(*rng) for qid, rng in profile_ranges.items()
        }
        
        log.info(f"Starting profile '{self.current_profile_name}' ({self.current_profile_category})")
        log.info(f"  Loads: Q0={self.current_load[0]:.2f}, Q1={self.current_load[1]:.2f}, Q7={self.current_load[7]:.2f} Mbps")
        
        # Start servers then clients
        self._start_servers()
        time.sleep(1.0)
        self._start_clients(packet_len)
        
        log.info("Traffic generation started")
        return {
            'profile_name': self.current_profile_name,
            'profile_category': self.current_profile_category,
            'loads': self.current_load.copy(),
        }
    
    def _start_servers(self):
        """Start iperf3 servers on receiver hosts."""
        for _, receiver, flow_id in self.traffic_pairs:
            for qid in ALL_QUEUES:
                port = _traffic_dst_port(flow_id, qid)
                cmd = (
                    f"bash -lc '{self.TRAFFIC_TAG}=1; "
                    f"while true; do iperf3 -s -p {port} -i 1 "
                    f"--logfile /tmp/{receiver}_iperf3_s_{port}.log; sleep 1; done'"
                )
                self._send_task(receiver, cmd)
    
    def _start_clients(self, packet_len: int):
        """Start iperf3 clients on sender hosts."""
        for sender, receiver, flow_id in self.traffic_pairs:
            dst_ip = self._host_to_ip(receiver)
            for qid in ALL_QUEUES:
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
                self._send_task(sender, cmd, delay=0.5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if os.geteuid() != 0:
        print("ERROR: Must run with sudo!")
        print("Usage: sudo python3 traffic_generator.py [test]")
        sys.exit(1)
    
    tm = TrafficManager()
    print(f"Hosts: {tm.traffic_hosts}")
    print(f"Senders: {tm.senders} -> Receivers: {tm.receivers}")
    print(f"Traffic pairs: {len(tm.traffic_pairs)}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("\nStarting traffic test (10 seconds)...")
        info = tm.start_traffic()
        print(f"Profile: {info['profile_name']} ({info['profile_category']})")
        time.sleep(10)
        print("\nStopping traffic...")
        tm.stop_traffic()
        print("Done")
