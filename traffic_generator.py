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
from typing import Dict, List, Tuple, Optional

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
    
    # 6 Training profiles: 2 light, 2 medium, 2 high
    # Format: {qid: (min_mbps, max_mbps)} where qid 0=Voice, 1=Video, 7=BE
    # Network: ToR-Agg bottleneck at 5 Mbps, each sender has 3 flows
    TRAFFIC_PROFILES = {
        # Light traffic (~10% increase)
        'light_1': {0: (0.18, 0.28), 1: (0.20, 0.35), 7: (0.32, 0.54)},
        'light_2': {0: (0.24, 0.35), 1: (0.30, 0.46), 7: (0.46, 0.68)},
        # Medium traffic (~20% increase then -15% reduction = net ~2% increase over original)
        'medium_1': {0: (0.31, 0.44), 1: (0.39, 0.54), 7: (0.60, 0.85)},
        'medium_2': {0: (0.37, 0.51), 1: (0.46, 0.66), 7: (0.71, 0.97)},
        # High traffic (~20% increase)
        'high_1': {0: (0.44, 0.64), 1: (0.55, 0.78), 7: (0.88, 1.18)},
        'high_2': {0: (0.55, 0.78), 1: (0.68, 0.98), 7: (1.07, 1.46)},
    }
    
    # TEST profiles for production - NOT used in training
    # These provide varied workload patterns to test agent robustness
    TEST_TRAFFIC_PROFILES = {
        # === BE-heavy scenarios (high BE, low voice/video) ===
        'test_be_heavy_1': {0: (0.05, 0.10), 1: (0.10, 0.20), 7: (1.25, 1.75)},
        'test_be_heavy_2': {0: (0.08, 0.15), 1: (0.15, 0.25), 7: (1.50, 2.00)},
        
        # === Video-heavy scenarios (high video, low voice/BE) ===
        'test_video_heavy_1': {0: (0.08, 0.15), 1: (1.25, 1.75), 7: (0.20, 0.35)},
        'test_video_heavy_2': {0: (0.10, 0.18), 1: (1.50, 2.00), 7: (0.25, 0.40)},
        
        # === Voice-heavy scenarios (high voice, low video/BE) ===
        'test_voice_heavy_1': {0: (1.25, 1.75), 1: (0.10, 0.20), 7: (0.20, 0.35)},
        'test_voice_heavy_2': {0: (1.50, 2.00), 1: (0.15, 0.25), 7: (0.25, 0.40)},
        
        # === Minimal load (near idle) ===
        'test_idle_1': {0: (0.02, 0.05), 1: (0.03, 0.08), 7: (0.05, 0.12)},
        'test_idle_2': {0: (0.05, 0.10), 1: (0.08, 0.15), 7: (0.10, 0.20)},
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
        
        # Use a dedicated random generator seeded with time
        # This ensures traffic variability even when global random is seeded for reproducibility
        self._rng = random.Random(time.time())
        
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
                # Randomize baseline loads now, save them for restoration after burst
                self._step_burst_baseline_loads = {
                    qid: self._rng.uniform(*rng) for qid, rng in profile_ranges.items()
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
                # Start with medium_1 baseline
                profile_ranges = self.ALL_PROFILES['medium_1']
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
        if is_bursty and self._step_burst_baseline_loads:
            self.current_load = self._step_burst_baseline_loads.copy()
        else:
            self.current_load = {
                qid: self._rng.uniform(*rng) for qid, rng in profile_ranges.items()
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
            'is_bursty': is_bursty,
        }
    
    def check_burst(self, burst_interval: float = 60.0, min_duration: float = 10.0, 
                    max_duration: float = 300.0) -> Optional[str]:
        """Check and manage periodic BE burst traffic.
        
        Call this regularly from the production loop. It will:
        - Start a burst every `burst_interval` seconds
        - Each burst has a random duration between min and max
        - Restart normal traffic when burst ends
        
        Args:
            burst_interval: Seconds between burst starts (default 60)
            min_duration: Minimum burst duration in seconds (default 10)
            max_duration: Maximum burst duration in seconds (default 300 = 5 min)
            
        Returns:
            Status message if state changed, None otherwise
        """
        current_time = time.time()
        
        # Initialize next burst time on first call
        if self._next_burst_time == 0.0:
            self._next_burst_time = current_time + burst_interval
            return None
        
        # Check if burst should end
        if self._burst_active and current_time >= self._burst_end_time:
            self._burst_active = False
            self._next_burst_time = current_time + burst_interval
            # Restart with base profile (test_bursty uses medium-ish base load)
            self.start_traffic(profile_name='medium_1')
            return "BURST ENDED - Returning to normal traffic"
        
        # Check if burst should start
        if not self._burst_active and current_time >= self._next_burst_time:
            self._burst_active = True
            burst_duration = self._rng.uniform(min_duration, max_duration)
            self._burst_end_time = current_time + burst_duration
            
            # Start burst with high BE load
            self.start_traffic(profile_name='test_be_heavy_1')
            
            # Log burst info
            mins = int(burst_duration // 60)
            secs = int(burst_duration % 60)
            duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            return f"BURST STARTED - High BE traffic for {duration_str}"
        
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
            log.info(f"[BURST SCHEDULE] Profile={bursty_profile}: "
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
            log.info(f"[BURST END] Ended after {actual_duration} steps. Returning to {baseline}.")
            result = "BURST ENDED"
        
        # Check if burst should start
        elif not self._step_burst_active and current_step >= self._step_burst_start_step and current_step < self._step_burst_end_step:
            self._step_burst_active = True
            self._step_burst_count = 0
            
            # Switch to burst profile
            self.start_traffic(profile_name=cfg['burst_profile'])
            remaining = self._step_burst_end_step - current_step
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
        """
        self.stop_traffic()
        time.sleep(0.3)
        
        # Restore exact loads saved at episode start
        if self._step_burst_baseline_loads:
            self.current_load = self._step_burst_baseline_loads.copy()
        else:
            # Fallback: randomize if no saved loads
            profile_ranges = self.ALL_PROFILES.get(baseline_profile, self.ALL_PROFILES['medium_1'])
            self.current_load = {
                qid: self._rng.uniform(*rng) for qid, rng in profile_ranges.items()
            }
        
        self.current_profile_name = baseline_profile
        self.current_profile_category = self.PROFILE_CATEGORIES.get(baseline_profile, 'medium')
        
        log.info(f"Restoring profile '{baseline_profile}' ({self.current_profile_category})")
        log.info(f"  Loads: Q0={self.current_load[0]:.2f}, Q1={self.current_load[1]:.2f}, Q7={self.current_load[7]:.2f} Mbps")
        
        self._start_servers()
        time.sleep(1.0)
        self._start_clients(packet_len=1250)
        log.info("Traffic generation started")
    
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
