#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_agent_4.py - Simplified DQN for per-queue QoS path optimization using P4 INT reports

Key Design Principles:
1. Single centralized DQN agent (not multi-agent) - eliminates coordination overhead
2. Frame stacking (92 features) - stacked observations + stacked one-hot actions
3. Action history as one-hot vectors - agent knows "I caused this" vs "happened naturally"
4. Clear SLA-based reward - bounded, no improvement bonus (avoids rewarding noise)
5. Focused action space (4 actions) - no-op + one change per queue
6. Prioritized Experience Replay - learn from rare important events
7. Proper episode boundaries - clear termination conditions
8. Tuned timing for 100% post-action data capture

State Composition (92 features):
- Stacked Observations: 169 metrics * 4 frames = 676 features
- Stacked Actions (one-hot): 10 * 4 frames = 40 features
- Total: 92 features

Benefits:
- Agent sees velocity/trends (is latency rising or falling?)
- Agent knows full action history via one-hot encoding
- Example: [1,0,0,0, 0,0,1,0, 0,0,0,0, 0,1,0,0] = noop→video→noop→voice
- Prevents sawtooth over-correction patterns

Author: Research Team
"""

import os
import sys
import time
import random
import logging
import argparse
import csv
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from controller import Controller

# =============================================================================
#                              LOGGING SETUP
# =============================================================================
# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Create a custom handler with immediate flushing
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[FlushingStreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# =============================================================================
#                           HYPERPARAMETERS
# =============================================================================
# Network
HIDDEN_DIM = 128        # Reduced from 256 for smaller state space
RAW_STATE_DIM = 82      # Down from 169 (removed One-Hot IDs)
STACK_SIZE = 6          # Keep at 6 to capture 4-step action delay
ACTION_DIM = 10         # No-op + 3 queues * 3 alts
# State composition: stacked observations + stacked one-hot actions
# Observations: 82 metrics * 6 frames = 492
# Actions: 10 (one-hot) * 6 frames = 60
# Total: 492 + 60 = 552
STATE_DIM = (RAW_STATE_DIM * STACK_SIZE) + (ACTION_DIM * STACK_SIZE)

# Learning
LR = 1e-4  # Increased from 1e-5 for faster convergence
GAMMA = 0.99  # Kept at 0.99 (correct for 4-step delay: gamma^4 = 0.96)
BATCH_SIZE = 64  # Increased from 32 for more stable gradients
MIN_REPLAY_SIZE = 500  # Start learning much sooner
REPLAY_CAPACITY = 50_000

# Prioritized Experience Replay
PER_ALPHA = 0.6  # Prioritization exponent
PER_BETA_START = 0.4  # Importance sampling start
PER_BETA_END = 1.0
PER_BETA_STEPS = 10_000

# Epsilon schedule
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 5_000  # Reduced from 10k for faster exploration->exploitation

# Target network - Soft updates (Polyak averaging) for smooth Q-value evolution
TAU = 0.005  # Soft update rate: target = TAU * online + (1-TAU) * target

# Environment timing - tuned for 100% post-action data capture
# Based on sync test results: first_change ~0.28s, query RTT ~0.3s
# Formula: DELAY_AFTER_ACTION >= WINDOW + SAFETY_LAG/1000 + first_change + margin
WINDOW_SECONDS = 2          # 3-second observation window
SAFETY_LAG_MS = 300         # 500ms safety lag for InfluxDB
COOLDOWN_SECONDS = 0.0      # Short cooldown for faster learning
DELAY_AFTER_ACTION = 2.5    # Wait 4s for 100% post-action data (3+0.5+0.3+margin)
DELAY_NO_ACTION = 2.5       # Low-pass filter: smooth observations during stable periods

# Episode
MAX_EPISODE_STEPS = 100
# Note: No early termination - let agent learn to maintain good state, not just fix bad state

# QoS thresholds (ms) - for reward calculation
SLA_THRESHOLDS = {
    0: 100.0,   # Voice - strictest
    1: 150.0,   # Video
    7: 200.0,   # Best-effort - most lenient
}
QIDS = (0, 1, 7)

# Normalization caps
DROP_CAP = 5.0  # drops per 100ms
UTIL_CAP = 100.0  # percentage

# Reward weights - TUNED to prevent tanh saturation
REWARD_SLA_MET_SCALE = 0.5          # Reduced from 1.0
REWARD_SLA_VIOLATED_SCALE = 0.4     # Reduced from 0.8
REWARD_DROP_PENALTY = 0.8           # Reduced from 1.5 (with sqrt compression)
REWARD_ACTION_COST = 0.50           # Increased from 0.30 to reduce network churn
# REWARD_IMPROVEMENT_BONUS removed - was causing instability by rewarding random jitter
# Frame stacking now provides velocity information instead

# Soft margin around SLA (reduces reward flip-flopping)
SLA_SOFT_MARGIN = 0.2  # 20% buffer zone - ignore measurement jitter

# =============================================================================
#                        PRIORITIZED REPLAY BUFFER
# =============================================================================
class SumTree:
    """Binary tree data structure for efficient priority sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        return self.tree[0]
    
    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using SumTree."""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.max_priority = 1.0
        self.epsilon = 1e-5
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with max priority."""
        data = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, data)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a batch with importance sampling weights."""
        indices = []
        priorities = []
        samples = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            samples.append(data)
        
        # Compute importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()  # Normalize
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            dones,
            indices,
            weights
        )
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self) -> int:
        return self.tree.n_entries


# =============================================================================
#                           DUELING DQN NETWORK
# =============================================================================
class DuelingDQN(nn.Module):
    """Dueling DQN architecture for better value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# =============================================================================
#                              DQN AGENT
# =============================================================================
class DQNAgent:
    """
    Centralized DQN agent for QoS path optimization.
    Uses Double DQN + Dueling + Prioritized Experience Replay.
    """
    
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        self.device = device
        self.action_dim = action_dim
        
        # Networks
        self.online_net = DuelingDQN(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(REPLAY_CAPACITY, PER_ALPHA)
        
        # Epsilon schedule
        self.eps = EPS_START
        self.step_count = 0
        
        # PER beta schedule
        self.beta = PER_BETA_START
        
        # Metrics
        self.losses = deque(maxlen=100)
        self.rewards = deque(maxlen=100)
        self.last_loss = None
    
    def select_action(self, state: np.ndarray, valid_mask: np.ndarray, 
                      explore: bool = True) -> int:
        """
        Select action using epsilon-greedy with action masking.
        
        Args:
            state: Current state vector
            valid_mask: Boolean mask of valid actions
            explore: Whether to use epsilon-greedy (False for evaluation)
        
        Returns:
            Selected action index
        """
        valid_actions = np.where(valid_mask)[0]
        
        if len(valid_actions) == 0:
            return 0  # Default to no-op
        
        # Epsilon-greedy exploration
        if explore and random.random() < self.eps:
            return int(np.random.choice(valid_actions))
        
        # Greedy action selection
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t).squeeze(0).cpu().numpy()
            
            # Mask invalid actions
            q_values[~valid_mask] = -np.inf
            return int(np.argmax(q_values))
    
    def update_epsilon(self):
        """Update epsilon based on step count."""
        self.step_count += 1
        progress = min(1.0, self.step_count / EPS_DECAY_STEPS)
        self.eps = EPS_END + (EPS_START - EPS_END) * (1 - progress)
        
        # Update PER beta
        beta_progress = min(1.0, self.step_count / PER_BETA_STEPS)
        self.beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * beta_progress
    
    def push_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.rewards.append(reward)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None
        
        # Sample from prioritized replay buffer
        (states, actions, rewards, next_states, dones, 
         indices, weights) = self.replay_buffer.sample(BATCH_SIZE, self.beta)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online net to select actions, target net to evaluate
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q[dones_t] = 0.0
            target_q = rewards_t + GAMMA * next_q
        
        # TD errors for priority update
        td_errors = (target_q - current_q).detach().cpu().numpy()
        
        # Weighted loss
        loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Soft update target network (Polyak averaging) - every step
        # This smooths Q-value evolution instead of sudden jumps from hard copies
        for target_param, online_param in zip(
            self.target_net.parameters(),
            self.online_net.parameters()
        ):
            target_param.data.copy_(
                TAU * online_param.data + (1.0 - TAU) * target_param.data
            )
        
        self.last_loss = loss.item()
        self.losses.append(self.last_loss)
        
        return self.last_loss
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'eps': self.eps,
            'beta': self.beta,
        }, path)
        log.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint.get('step_count', 0)
        self.eps = checkpoint.get('eps', EPS_END)
        self.beta = checkpoint.get('beta', PER_BETA_END)
        log.info(f"Model loaded from {path}")
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'eps': self.eps,
            'beta': self.beta,
            'step_count': self.step_count,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'last_loss': self.last_loss,
        }


# =============================================================================
#                           ENVIRONMENT
# =============================================================================
class QoSRoutingEnv:
    """
    Environment for QoS-aware routing optimization using P4 INT metrics.
    
    State: 552 features (6-frame stacking to capture 4-step action delay)
      Frame stacking provides velocity/trend information.
      
      Raw observation (82 features):
        - Per queue (27 features × 3 = 81):
            - Basic metrics: lat_ratio, drop_norm, util_norm (3)
            - Bottleneck: present, drop, lat (3)
            - Alternatives (3 × 6 = 18):
                - available, drop_vs_bn, lat_vs_bn, role, is_current, usage (6)
            - History: total_changes, steps_since_change, has_pending (3)
        - Global (1): max_pressure
      
    Actions: 10
      - 0: No-op
      - 1-3: Queue 0 -> Alt 0, 1, 2
      - 4-6: Queue 1 -> Alt 0, 1, 2
      - 7-9: Queue 7 -> Alt 0, 1, 2
    
    Reward:
      - SLA-based with soft margin, drop penalty, action cost
    """
    
    # Action to (queue, alt_index) mapping
    # alt_index is 0-based index into the available alternatives list
    ACTION_MAP = {
        0: None,           # No-op
        1: (0, 0), 2: (0, 1), 3: (0, 2),  # Voice alts
        4: (1, 0), 5: (1, 1), 6: (1, 2),  # Video alts
        7: (7, 0), 8: (7, 1), 9: (7, 2),  # BE alts
    }
    
    MAX_ALTS = 3
    NUM_SWITCHES = 10  # Standard topology size (a1-4, c1-2, t1-4)
    
    def __init__(self, bucket: str, token: str, org: str, url: str, verbose: bool = False):
        self.bucket = bucket
        self.org = org
        self.url = url
        
        # InfluxDB client
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=5000)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        # Controller for routing changes
        self.controller = Controller(verbose=verbose)
        
        # Initialize switch mapping for One-Hot encoding
        # We need a stable mapping of switch IDs to indices 0..N-1
        self.all_switch_ids = self.controller.get_all_switch_ids()
        # Ensure we have at least NUM_SWITCHES capacity (padding if necessary)
        # For this topology, we expect ~10 switches. 
        self.sid_to_idx = {sid: i for i, sid in enumerate(self.all_switch_ids)}
        log.info(f"Initialized switch mapping: {self.sid_to_idx}")
        
        # Global step counter (persists across episodes)
        self.global_step = 0
       
        # Episode state
        self.episode_step = 0
        self.sla_streak = 0
        self.last_action_time = 0.0
        self.last_action = 0
        
        # Cache snapshots for comparison
        self.last_snapshot = None
        
        # Frame stacking for velocity/trend detection
        # Stores last STACK_SIZE raw observation states (each 19-dim)
        self.frame_stack: deque = deque(maxlen=STACK_SIZE)
        
        # Action stacking for causality tracking
        # Stores last STACK_SIZE actions as one-hot vectors (each 4-dim)
        self.action_stack: deque = deque(maxlen=STACK_SIZE)
        
        # Preserve action history across episodes to avoid "Reset Amnesia"
        # Network switches persist, so agent should remember what it did
        self.prev_episode_actions: Optional[deque] = None
        
        # Preserve observation history across episodes to avoid "Reset Amnesia"
        # Network state persists, so agent should remember what it observed
        self.prev_episode_frames: Optional[deque] = None
        
    
    def reset(self) -> np.ndarray:
        """Reset episode and return initial stacked state.
        
        Note: Action history is preserved across episodes to avoid "Reset Amnesia".
        Network switches persist their state, so the agent needs to remember
        what actions it took previously to understand the current network state.
        """
        self.episode_step = 0
        self.sla_streak = 0
        self.last_action = 0
        self.last_action_time = time.monotonic()
        
        # Collect initial snapshot
        self.last_snapshot = self._collect_snapshot()
        raw_state = self._build_raw_state(self.last_snapshot)
        
        # Debug assertion to catch dimension mismatches early
        assert len(raw_state) == RAW_STATE_DIM, f"State dim mismatch: {len(raw_state)} != {RAW_STATE_DIM}"
        
        # Preserve observation history across episodes (fix "Reset Amnesia")
        # Network state persists, so agent should remember what it observed
        if self.prev_episode_frames is not None:
            # We are continuing from a previous episode, so keep the observation history too!
            self.frame_stack = deque(self.prev_episode_frames, maxlen=STACK_SIZE)
        else:
            # Cold start (only for the very first episode)
            self.frame_stack.clear()
            for _ in range(STACK_SIZE):
                self.frame_stack.append(raw_state.copy())
        
        # Preserve action history across episodes (fix "Reset Amnesia")
        # Network state persists, so agent should remember its previous actions
        if self.prev_episode_actions is not None:
            # Use actions from previous episode
            self.action_stack = deque(self.prev_episode_actions, maxlen=STACK_SIZE)
        else:
            # First episode: initialize with no-ops
            noop_onehot = self._action_to_onehot(0)
            self.action_stack.clear()
            for _ in range(STACK_SIZE):
                self.action_stack.append(noop_onehot.copy())
        
        return self._build_stacked_state()
    
    def _time_window(self) -> Tuple[str, str]:
        """Get time window for queries."""
        stop_dt = datetime.utcnow() - timedelta(milliseconds=SAFETY_LAG_MS)
        start_dt = stop_dt - timedelta(seconds=WINDOW_SECONDS)
        return start_dt.isoformat() + 'Z', stop_dt.isoformat() + 'Z'
    
    def _collect_snapshot(self) -> Dict[int, Dict]:
        """
        Collect metrics snapshot from InfluxDB.
        
        Returns dict: qid -> {
            'lat_p95': float,
            'drop_p95': float,
            'util_p95': float,
            'hot_src_ip': str,
            'hot_dst_ip': str,
            'alt_exists': bool,
            'bottleneck_sid': int,
        }
        """
        start, stop = self._time_window()
        
        # Initialize snapshot with defaults
        snapshot = {qid: {
            'lat_p95': SLA_THRESHOLDS[qid] * 2,  # Default: 2x SLA (bad)
            'drop_p95': DROP_CAP,
            'util_p95': UTIL_CAP,
            'hot_src_ip': None,
            'hot_dst_ip': None,
            'alt_exists': False,
            'bottleneck_sid': None,
            'bottleneck_drop': 0.0,  # Bottleneck switch drop rate
            'bottleneck_lat': 0.0,   # Bottleneck switch latency
            'bottleneck_role': 'other',  # Bottleneck switch role (tor/agg/core/other)
            'path_nodes': [],
        } for qid in QIDS}
        
        # Query aggregated metrics
        flux = f'''
        base = from(bucket:"{self.bucket}")
            |> range(start:{start}, stop:{stop})
            |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
            |> toFloat()
        
        lat_p95 = base
            |> filter(fn: (r) => r._measurement == "flow_latency")
            |> group(columns:["queue_id"])
            |> quantile(q:0.95, method:"estimate_tdigest")
            |> set(key:"_measurement", value:"lat_p95")
        
        drop_p95 = base
            |> filter(fn: (r) => r._measurement == "q_drop_rate_100ms")
            |> group(columns:["queue_id"])
            |> quantile(q:0.95, method:"estimate_tdigest")
            |> set(key:"_measurement", value:"drop_p95")
        
        util_p95 = base
            |> filter(fn: (r) => r._measurement == "tx_utilization")
            |> group(columns:["queue_id"])
            |> quantile(q:0.95, method:"estimate_tdigest")
            |> set(key:"_measurement", value:"util_p95")
        
        union(tables:[lat_p95, drop_p95, util_p95])
        '''
        
        try:
            tables = self.query_api.query(org=self.org, query=flux)
            for table in tables or []:
                for record in table.records:
                    try:
                        qid = int(record.values.get('queue_id', -1))
                        if qid not in snapshot:
                            continue
                        measurement = record.get_measurement()
                        value = record.get_value()
                        if value is not None:
                            snapshot[qid][measurement] = float(value)
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            log.warning(f"Failed to query metrics: {e}")
        
        # 2. Get Path and Bottleneck Info
        # Also identify ALL relevant switches (bottlenecks + all alts) to query in one go
        switches_to_query = set()
        
        for qid in QIDS:
            # Find hottest demand
            hot = self._get_hottest_demand(qid)
            if not hot:
                log.info(f"[Snapshot] Queue {qid}: No hot demand found, skipping bottleneck detection")
                continue
            
            src_ip, dst_ip = hot
            log.info(f"[Snapshot] Queue {qid}: hot_demand=({src_ip}, {dst_ip})")
            snapshot[qid]['hot_src_ip'] = src_ip
            snapshot[qid]['hot_dst_ip'] = dst_ip
            
            # Get current path
            path = self.controller.get_path_by_ips(src_ip, dst_ip)
            if not path:
                log.info(f"[Snapshot] Queue {qid}: No path found for ({src_ip}, {dst_ip})")
                continue
            
            snapshot[qid]['path_nodes'] = list(path)
            
            # Local identification of bottleneck (based on previous knowledge? No, we don't have update stats yet)
            # Actually, to identify the bottleneck, we MUST query path metrics first.
            # So this has to be a two-step process:
            # Step A: Get path nodes -> Query their metrics -> Find bottleneck
            # Step B: Get alternatives for bottleneck -> Query Alt metrics
            
            # --- Step A: Path Metrics ---
            sw_names = [n for n in path if isinstance(n, str) and n[0] in ('t', 'a', 'c')]
            sw_ids = [self.controller.switch_name_to_id.get(n) for n in sw_names]
            sw_ids = [int(s) for s in sw_ids if s is not None]
            
            if not sw_ids:
                continue
                
            # Query just these path switches first to find bottleneck
            path_metrics = self._query_switch_metrics(sw_ids)
            
            # Identify Bottleneck (SKIP ToR switches - they have no alternatives)
            best_sid, best_score = None, -1.0
            for sid in sw_ids:
                # Skip ToR switches - they're at edge and have no alt paths
                role = self.controller._role_of_sid(sid)
                if role == 'tor':
                    continue
                
                r = path_metrics.get(sid, {'drop': 0, 'lat': 0})
                drop_norm = min(r['drop'], DROP_CAP) / DROP_CAP
                lat_norm = min(r['lat'], SLA_THRESHOLDS[qid]) / SLA_THRESHOLDS[qid]
                score = 0.6 * drop_norm + 0.4 * lat_norm
                if score > best_score:
                    best_sid, best_score = sid, score
            
            snapshot[qid]['bottleneck_sid'] = best_sid
            
            if best_sid is not None:
                # Store bottleneck stats
                bm = path_metrics.get(best_sid, {'drop': 0, 'lat': 0})
                snapshot[qid]['bottleneck_drop'] = bm['drop']
                snapshot[qid]['bottleneck_lat'] = bm['lat']
                snapshot[qid]['bottleneck_role'] = self.controller._role_of_sid(best_sid)
                
                # --- Step B: Alternatives ---
                # Get alternatives for this bottleneck
                alts = self.controller.find_all_alternates(best_sid, path)
                log.info(f"[Snapshot] Queue {qid}: bottleneck={best_sid}, role={snapshot[qid]['bottleneck_role']}, alternatives={alts}")
                
                # Filter alts (ignore if name not known)
                valid_alts = []
                for alt_name in alts:
                    alt_sid = self.controller.switch_name_to_id.get(alt_name)
                    if alt_sid is not None:
                        valid_alts.append((alt_name, int(alt_sid)))
                        switches_to_query.add(int(alt_sid))
                
                snapshot[qid]['_candidate_alts'] = valid_alts
        
        # 3. Query Alternatives Metrics (Bulk)
        if switches_to_query:
            alt_metrics = self._query_switch_metrics(list(switches_to_query))
            
            # Fill alternative stats
            for qid in QIDS:
                candidates = snapshot[qid].get('_candidate_alts', [])
                final_alts = []
                for name, sid in candidates:
                    m = alt_metrics.get(sid, {'drop': 0, 'lat': 0})
                    final_alts.append({
                        'name': name,
                        'drop': m['drop'],
                        'lat': m['lat']
                    })
                snapshot[qid]['alternatives'] = final_alts
                snapshot[qid]['alt_exists'] = bool(final_alts)
                
        return snapshot

    def _query_switch_metrics(self, sw_ids: List[int]) -> Dict[int, Dict]:
        """Query drop and latency for a list of switch IDs."""
        if not sw_ids:
            return {}
            
        sid_filter = " or ".join([f'r.switch_id == "{sid}"' for sid in sw_ids])
        start, stop = self._time_window()
        
        flux = f'''
        from(bucket:"{self.bucket}")
            |> range(start:{start}, stop:{stop})
            |> filter(fn: (r) => {sid_filter})
            |> filter(fn: (r) => r._measurement == "q_drop_rate_100ms" or r._measurement == "switch_latency")
            |> toFloat()
            |> group(columns:["switch_id", "_measurement"])
            |> max(column:"_value")
            |> group(columns:["switch_id"])
            |> pivot(rowKey:["switch_id"], columnKey:["_measurement"], valueColumn:"_value")
        '''
        
        results = {}
        try:
            tables = self.query_api.query(org=self.org, query=flux)
            for table in tables or []:
                for record in table.records:
                    sid = int(record.values.get('switch_id', 0) or 0)
                    results[sid] = {
                        'drop': float(record.values.get('q_drop_rate_100ms', 0) or 0),
                        'lat': float(record.values.get('switch_latency', 0) or 0),
                    }
        except Exception as e:
            log.debug(f"Failed to query switch metrics: {e}")
            
        return results
    
    def _get_hottest_demand(self, qid: int) -> Optional[Tuple[str, str]]:
        """Get the demand with highest latency for a queue."""
        start, stop = self._time_window()
        flux = f'''
        from(bucket:"{self.bucket}")
            |> range(start:{start}, stop:{stop})
            |> filter(fn: (r) => r._measurement == "flow_latency" and r.queue_id == "{qid}")
            |> toFloat()
            |> group(columns:["src_ip", "dst_ip"])
            |> mean(column:"_value")
            |> group()
            |> sort(columns:["_value"], desc:true)
            |> limit(n:1)
        '''
        try:
            tables = self.query_api.query(org=self.org, query=flux)
            for table in tables or []:
                for record in table.records:
                    src = record.values.get('src_ip')
                    dst = record.values.get('dst_ip')
                    if src and dst:
                        return str(src), str(dst)
        except Exception as e:
            log.debug(f"Failed to get hottest demand for qid={qid}: {e}")
        return None
    
    def _fill_path_info(self, qid: int, snap_q: Dict):
        """Fill path information including bottleneck and alternate availability."""
        src_ip = snap_q.get('hot_src_ip')
        dst_ip = snap_q.get('hot_dst_ip')
        if not (src_ip and dst_ip):
            return
        
        path = self.controller.get_path_by_ips(src_ip, dst_ip)
        if not path:
            return
        
        snap_q['path_nodes'] = list(path)
        
        # Get switch IDs on path
        sw_names = [n for n in path if isinstance(n, str) and n[0] in ('t', 'a', 'c')]
        sw_ids = [self.controller.switch_name_to_id.get(n) for n in sw_names]
        sw_ids = [int(s) for s in sw_ids if s is not None]
        
        if not sw_ids:
            return
        
        # Query per-switch metrics to find bottleneck
        sid_filter = " or ".join([f'r.switch_id == "{sid}"' for sid in sw_ids])
        start, stop = self._time_window()
        
        flux = f'''
        from(bucket:"{self.bucket}")
            |> range(start:{start}, stop:{stop})
            |> filter(fn: (r) => r.queue_id == "{qid}" and ({sid_filter}))
            |> filter(fn: (r) => r._measurement == "q_drop_rate_100ms" or r._measurement == "switch_latency")
            |> toFloat()
            |> group(columns:["switch_id", "_measurement"])
            |> max(column:"_value")
            |> group(columns:["switch_id"])
            |> pivot(rowKey:["switch_id"], columnKey:["_measurement"], valueColumn:"_value")
        '''
        
        rows = {}
        try:
            tables = self.query_api.query(org=self.org, query=flux)
            for table in tables or []:
                for record in table.records:
                    sid = int(record.values.get('switch_id', 0) or 0)
                    if sid not in sw_ids:
                        continue
                    rows[sid] = {
                        'drop': float(record.values.get('q_drop_rate_100ms', 0) or 0),
                        'lat': float(record.values.get('switch_latency', 0) or 0),
                    }
        except Exception as e:
            log.debug(f"Failed to query path metrics: {e}")
        
        # Find bottleneck (highest combined score)
        best_sid, best_score = None, -1.0
        for sid in sw_ids:
            r = rows.get(sid, {'drop': 0, 'lat': 0})
            drop_norm = min(r['drop'], DROP_CAP) / DROP_CAP
            lat_norm = min(r['lat'], SLA_THRESHOLDS[qid]) / SLA_THRESHOLDS[qid]
            score = 0.6 * drop_norm + 0.4 * lat_norm
            if score > best_score:
                best_sid, best_score = sid, score
        
        snap_q['bottleneck_sid'] = best_sid
        
        # Store bottleneck metrics for state representation
        if best_sid is not None:
            bn_metrics = rows.get(best_sid, {'drop': 0, 'lat': 0})
            snap_q['bottleneck_drop'] = bn_metrics.get('drop', 0)
            snap_q['bottleneck_lat'] = bn_metrics.get('lat', 0)
            snap_q['bottleneck_role'] = self.controller.switch_id_role.get(best_sid, 'other')
        
        # Check if alternate path exists
        if best_sid is not None:
            snap_q['alt_exists'] = self.controller.has_alternate_for_worst(best_sid, path)
    
    def _build_raw_state(self, snapshot: Dict[int, Dict]) -> np.ndarray:
        """
        Build 82-dimensional raw observation vector with relative metrics encoding.
        (Actions are stacked separately as one-hot vectors)
        
        Layout per queue (27 features):
          [0-2]: Basic metrics (lat_ratio, drop_norm, util_norm)
          [3-5]: Bottleneck info (present, drop, lat)
          [6-23]: Alternatives 3 × 6 = 18 (available, drop_vs_bn, lat_vs_bn, role, is_current, usage)
          [24-26]: History (total_changes, steps_since_change, has_pending)
        
        Global (1): Max pressure
        
        Total: 3×27 + 1 = 82 features
        """
        state = np.zeros(RAW_STATE_DIM, dtype=np.float32)
        
        # Role encoding: tor=0, agg=0.5, core=1.0
        role_map = {'tor': 0.0, 'agg': 0.5, 'core': 1.0, 'other': 0.0}
        
        pressures = []
        idx = 0
        
        for qid in QIDS:
            q = snapshot[qid]
            sla = SLA_THRESHOLDS[qid]
            
            # --- 1. Basic Metrics (3) ---
            # Use LINEAR scaling (not log) for consistency with reward
            lat_ratio = q['lat_p95'] / sla
            drop_norm = min(q['drop_p95'], DROP_CAP) / DROP_CAP
            util_norm = min(q['util_p95'], UTIL_CAP) / UTIL_CAP
            
            state[idx] = lat_ratio
            state[idx+1] = drop_norm
            state[idx+2] = util_norm
            idx += 3
            
            # --- 2. Bottleneck Info (3) ---
            bn_sid = q.get('bottleneck_sid')
            if bn_sid is not None:
                state[idx] = 1.0  # Present
                bn_drop = min(q.get('bottleneck_drop', 0), DROP_CAP) / DROP_CAP
                bn_lat = q.get('bottleneck_lat', 0) / sla
                state[idx+1] = bn_drop
                state[idx+2] = bn_lat
            else:
                # No bottleneck identified
                state[idx] = 0.0
                bn_drop = 0.0
                bn_lat = 0.0
            
            idx += 3
            
            # --- 3. Alternatives (3 × 6 = 18) ---
            alts = q.get('alternatives', [])
            path = q.get('path_nodes', [])
            
            # Normalize usage counts for this queue
            max_usage = max(self.controller.get_usage_count(a['name']) for a in alts) if alts else 1
            if max_usage == 0:
                max_usage = 1
            
            for i in range(self.MAX_ALTS):
                if i < len(alts):
                    alt = alts[i]
                    alt_name = alt['name']
                    
                    # Available
                    state[idx] = 1.0
                    
                    # Relative metrics (vs bottleneck)
                    alt_drop = min(alt.get('drop', 0), DROP_CAP) / DROP_CAP
                    alt_lat = alt.get('lat', 0) / sla
                    state[idx+1] = alt_drop - bn_drop  # Negative = better than bottleneck
                    state[idx+2] = alt_lat - bn_lat
                    
                    # Role (topology info)
                    alt_sid = self.controller.switch_name_to_id.get(alt_name)
                    alt_role = self.controller._role_of_sid(alt_sid) if alt_sid else 'other'
                    state[idx+3] = role_map.get(alt_role, 0.0)
                    
                    # Is on current path?
                    state[idx+4] = 1.0 if alt_name in path else 0.0
                    
                    # Usage count (normalized)
                    usage = self.controller.get_usage_count(alt_name)
                    state[idx+5] = usage / max_usage
                else:
                    # No alternative at this index
                    state[idx:idx+6] = 0.0
                
                idx += 6
            
            # --- 4. History (3) ---
            total_changes, steps_since = self.controller.get_queue_history(qid, self.global_step)
            has_pending = 1.0 if self.controller.has_pending_change_for_qid(qid) else 0.0
            
            # Normalize change counts (soft cap at 100)
            state[idx] = min(total_changes, 100) / 100.0
            # Normalize steps since change (soft cap at 1000)
            state[idx+1] = min(steps_since, 1000) / 1000.0
            state[idx+2] = has_pending
            idx += 3
            
            # Track pressure for global feature
            pressure = 0.5 * lat_ratio + 0.3 * drop_norm + 0.2 * util_norm
            pressures.append(pressure)
        
        # Global max pressure
        state[idx] = max(pressures) if pressures else 0.0
        
        return state
    
    def _action_to_onehot(self, action: int) -> np.ndarray:
        """Convert action index to one-hot vector."""
        onehot = np.zeros(ACTION_DIM, dtype=np.float32)
        onehot[action] = 1.0
        return onehot
    
    def _build_stacked_state(self) -> np.ndarray:
        """
        Build stacked state from observation stack + action stack.
        
        Returns:
            552-dimensional state vector:
              - [0-491]: Stacked observations (82 * 6 = 492 features)
              - [492-551]: Stacked one-hot actions (10 * 6 = 60 features)
            
        Layout: [obs_t-3, obs_t-2, obs_t-1, obs_t, act_t-3, act_t-2, act_t-1, act_t]
            
        This allows the agent to see:
        - Trends: If latency is rising or falling (from observation history)
        - Causality: Full action history as one-hot vectors
        - Example: If action_stack = [[1,0,0,0], [0,0,1,0], [1,0,0,0], [0,1,0,0]]
                   means: noop -> video -> noop -> voice
        """
        # Concatenate observations: oldest first, newest last
        stacked_obs = np.concatenate(list(self.frame_stack), axis=0)
        
        # Concatenate one-hot actions: oldest first, newest last
        stacked_actions = np.concatenate(list(self.action_stack), axis=0)
        
        # Combine: [76 obs features] + [16 action features] = 92 total
        return np.concatenate([stacked_obs, stacked_actions], axis=0)
    
    def _get_valid_actions(self, snapshot: Dict[int, Dict]) -> np.ndarray:
        """
        Get mask of valid actions based on available alternatives.
        """
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[0] = True  # No-op always valid
        
        now = time.monotonic()
        in_cooldown = (now - self.last_action_time) < COOLDOWN_SECONDS
        
        if not in_cooldown:
            for action, mapping in self.ACTION_MAP.items():
                if mapping is None:  # Skip no-op
                    continue
                qid, alt_idx = mapping
                
                # Check if this queue has enough alternatives
                q = snapshot[qid]
                if q.get('bottleneck_sid') is not None:
                    alts = q.get('alternatives', [])
                    if alt_idx < len(alts):
                        mask[action] = True
        
        return mask
    
    def _compute_reward(self, snapshot: Dict[int, Dict], action: int) -> Tuple[float, Dict]:
        """
        Compute reward based on SLA compliance with improved stability.
        
        Key features:
        1. Soft margin around SLA to reduce flip-flopping
        2. Higher drop penalty (drops are most actionable)
        3. Tanh compression to bound extreme values smoothly [-2.5, +2.5]
        4. Small action cost to discourage unnecessary changes
        
        Returns:
            (reward, info_dict)
        """
        raw_reward = 0.0
        info = {'sla_met': [], 'sla_violated': [], 'per_queue': {}}
        
        for qid in QIDS:
            q = snapshot[qid]
            sla = SLA_THRESHOLDS[qid]
            lat = q['lat_p95']
            
            # Calculate ratio with soft margin
            # SLA_SOFT_MARGIN=0.2 means 80-120% of SLA is "neutral zone"
            ratio = lat / sla
            margin_low = 1.0 - SLA_SOFT_MARGIN   # 0.8
            margin_high = 1.0 + SLA_SOFT_MARGIN  # 1.2
            
            if ratio <= margin_low:
                # Clearly under SLA: positive reward
                # Use sqrt for diminishing returns (don't over-reward very low latency)
                headroom = margin_low - ratio  # How much below margin
                component = REWARD_SLA_MET_SCALE * np.sqrt(headroom / margin_low)
                info['sla_met'].append(qid)
            elif ratio <= margin_high:
                # Within margin: small neutral reward (avoid flip-flopping)
                # Linear interpolation from +0.1 to -0.1
                t = (ratio - margin_low) / (margin_high - margin_low)  # 0 to 1
                component = 0.1 * (1.0 - 2.0 * t)  # +0.1 to -0.1
                info['sla_met'].append(qid)  # Still counts as met
            else:
                # Above margin: penalty
                # Use tanh to compress extreme violations smoothly
                excess = ratio - margin_high  # How much above margin
                # tanh(x) saturates at ~1 for x>2, so max penalty ~1.5
                component = -REWARD_SLA_VIOLATED_SCALE * np.tanh(excess)
                info['sla_violated'].append(qid)
            
            raw_reward += component
            
            # 2. Drop Penalty - sqrt compression for stability
            # Drops are most punishing but sqrt reduces spike sensitivity
            # sqrt(x): 0.25→0.5, 0.5→0.71, 1.0→1.0 (compresses low-mid range)
            drop_raw = min(q['drop_p95'], DROP_CAP) / DROP_CAP
            drop_penalty = REWARD_DROP_PENALTY * np.sqrt(drop_raw)
            raw_reward -= drop_penalty
            
            # Store per-queue info for debugging
            info['per_queue'][qid] = {
                'lat': lat,
                'ratio': ratio,
                'component': component,
                'drop_penalty': drop_penalty
            }
        
        # 3. Action Cost (reduced - don't discourage necessary actions)
        if action != 0:
            raw_reward -= REWARD_ACTION_COST
        
        # 4. Stability: Symmetric soft clipping with tanh
        # Maps to [-2.5, +2.5] range - symmetric to reduce bias
        reward = 2.5 * np.tanh(raw_reward / 2.5)
        
        info['raw_reward'] = raw_reward  # Pre-clipping for debugging
        
        return reward, info
    
    def _apply_action(self, action: int, snapshot: Dict[int, Dict]) -> Tuple[bool, Optional[str]]:
        """
        Apply routing action based on explicit alternative selection.
        """
        if action == 0:
            return False, None  # No-op
        
        # Parse action from map
        mapping = self.ACTION_MAP.get(action)
        if not mapping:
            return False, None
            
        qid, alt_idx = mapping
        
        q = snapshot[qid]
        src_ip = q.get('hot_src_ip')
        dst_ip = q.get('hot_dst_ip')
        bottleneck_sid = q.get('bottleneck_sid')
        alts = q.get('alternatives', [])
        
        if not (src_ip and dst_ip and bottleneck_sid):
            log.warning(f"Cannot apply action {action} (q={qid}): missing path info")
            return False, None
            
        # Check if requested alternative index exists
        if alt_idx >= len(alts):
            log.warning(f"Cannot apply action {action}: alt index {alt_idx} out of range ({len(alts)} avail)")
            return False, None
            
        target_alt = alts[alt_idx]
        alt_name = target_alt['name']
        
        ok, msg = self.controller.reroute_one_demand_symmetric(
            src_ip=src_ip, dst_ip=dst_ip, qid=qid,
            worst_switch_id=int(bottleneck_sid), alt_switch_name=alt_name
        )
        
        if ok:
            log.info(f"[ACTION {action}] Rerouted qid={qid} to Alt {alt_idx} ({alt_name}) [bn={bottleneck_sid}]")
            # Track usage and history
            self.controller.track_usage(alt_name)
            self.controller.record_queue_change(qid, self.global_step)
        else:
            log.warning(f"[ACTION {action}] Reroute failed: {msg}")
        
        return ok, alt_name
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action index (0-3)
        
        Returns:
            (next_state, reward, done, info)
            next_state is a 716-dim stacked state
        """
        # Increment global step counter (persists across episodes)
        self.global_step += 1
        self.episode_step += 1
        
        # Get current snapshot
        current_snapshot = self.last_snapshot
        
        # Apply action
        action_applied, alt_used = self._apply_action(action, current_snapshot)
        
        # Update last action tracking
        if action != 0:
            self.last_action_time = time.monotonic()
        self.last_action = action
        
        # Wait for network to settle
        delay = DELAY_AFTER_ACTION if action_applied else DELAY_NO_ACTION
        time.sleep(delay)
        
        # Collect new snapshot
        next_snapshot = self._collect_snapshot()
        
        # Compute reward
        reward, info = self._compute_reward(next_snapshot, action)
        
        # Check episode termination
        all_sla_met = len(info['sla_met']) == len(QIDS)
        if all_sla_met:
            self.sla_streak += 1
        else:
            self.sla_streak = 0
        
        # Episode ends only at MAX_EPISODE_STEPS - no early termination
        # This lets agent learn "maintenance" (sustaining good state), not just "fixing"
        done = self.episode_step >= MAX_EPISODE_STEPS
        
        info['episode_step'] = self.episode_step
        info['sla_streak'] = self.sla_streak
        info['action_applied'] = action_applied
        if alt_used is not None:
            info['alt_used'] = alt_used
        info['all_sla_met'] = all_sla_met
        
        # Build raw observation and add to frame stack
        raw_state = self._build_raw_state(next_snapshot)
        self.frame_stack.append(raw_state)
        
        # Convert action to one-hot and add to action stack
        action_onehot = self._action_to_onehot(action)
        self.action_stack.append(action_onehot)
        
        # Build stacked state (92-dim: 76 obs + 16 actions)
        next_state = self._build_stacked_state()
        self.last_snapshot = next_snapshot
        
        # Save action and observation history for next episode (fix "Reset Amnesia")
        if done:
            self.prev_episode_actions = deque(self.action_stack, maxlen=STACK_SIZE)
            self.prev_episode_frames = deque(self.frame_stack, maxlen=STACK_SIZE)
        
        return next_state, reward, done, info
    
    def get_valid_actions(self) -> np.ndarray:
        """Get valid action mask for current state."""
        return self._get_valid_actions(self.last_snapshot)
    
    def write_training_metrics(self, step: int, agent_stats: Dict, 
                                reward: float, action: int, info: Dict):
        """
        Write training metrics to InfluxDB for Grafana monitoring.
        
        Metrics logged:
        - step: Training step number
        - action: Action taken (0=noop, 1=voice, 2=video, 3=best)
        - reward: Actual reward (tanh-clipped to [-2.5, +2.5])
        - raw_reward: Pre-clipping reward for debugging
        - eps: Epsilon (exploration rate)
        - avg_loss: Average DQN loss over last 100 steps
        - avg_reward_100: Rolling average reward over last 100 steps
        - buffer_size: Replay buffer size
        - episode_step: Step within current episode
        - sla_met_count: Number of queues meeting SLA (0-3)
        - sla_streak: Consecutive steps with all SLAs met
        - alt_used: Which alternate switch was used (if action taken)
        """
        try:
            p = (
                Point("rl_training_v4")
                .field("step", int(step))
                .field("action", int(action))
                .field("reward", float(reward))
                .field("eps", float(agent_stats['eps']))
                .field("avg_loss", float(agent_stats['avg_loss']))
                .field("avg_reward_100", float(agent_stats['avg_reward']))
                .field("buffer_size", int(agent_stats['buffer_size']))
                .field("episode_step", int(info.get('episode_step', 0)))
                .field("sla_met_count", len(info.get('sla_met', [])))
                .field("sla_streak", int(info.get('sla_streak', 0)))
                .time(datetime.utcnow())
            )
            
            # Optional fields
            if 'raw_reward' in info:
                p = p.field("raw_reward", float(info['raw_reward']))
            if info.get('alt_used'):
                p = p.field("alt_used", str(info['alt_used']))
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=[p])
        except Exception as e:
            log.debug(f"Failed to write training metrics: {e}")
    
    def close(self):
        """Clean up resources."""
        try:
            self.write_api.close()
            self.client.close()
        except Exception:
            pass


# =============================================================================
#                              TRAINING LOOP
# =============================================================================
def train(args):
    """Main training loop."""
    log.info("=" * 60)
    log.info("Starting RL Training - DQN Agent v4 (Stacked Obs + Actions)")
    log.info("=" * 60)
    log.info(f"Configuration:")
    log.info(f"  State dim: {STATE_DIM} ({RAW_STATE_DIM} obs * {STACK_SIZE} + {ACTION_DIM} act * {STACK_SIZE})")
    log.info(f"  Action dim: {ACTION_DIM} (one-hot encoded in state)")
    log.info(f"  Hidden dim: {HIDDEN_DIM}")
    log.info(f"  Learning rate: {LR}, Gamma: {GAMMA}")
    log.info(f"  Batch size: {BATCH_SIZE}, Replay capacity: {REPLAY_CAPACITY}")
    log.info(f"  Min replay: {MIN_REPLAY_SIZE}")
    log.info(f"  Epsilon: {EPS_START} -> {EPS_END} over {EPS_DECAY_STEPS} steps")
    log.info(f"  Timing: Window={WINDOW_SECONDS}s, Delay={DELAY_AFTER_ACTION}s, Cooldown={COOLDOWN_SECONDS}s")
    log.info(f"  Max steps: {args.steps}")
    log.info("=" * 60)
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    # Initialize environment and agent
    env = QoSRoutingEnv(args.influx_bucket, args.influx_token, 
                        args.influx_org, args.influx_url)
    agent = DQNAgent(STATE_DIM, ACTION_DIM, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = args.resume
        if not resume_path.endswith('.pth'):
            resume_path = os.path.join(args.save_dir, f"dqn_v4_{args.resume}.pth")
        if os.path.exists(resume_path):
            agent.load(resume_path)
            log.info(f"Resumed training from {resume_path}")
        else:
            log.warning(f"Checkpoint not found: {resume_path}, starting fresh")
    
    # Training state
    total_steps = 0
    episode = 0
    best_avg_reward = -float('inf')
    
    # Checkpoints
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    checkpoint_steps = {
        int(args.steps * 0.25): '25pct',
        int(args.steps * 0.50): '50pct',
        int(args.steps * 0.75): '75pct',
        args.steps: 'final',
    }
    
    # CSV logging for local analysis
    csv_path = f"data/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'step', 'episode', 'action', 'reward', 'raw_reward',
        'sla_met', 'sla_streak', 'eps', 'loss', 'avg_reward_100', 'alt_used', 'timestamp'
    ])
    log.info(f"Training log: {csv_path}")
    
    try:
        while total_steps < args.steps:
            episode += 1
            state = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            log.info(f"\n{'='*50}")
            log.info(f"Episode {episode} started (total steps: {total_steps})")
            
            while not done and total_steps < args.steps:
                total_steps += 1
                episode_steps += 1
                
                # Get valid actions and select action
                valid_mask = env.get_valid_actions()
                action = agent.select_action(state, valid_mask, explore=True)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store experience and train
                agent.push_experience(state, action, reward, next_state, done)
                loss = agent.train_step()
                agent.update_epsilon()
                
                episode_reward += reward
                state = next_state
                
                # Logging
                stats = agent.get_stats()
                
                if total_steps % args.log_every == 0:
                    loss_str = f"{stats['last_loss']:.4f}" if stats['last_loss'] is not None else "N/A"
                    # Map action to readable name
                    if action == 0:
                        action_name = "noop"
                    elif 1 <= action <= 3:
                        action_name = f"v0-alt{action-1}"
                    elif 4 <= action <= 6:
                        action_name = f"v1-alt{action-4}"
                    elif 7 <= action <= 9:
                        action_name = f"be-alt{action-7}"
                    else:
                        action_name = str(action)
                    log.info(
                        f"[Step {total_steps}] "
                        f"action={action_name:8s} "
                        f"reward={reward:+.2f} "
                        f"eps={stats['eps']:.3f} "
                        f"buffer={stats['buffer_size']:5d} "
                        f"loss={loss_str:>8} "
                        f"sla={len(info['sla_met'])}/3 "
                        f"streak={info['sla_streak']}"
                    )
                
                # Write metrics to InfluxDB
                env.write_training_metrics(total_steps, stats, reward, action, info)
                
                # Write to local CSV
                csv_writer.writerow([
                    total_steps, episode, action, reward,
                    info.get('raw_reward', 0),
                    len(info['sla_met']), info['sla_streak'],
                    stats['eps'], stats['avg_loss'], stats['avg_reward'],
                    info.get('alt_used', ''), datetime.now().isoformat()
                ])
                csv_file.flush()  # Ensure data is written immediately
                
                # Save checkpoints
                if total_steps in checkpoint_steps:
                    tag = checkpoint_steps[total_steps]
                    path = os.path.join(args.save_dir, f"dqn_v4_{tag}.pth")
                    agent.save(path)
                    log.info(f"Checkpoint saved: {path}")
            
            # Episode summary
            log.info(
                f"Episode {episode} finished: "
                f"steps={episode_steps}, "
                f"reward={episode_reward:.2f}, "
                f"avg_reward_100={stats['avg_reward']:.2f}"
            )
            
            # Track best model
            if stats['avg_reward'] > best_avg_reward and len(agent.rewards) >= 50:
                best_avg_reward = stats['avg_reward']
                path = os.path.join(args.save_dir, "dqn_v4_best.pth")
                agent.save(path)
                log.info(f"New best model saved: avg_reward={best_avg_reward:.3f}")
    
    except KeyboardInterrupt:
        log.info("\nTraining interrupted by user")
    
    finally:
        # Save final model
        path = os.path.join(args.save_dir, "dqn_v4_final.pth")
        agent.save(path)
        env.close()
        csv_file.close()
        
    log.info("\nTraining complete!")
    log.info(f"Final stats: {agent.get_stats()}")
    log.info(f"Training log saved to: {csv_path}")


def evaluate(args):
    """Evaluation loop (no training)."""
    log.info("=" * 60)
    log.info("Starting RL Evaluation - DQN Agent v4")
    log.info("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize environment and agent
    env = QoSRoutingEnv(args.influx_bucket, args.influx_token,
                        args.influx_org, args.influx_url, verbose=args.verbose)
    agent = DQNAgent(STATE_DIM, ACTION_DIM, device)
    
    # Load weights
    weights_path = os.path.join(args.save_dir, f"dqn_v4_{args.weights_tag}.pth")
    if not os.path.exists(weights_path):
        log.error(f"Weights file not found: {weights_path}")
        return
    
    agent.load(weights_path)
    agent.eps = 0.0  # No exploration during evaluation
    
    total_reward = 0.0
    sla_met_total = 0
    sla_checks = 0
    
    try:
        state = env.reset()
        
        for step in range(1, args.steps + 1):
            valid_mask = env.get_valid_actions()
            action = agent.select_action(state, valid_mask, explore=False)
            
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            sla_met_total += len(info['sla_met'])
            sla_checks += len(QIDS)
            
            if action == 0:
                action_name = "noop"
            elif 1 <= action <= 3:
                action_name = f"voice-{action-1}"
            elif 4 <= action <= 6:
                action_name = f"video-{action-4}"
            elif 7 <= action <= 9:
                action_name = f"best-{action-7}"
            else:
                action_name = str(action)
            
            if step % args.log_every == 0:
                log.info(
                    f"[Eval Step {step}] "
                    f"action={action_name:6s} "
                    f"reward={reward:+.2f} "
                    f"sla_met={len(info['sla_met'])}/3"
                )
            
            state = next_state
            
            if done:
                state = env.reset()
    
    except KeyboardInterrupt:
        log.info("\nEvaluation interrupted")
    
    finally:
        env.close()
    
    log.info("\nEvaluation complete!")
    log.info(f"Total reward: {total_reward:.2f}")
    log.info(f"SLA compliance: {sla_met_total}/{sla_checks} ({100*sla_met_total/max(1,sla_checks):.1f}%)")


# =============================================================================
#                                 MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DQN Agent v4 for QoS Routing")
    
    # Mode
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                        help='Training or evaluation mode')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=30000,
                        help='Total training/eval steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-every', type=int, default=10,
                        help='Log frequency')
    
    # Model
    parser.add_argument('--save-dir', default='training_files',
                        help='Directory to save/load model weights')
    parser.add_argument('--weights-tag', default='final',
                        help='Weight file tag for evaluation (e.g., final, best, 50pct)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint (e.g., 50pct, best, or path to .pth file)')
    
    # InfluxDB
    parser.add_argument('--influx-url', default='http://192.168.201.1:8086')
    parser.add_argument('--influx-org', default='research')
    parser.add_argument('--influx-bucket', default='INT')
    parser.add_argument('--influx-token', 
                        default='0fO0ojKAANp-7aEehJHRDWEKE-cSNoIEHY2aK8dd1KI0VWpmO1GAsMJhRh_B1U8bXDIaozHMDVv1yEkCPm230w==')
    
    # Controller output verbosity
    parser.add_argument('--verbose', action='store_true',
                        help='Show verbose P4 controller output (route add/delete messages)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()

