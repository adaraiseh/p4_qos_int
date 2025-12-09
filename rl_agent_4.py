#!/usr/bin/env python3
"""
rl_agent_4.py - Simplified DQN for per-queue QoS path optimization using P4 INT reports

Key Design Principles:
1. Single centralized DQN agent (not multi-agent) - eliminates coordination overhead
2. Compact state space (13 features) - only essential normalized metrics
3. Clear SLA-based reward - bounded, easy to understand credit assignment
4. Focused action space (4 actions) - no-op + one change per queue
5. Prioritized Experience Replay - learn from rare important events
6. Proper episode boundaries - clear termination conditions
7. Longer observation window (3s) - stable metrics
8. Shorter cooldown (2s) - faster learning cycles

Author: Research Team
"""

import os
import sys
import time
import random
import logging
import argparse
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
HIDDEN_DIM = 128
STATE_DIM = 13  # Compact state: 3 queues × 4 features + 1 global
ACTION_DIM = 4  # no-op, change_voice, change_video, change_best

# Learning
LR = 3e-4
GAMMA = 0.95  # Lower discount - focus on immediate effects of routing changes
BATCH_SIZE = 32
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
EPS_DECAY_STEPS = 5_000

# Target network
TARGET_UPDATE_FREQ = 100  # Hard update every N steps
TAU = 0.005  # Soft update rate (if using soft updates)

# Environment
WINDOW_SECONDS = 3  # Longer window for stable metrics
SAFETY_LAG_MS = 500
COOLDOWN_SECONDS = 2.0  # Shorter cooldown for faster learning
DELAY_AFTER_ACTION = 0.5
DELAY_NO_ACTION = 0.1

# Episode
MAX_EPISODE_STEPS = 100
SLA_STREAK_TO_END = 15  # End episode early if SLA consistently met

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

# Reward weights
REWARD_SLA_MET = 1.0
REWARD_SLA_VIOLATED_SCALE = 1.5
REWARD_DROP_PENALTY = 0.3
REWARD_ACTION_COST = 0.1  # Small cost for taking action (encourages stability)

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
        
        # Update target network periodically
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
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
    
    State: 13 features
      - Per queue (3 queues × 4 features = 12):
        - lat_ratio: p95 latency / SLA threshold, clipped to [0, 2]
        - drop_norm: drop rate normalized to [0, 1]
        - util_norm: tx utilization normalized to [0, 1]
        - alt_available: 1.0 if alternate path exists, 0.0 otherwise
      - Global (1 feature):
        - max_pressure: maximum pressure across all queues
    
    Actions: 4
      - 0: No-op (do nothing)
      - 1: Change voice path (qid=0)
      - 2: Change video path (qid=1)
      - 3: Change best-effort path (qid=7)
    
    Reward:
      - +1.0 per queue meeting SLA
      - -1.5 × violation_ratio per queue violating SLA
      - -0.3 × drop_norm penalty
      - -0.1 action cost (for non-noop actions)
    """
    
    # Action to queue mapping
    ACTION_TO_QID = {
        0: None,  # No-op
        1: 0,     # Voice
        2: 1,     # Video
        3: 7,     # Best-effort
    }
    
    def __init__(self, bucket: str, token: str, org: str, url: str):
        self.bucket = bucket
        self.org = org
        self.url = url
        
        # InfluxDB client
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=5000)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        # Controller for routing changes
        self.controller = Controller()
        
        # Episode state
        self.episode_step = 0
        self.sla_streak = 0
        self.last_action_time = 0.0
        self.last_action = 0
        
        # Cache last snapshot
        self.last_snapshot = None
    
    def reset(self) -> np.ndarray:
        """Reset episode and return initial state."""
        self.episode_step = 0
        self.sla_streak = 0
        self.last_action = 0
        self.last_action_time = time.monotonic()
        
        self.last_snapshot = self._collect_snapshot()
        return self._build_state(self.last_snapshot)
    
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
        
        # Get hottest demand and path info for each queue
        for qid in QIDS:
            hot = self._get_hottest_demand(qid)
            if hot:
                src_ip, dst_ip = hot
                snapshot[qid]['hot_src_ip'] = src_ip
                snapshot[qid]['hot_dst_ip'] = dst_ip
                self._fill_path_info(qid, snapshot[qid])
        
        return snapshot
    
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
        
        # Check if alternate path exists
        if best_sid is not None:
            snap_q['alt_exists'] = self.controller.has_alternate_for_worst(best_sid, path)
    
    def _build_state(self, snapshot: Dict[int, Dict]) -> np.ndarray:
        """
        Build compact 13-dimensional state vector.
        
        Layout:
          [0-3]: Voice (lat_ratio, drop_norm, util_norm, alt_available)
          [4-7]: Video
          [8-11]: Best-effort
          [12]: Global max pressure
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        
        pressures = []
        idx = 0
        for qid in QIDS:
            q = snapshot[qid]
            sla = SLA_THRESHOLDS[qid]
            
            # Normalized metrics
            lat_ratio = min(q['lat_p95'] / sla, 2.0) / 2.0  # [0, 1]
            drop_norm = min(q['drop_p95'], DROP_CAP) / DROP_CAP
            util_norm = min(q['util_p95'], UTIL_CAP) / UTIL_CAP
            alt_avail = 1.0 if q['alt_exists'] else 0.0
            
            state[idx:idx+4] = [lat_ratio, drop_norm, util_norm, alt_avail]
            idx += 4
            
            # Track pressure for global feature
            pressure = 0.5 * lat_ratio + 0.3 * drop_norm + 0.2 * util_norm
            pressures.append(pressure)
        
        # Global max pressure
        state[12] = max(pressures)
        
        return state
    
    def _get_valid_actions(self, snapshot: Dict[int, Dict]) -> np.ndarray:
        """
        Get mask of valid actions.
        
        - No-op (0) is always valid
        - Change actions (1-3) valid only if alternate exists and not in cooldown
        """
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[0] = True  # No-op always valid
        
        now = time.monotonic()
        in_cooldown = (now - self.last_action_time) < COOLDOWN_SECONDS
        
        if not in_cooldown:
            for action, qid in self.ACTION_TO_QID.items():
                if qid is not None:
                    if snapshot[qid]['alt_exists'] and snapshot[qid]['bottleneck_sid'] is not None:
                        mask[action] = True
        
        return mask
    
    def _compute_reward(self, snapshot: Dict[int, Dict], action: int) -> Tuple[float, Dict]:
        """
        Compute reward based on SLA compliance.
        
        Returns:
            (reward, info_dict)
        """
        reward = 0.0
        info = {'sla_met': [], 'sla_violated': []}
        
        for qid in QIDS:
            q = snapshot[qid]
            sla = SLA_THRESHOLDS[qid]
            lat = q['lat_p95']
            
            if lat <= sla:
                # SLA met - positive reward
                reward += REWARD_SLA_MET
                info['sla_met'].append(qid)
            else:
                # SLA violated - negative reward proportional to violation
                violation_ratio = min((lat - sla) / sla, 1.0)
                reward -= REWARD_SLA_VIOLATED_SCALE * (1 + violation_ratio)
                info['sla_violated'].append(qid)
            
            # Drop penalty (always applies)
            drop_norm = min(q['drop_p95'], DROP_CAP) / DROP_CAP
            reward -= REWARD_DROP_PENALTY * drop_norm
        
        # Action cost
        if action != 0:
            reward -= REWARD_ACTION_COST
        
        # Clip reward to reasonable range
        reward = max(-10.0, min(5.0, reward))
        
        info['total_reward'] = reward
        return reward, info
    
    def _apply_action(self, action: int, snapshot: Dict[int, Dict]) -> bool:
        """
        Apply routing action.
        
        Returns:
            True if action was successfully applied
        """
        if action == 0:
            return False  # No-op
        
        qid = self.ACTION_TO_QID.get(action)
        if qid is None:
            return False
        
        q = snapshot[qid]
        src_ip = q.get('hot_src_ip')
        dst_ip = q.get('hot_dst_ip')
        bottleneck_sid = q.get('bottleneck_sid')
        path = q.get('path_nodes', [])
        
        if not (src_ip and dst_ip and bottleneck_sid):
            log.warning(f"Cannot apply action {action}: missing path info for qid={qid}")
            return False
        
        alt = self.controller.find_alternate_for_worst(int(bottleneck_sid), path)
        if not alt:
            log.warning(f"Cannot apply action {action}: no alternate found for qid={qid}")
            return False
        
        ok, msg = self.controller.reroute_one_demand_symmetric(
            src_ip=src_ip, dst_ip=dst_ip, qid=qid,
            worst_switch_id=int(bottleneck_sid), alt_switch_name=alt
        )
        
        if ok:
            log.info(f"[ACTION {action}] Rerouted qid={qid}: {src_ip}->{dst_ip} via {alt} (was bottleneck sid={bottleneck_sid})")
        else:
            log.warning(f"[ACTION {action}] Reroute failed for qid={qid}: {msg}")
        
        return ok
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action index (0-3)
        
        Returns:
            (next_state, reward, done, info)
        """
        self.episode_step += 1
        
        # Get current snapshot
        current_snapshot = self.last_snapshot
        
        # Apply action
        action_applied = self._apply_action(action, current_snapshot)
        
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
        
        done = (
            self.episode_step >= MAX_EPISODE_STEPS or
            self.sla_streak >= SLA_STREAK_TO_END
        )
        
        info['episode_step'] = self.episode_step
        info['sla_streak'] = self.sla_streak
        info['action_applied'] = action_applied
        info['all_sla_met'] = all_sla_met
        
        # Build next state
        next_state = self._build_state(next_snapshot)
        self.last_snapshot = next_snapshot
        
        return next_state, reward, done, info
    
    def get_valid_actions(self) -> np.ndarray:
        """Get valid action mask for current state."""
        return self._get_valid_actions(self.last_snapshot)
    
    def write_training_metrics(self, step: int, agent_stats: Dict, 
                                reward: float, action: int, info: Dict):
        """Write training metrics to InfluxDB."""
        try:
            p = (
                Point("rl_training_v4")
                .tag("action", str(action))
                .field("step", int(step))
                .field("reward", float(reward))
                .field("eps", float(agent_stats['eps']))
                .field("beta", float(agent_stats['beta']))
                .field("buffer_size", int(agent_stats['buffer_size']))
                .field("avg_loss", float(agent_stats['avg_loss']))
                .field("avg_reward_100", float(agent_stats['avg_reward']))
                .field("episode_step", int(info.get('episode_step', 0)))
                .field("sla_streak", int(info.get('sla_streak', 0)))
                .field("sla_met_count", len(info.get('sla_met', [])))
                .time(datetime.utcnow())
            )
            
            if agent_stats['last_loss'] is not None:
                p = p.field("last_loss", float(agent_stats['last_loss']))
            
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
    log.info("Starting RL Training - DQN Agent v4")
    log.info("=" * 60)
    log.info(f"Configuration:")
    log.info(f"  State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    log.info(f"  Hidden dim: {HIDDEN_DIM}")
    log.info(f"  Learning rate: {LR}, Gamma: {GAMMA}")
    log.info(f"  Batch size: {BATCH_SIZE}, Replay capacity: {REPLAY_CAPACITY}")
    log.info(f"  Min replay: {MIN_REPLAY_SIZE}")
    log.info(f"  Epsilon: {EPS_START} -> {EPS_END} over {EPS_DECAY_STEPS} steps")
    log.info(f"  Window: {WINDOW_SECONDS}s, Cooldown: {COOLDOWN_SECONDS}s")
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
    
    # Training state
    total_steps = 0
    episode = 0
    best_avg_reward = -float('inf')
    
    # Checkpoints
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_steps = {
        int(args.steps * 0.25): '25pct',
        int(args.steps * 0.50): '50pct',
        int(args.steps * 0.75): '75pct',
        args.steps: 'final',
    }
    
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
                action_name = ['noop', 'voice', 'video', 'best'][action]
                stats = agent.get_stats()
                
                if total_steps % args.log_every == 0:
                    loss_str = f"{stats['last_loss']:.4f}" if stats['last_loss'] is not None else "N/A"
                    log.info(
                        f"[Step {total_steps}] "
                        f"action={action_name:6s} "
                        f"reward={reward:+.2f} "
                        f"eps={stats['eps']:.3f} "
                        f"buffer={stats['buffer_size']:5d} "
                        f"loss={loss_str:>8} "
                        f"sla_met={len(info['sla_met'])}/3 "
                        f"streak={info['sla_streak']}"
                    )
                
                # Write metrics to InfluxDB
                env.write_training_metrics(total_steps, stats, reward, action, info)
                
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
        
    log.info("\nTraining complete!")
    log.info(f"Final stats: {agent.get_stats()}")


def evaluate(args):
    """Evaluation loop (no training)."""
    log.info("=" * 60)
    log.info("Starting RL Evaluation - DQN Agent v4")
    log.info("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize environment and agent
    env = QoSRoutingEnv(args.influx_bucket, args.influx_token,
                        args.influx_org, args.influx_url)
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
            
            action_name = ['noop', 'voice', 'video', 'best'][action]
            
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
    parser.add_argument('--steps', type=int, default=10000,
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
    
    # InfluxDB
    parser.add_argument('--influx-url', default='http://192.168.201.1:8086')
    parser.add_argument('--influx-org', default='research')
    parser.add_argument('--influx-bucket', default='INT')
    parser.add_argument('--influx-token', 
                        default='0fO0ojKAANp-7aEehJHRDWEKE-cSNoIEHY2aK8dd1KI0VWpmO1GAsMJhRh_B1U8bXDIaozHMDVv1yEkCPm230w==')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()

