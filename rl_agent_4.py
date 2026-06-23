#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_agent_4.py - Simplified DQN for per-queue QoS path optimization using P4 INT reports

Key Design Principles:
1. Single centralized DQN agent
2. Frame stacking (960 features) - stacked observations + stacked one-hot actions
3. Action history as one-hot vectors - agent knows "I caused this" vs "happened naturally"
4. Clear SLA-based reward - bounded, no improvement bonus (avoids rewarding noise)
5. Focused action space (8 actions) - no-op + 6 single + 1 multi-queue
6. Prioritized Experience Replay - learn from rare important events
7. Proper episode boundaries - clear termination conditions
8. Tuned timing for 100% post-action data capture
9. Queue-specific bottleneck detection and alternative metrics
10. EMA temporal smoothing for latency trends

State Composition (960 features):
- Stacked Observations: 52 metrics * 16 frames = 832 features
- Stacked Actions (one-hot): 8 actions * 16 frames = 128 features
- Total: 960 features

Benefits:
- Agent sees velocity/trends (is latency rising or falling?)
- Agent knows full action history via one-hot encoding
- Queue-specific metrics ensure accurate comparison between bottleneck and alternatives
- Multi-queue action enables rapid stabilization
- Prevents sawtooth over-correction patterns

Author: Research Team
"""

import os
import sys
import time
import random
import logging
import argparse
import math
import glob
import copy
import csv
import json
import hashlib
import platform
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =============================================================================
#                              LOGGING SETUP
# =============================================================================
# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import unified logging configuration
from logging_config import (
    get_log_file_path,
    normalize_artifact_permissions,
    setup_unified_logging,
    set_console_level,
)

def setup_logging(log_level: str = "info"):
    """Configure logging with appropriate level and file output.

    Args:
        log_level: File log level ("debug", "info", "warning", "error")
    """
    # Configure root logger with unified logging (file + console)
    setup_unified_logging(module_name="rl_agent", log_level=log_level)

    # Set level for this module
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    # Ensure traffic_generator logger matches
    logging.getLogger('traffic_generator').setLevel(logging.DEBUG)

    # Ensure controller logger matches
    logging.getLogger('controller').setLevel(logging.DEBUG)

    # Log the log file location
    log_path = get_log_file_path()
    if log_path:
        log.info(f"Debug logs will be written to: {log_path}")

log = logging.getLogger(__name__)
# Default to INFO until setup_logging is called
log.setLevel(logging.INFO)

# Now import modules that use logging
from controller import Controller
from traffic_generator import TrafficManager
from config.schema import MAX_SWITCHES
from report_collector.local_telemetry_cache import (
    DEFAULT_SOCKET_PATH,
    LocalTelemetryClient,
    iso_to_ns,
)

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


# =============================================================================
#                           HYPERPARAMETERS
# =============================================================================
# Network
HIDDEN_DIM = 128        # Reduced from 256 for smaller state space
RAW_STATE_DIM = 52      # INCREASED from 50 to 52: 3×16 + 2 + 2 (added topology encoding: is_fat_tree, is_leaf_spine)
STACK_SIZE = 16         # Extended for burst detection (~32s history at ~2s/step)
STACK_DECAY = 0.85      # SHARPENED from 0.95 to 0.85 - reduces weight of stale data (oldest ~8% vs 46%)
ACTION_DIM = 8          # No-op + 6 single (3 queues × 2 alts) + 1 multi
# State composition: stacked observations + stacked one-hot actions
# Observations: 52 metrics * 16 frames = 832
# Actions: 8 (one-hot) * 16 frames = 128
# Total: 832 + 128 = 960
STATE_DIM = (RAW_STATE_DIM * STACK_SIZE) + (ACTION_DIM * STACK_SIZE)

# Temporal smoothing
LAT_EMA_ALPHA = 0.3     # EMA smoothing for latency ratio

# Learning
LR = 1e-4  # Increased from 1e-5 for faster convergence
GAMMA = 0.97  # Faster credit assignment for ~2s step delay
BATCH_SIZE = 64  # Increased from 32 for more stable gradients
MIN_REPLAY_SIZE = 500  # Start learning much sooner
REPLAY_CAPACITY = 50_000

# Prioritized Experience Replay
PER_ALPHA = 0.6  # Prioritization exponent
PER_BETA_START = 0.4  # Importance sampling start
PER_BETA_END = 1.0
PER_BETA_STEPS = 25_000  # Anneal to 1.0 by ~50% of 50K training

# Epsilon schedule
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 45_000  # Decay over 80% of 55K training for thorough exploration

# Target network - Soft updates (Polyak averaging) for smooth Q-value evolution
TAU = 0.005  # Soft update rate: target = TAU * online + (1-TAU) * target

# EWC (Elastic Weight Consolidation) - prevents catastrophic forgetting
EWC_LAMBDA = 5000.0       # Regularization strength (tune: 1000-10000)
EWC_FISHER_SAMPLES = 200  # Samples for Fisher matrix estimation

# Environment timing - tuned for faster training with acceptable data capture
# Based on sync test results: first_change ~0.28s, query RTT ~0.3s
# Formula: DELAY_AFTER_ACTION >= WINDOW + first_change + INT_sample_period
# With 100ms INT sampling: 1.0s window ensures ~10 samples captured
WINDOW_SECONDS = 1.0        # Observation window
SAFETY_LAG_MS = 0           # REMOVED: No safety lag (user constraint: minimize delay)
COOLDOWN_SECONDS = 0.0      # No cooldown for faster learning
DELAY_AFTER_ACTION = 1.0    # Sleep after action
DELAY_NO_ACTION = 1.0       # Match action delay

# Freshness validation - minimum data points required per metric in window
MIN_POINTS_PER_METRIC = 1   # Require at least 1 point (2 is too strict)

# Episode
MAX_EPISODE_STEPS = 100
# Note: No early termination - let agent learn to maintain good state, not just fix bad state
INVALID_RECOVERY_STREAK = 8
TELEMETRY_LIVENESS_RETRIES = 6
TELEMETRY_LIVENESS_INTERVAL_SECONDS = 0.5
TELEMETRY_LIVENESS_RESTARTS = 1

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
LAT_RATIO_CAP = 3.0  # Cap latency ratio at 3x SLA (prevents instability)
LAT_DIFF_CAP = 2.0   # Cap lat difference features to [-2, +2]

# Reward weights - TUNED to prevent tanh saturation and balance incentives
REWARD_SLA_MET_SCALE = 0.5         # Positive reward for meeting SLA
REWARD_SLA_VIOLATED_SCALE = 0.35    # INCREASED from 0.1 to 0.35 - violations must be penalized adequately
REWARD_DROP_PENALTY = 0.8           # Reduced from 1.5 (with sqrt compression)

# Queue-specific action cost (replaces pressure-based approach)
# Cost depends on whether the TARGETED queue's SLA is met:
#   - If targeting a healthy queue → high cost (risky, don't break what works)
#   - If targeting a sick queue → low cost (encouraged to fix)
REWARD_ACTION_COST_HEALTHY = 0.5   # Cost when targeting a queue with SLA met (reduced from 0.65)
REWARD_ACTION_COST_SICK = 0.10      # Cost when targeting a queue with SLA violated

# Soft margin around SLA (reduces reward flip-flopping)
SLA_SOFT_MARGIN = 0.1  # REDUCED from 0.2 to 0.1 (10%) - tighter margin provides stronger training signal
SLA_MARGIN_LOW = 1.0 - SLA_SOFT_MARGIN   # 0.9 - below this is clearly meeting SLA
SLA_MARGIN_HIGH = 1.0 + SLA_SOFT_MARGIN  # 1.1 - above this is clearly violating SLA

# Pre-computed decay weights for state stacking (CPU optimization)
# weights[i] = STACK_DECAY^(STACK_SIZE-1-i), so oldest frame (i=0) has smallest weight
DECAY_WEIGHTS = np.array([STACK_DECAY ** (STACK_SIZE - 1 - i) for i in range(STACK_SIZE)])

# Target network update frequency (CPU optimization - update every N steps instead of every step)
TARGET_UPDATE_FREQ = 4

# =============================================================================
#                          HELPER FUNCTIONS
# =============================================================================
def action_to_name(action: int) -> str:
    """Convert action index to human-readable name."""
    ACTION_NAMES = {
        0: "noop",
        1: "v0-alt0", 2: "v0-alt1",
        3: "v1-alt0", 4: "v1-alt1",
        5: "be-alt0", 6: "be-alt1",
        7: "multi"
    }
    return ACTION_NAMES.get(action, str(action))


def resolve_checkpoint_path(save_dir: str, tag: str) -> Optional[str]:
    """Find latest checkpoint matching tag, with fallback to legacy naming."""
    pattern = os.path.join(save_dir, f"*-dqn_v4_{tag}.pth")
    matching = sorted(glob.glob(pattern), reverse=True)
    if matching:
        return matching[0]
    legacy = os.path.join(save_dir, f"dqn_v4_{tag}.pth")
    return legacy if os.path.exists(legacy) else None


def load_topology(args):
    """Load topology from config, return (builder, rules_dir)."""
    if not args.config:
        return None, args.rules_dir
    from topology.factory import create_topology
    builder = create_topology(args.config)
    rules_dir = args.rules_dir
    if rules_dir is None:
        topo_name = builder.config.topology.name.replace('-', '_')
        rules_dir = f"rules/{topo_name}"
    log.info(f"Topology: {builder.config.topology.name} ({len(builder.switches)}/{MAX_SWITCHES} max)")
    return builder, rules_dir


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
        # Iterative version - avoids recursive call overhead at deep tree levels
        while idx != 0:
            parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent
    
    def _retrieve(self, idx: int, s: float) -> int:
        # Iterative version - avoids recursive call overhead at deep tree levels
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
    
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
        # max_priority stores RAW priority (|TD| + eps), NOT exponentiated
        # This avoids double-alpha bug when push() applies alpha
        self.max_priority = 1.0
        self.epsilon = 1e-5
    
    def push(self, state, action, reward, next_state, terminated, next_valid_mask):
        """Add experience with max priority (ensures new samples get sampled)."""
        data = (state, action, reward, next_state, terminated, next_valid_mask)
        # Apply alpha once here (max_priority is raw, not exponentiated)
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
        
        if samples and len(samples[0]) == 6:
            states, actions, rewards, next_states, terminateds, next_valid_masks = zip(*samples)
        else:
            # Backward compatibility for older checkpoints whose replay entries
            # predate next-action-mask storage. These legacy samples keep the
            # old unconstrained target behavior until naturally overwritten by
            # new masked experiences.
            states, actions, rewards, next_states, terminateds = zip(*samples)
            next_valid_masks = [np.ones(ACTION_DIM, dtype=bool) for _ in samples]
        
        return (
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            terminateds,
            np.array(next_valid_masks, dtype=bool),
            indices,
            weights
        )
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            # Raw priority (before alpha exponentiation)
            raw_priority = abs(td_error) + self.epsilon
            # Update max_priority with RAW value (alpha applied in push())
            self.max_priority = max(self.max_priority, raw_priority)
            # Tree stores exponentiated priority
            self.tree.update(idx, raw_priority ** self.alpha)
    
    def __len__(self) -> int:
        return self.tree.n_entries


class MultiTopologyReplayBuffer:
    """
    Manages separate replay buffers per topology with balanced sampling.
    Each topology gets its own PrioritizedReplayBuffer.
    """

    def __init__(self, capacity_per_topology: int, alpha: float = 0.6):
        self.capacity_per_topology = capacity_per_topology
        self.alpha = alpha
        self.buffers: Dict[str, PrioritizedReplayBuffer] = {}
        self.current_topology = None
        # Track max_priority across all buffers for consistent new experience priority
        self.max_priority = 1.0

    def set_topology(self, topology_name: str):
        """Set current topology for push operations."""
        self.current_topology = topology_name
        if topology_name not in self.buffers:
            self.buffers[topology_name] = PrioritizedReplayBuffer(
                self.capacity_per_topology, self.alpha
            )
            log.info(f"Created replay buffer for topology: {topology_name}")

    def push(self, state, action, reward, next_state, terminated, next_valid_mask):
        """Add experience to current topology's buffer."""
        if self.current_topology is None:
            raise ValueError("Must call set_topology() before push()")
        self.buffers[self.current_topology].push(
            state, action, reward, next_state, terminated, next_valid_mask
        )

    def sample(self, batch_size: int, beta: float = 0.4, balance: bool = True):
        """
        Sample from buffers with optional balancing across topologies.

        If balance=True, samples equally from each topology buffer.
        If balance=False, samples only from current topology buffer.
        """
        if not self.buffers:
            raise ValueError("No buffers available")

        if not balance or len(self.buffers) == 1:
            # Sample from current topology only
            return self.buffers[self.current_topology].sample(batch_size, beta)

        # Balanced sampling: equal samples from each topology
        n_topos = len(self.buffers)
        samples_per_topo = batch_size // n_topos
        remainder = batch_size % n_topos

        all_states, all_actions, all_rewards = [], [], []
        all_next_states, all_terminateds, all_next_valid_masks = [], [], []
        all_indices, all_weights = [], []

        for i, (topo_name, buffer) in enumerate(self.buffers.items()):
            if len(buffer) < samples_per_topo:
                # Not enough samples in this buffer, skip
                continue

            n_samples = samples_per_topo + (1 if i < remainder else 0)
            (states, actions, rewards, next_states, terminateds, next_valid_masks,
             indices, weights) = buffer.sample(n_samples, beta)

            all_states.append(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_next_states.append(next_states)
            all_terminateds.extend(terminateds)
            all_next_valid_masks.append(next_valid_masks)
            # Tag indices with topology for priority updates
            all_indices.extend([(topo_name, idx) for idx in indices])
            all_weights.extend(weights)

        if not all_states:
            # Fallback to current topology
            return self.buffers[self.current_topology].sample(batch_size, beta)

        return (
            np.concatenate(all_states),
            all_actions,
            all_rewards,
            np.concatenate(all_next_states),
            all_terminateds,
            np.concatenate(all_next_valid_masks),
            all_indices,  # Now tuples of (topology_name, index)
            np.array(all_weights)
        )

    def update_priorities(self, indices, td_errors: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx_entry, td_error in zip(indices, td_errors):
            if isinstance(idx_entry, tuple):
                topo_name, idx = idx_entry
                self.buffers[topo_name].update_priorities([idx], np.array([td_error]))
            else:
                # Single buffer mode - should not happen but handle gracefully
                self.buffers[self.current_topology].update_priorities([idx_entry], np.array([td_error]))

    def __len__(self) -> int:
        return sum(len(b) for b in self.buffers.values())

    def get_stats(self) -> Dict[str, int]:
        """Get buffer sizes per topology."""
        return {name: len(buf) for name, buf in self.buffers.items()}


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
    Supports EWC for multi-topology training and separate replay buffers.
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device,
                 lr: float = LR, multi_buffer: bool = False,
                 buffer_capacity: int = REPLAY_CAPACITY,
                 balanced_sampling: bool = False):
        self.device = device
        self.action_dim = action_dim
        self.lr = lr  # Store for logging
        self.multi_buffer = multi_buffer
        self.balanced_sampling = balanced_sampling

        # Networks
        self.online_net = DuelingDQN(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer with configurable learning rate
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Replay buffer - single or multi-topology
        if multi_buffer:
            self.replay_buffer = MultiTopologyReplayBuffer(buffer_capacity, PER_ALPHA)
        else:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, PER_ALPHA)

        # Epsilon schedule
        self.eps = EPS_START
        self.step_count = 0      # Global training steps (for PER beta)
        self.eps_step_count = 0  # Steps for epsilon decay (can be reset)

        # PER beta schedule
        self.beta = PER_BETA_START

        # EWC (Elastic Weight Consolidation) state
        self.ewc_fisher = None
        self.ewc_optimal_params = None
        self.ewc_lambda = 0.0  # Set via args, 0 means disabled

        # Metrics
        self.losses = deque(maxlen=100)
        self.rewards = deque(maxlen=100)
        self.last_loss = None

        # Q-value tracking for logging
        self.q_values_max = deque(maxlen=100)
        self.q_values_mean = deque(maxlen=100)
    
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
            
            # Track Q-value stats BEFORE masking (for logging)
            valid_q = q_values[valid_mask]
            if len(valid_q) > 0:
                self.q_values_max.append(float(np.max(valid_q)))
                self.q_values_mean.append(float(np.mean(valid_q)))
            
            # Mask invalid actions
            q_values[~valid_mask] = -np.inf
            return int(np.argmax(q_values))
    
    def update_epsilon(self):
        """Update epsilon based on epsilon step count (Decoupled from global step count)."""
        self.step_count += 1      # Always increments (Global progress)
        self.eps_step_count += 1  # Increments but can be reset
        
        progress = min(1.0, self.eps_step_count / EPS_DECAY_STEPS)
        self.eps = EPS_END + (EPS_START - EPS_END) * (1 - progress)
        
        # Update PER beta
        beta_progress = min(1.0, self.step_count / PER_BETA_STEPS)
        self.beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * beta_progress
    
    def push_experience(self, state, action, reward, next_state, terminated, next_valid_mask):
        """Add experience to replay buffer."""
        if next_valid_mask is None:
            next_valid_mask = np.zeros(self.action_dim, dtype=bool)
            next_valid_mask[0] = True
        self.replay_buffer.push(
            state, action, reward, next_state, terminated, next_valid_mask
        )
        self.rewards.append(reward)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None

        # Sample from prioritized replay buffer (with optional balanced sampling)
        if self.multi_buffer:
            (states, actions, rewards, next_states, terminateds,
             next_valid_masks, indices, weights) = self.replay_buffer.sample(
                 BATCH_SIZE, self.beta, balance=self.balanced_sampling)
        else:
            (states, actions, rewards, next_states, terminateds,
             next_valid_masks, indices, weights) = self.replay_buffer.sample(BATCH_SIZE, self.beta)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        terminated_t = torch.BoolTensor(terminateds).to(self.device)
        next_valid_masks_t = torch.BoolTensor(next_valid_masks).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: use online net to select actions, target net to evaluate
        with torch.no_grad():
            next_q_online = self.online_net(next_states_t)
            next_q_online = next_q_online.masked_fill(~next_valid_masks_t, -1e9)
            next_actions = next_q_online.argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q[terminated_t] = 0.0  # Only zero bootstrap for true terminals, not truncations
            target_q = rewards_t + GAMMA * next_q

        # TD errors for priority update
        td_errors = (target_q - current_q).detach().cpu().numpy()

        # Weighted base loss (TD loss)
        base_loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        # Add EWC penalty if enabled (prevents catastrophic forgetting)
        ewc_loss = self.ewc_penalty()
        loss = base_loss + ewc_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        # Soft update target network (Polyak averaging) - every TARGET_UPDATE_FREQ steps
        # CPU optimization: reduce update frequency and use in-place lerp
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            # Compensate for reduced frequency with higher effective TAU
            effective_tau = TAU * TARGET_UPDATE_FREQ
            for target_param, online_param in zip(
                self.target_net.parameters(),
                self.online_net.parameters()
            ):
                # In-place lerp is faster than copy_ with arithmetic
                target_param.data.lerp_(online_param.data, effective_tau)

        self.last_loss = loss.item()
        self.losses.append(self.last_loss)

        return self.last_loss
    
    def save(self, path: str):
        """Save model checkpoint including replay buffer."""
        checkpoint = {
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'eps_step_count': self.eps_step_count,
            'eps': self.eps,
            'beta': self.beta,
            'multi_buffer': self.multi_buffer,
            'balanced_sampling': self.balanced_sampling,
        }
        if self.multi_buffer:
            checkpoint['multi_replay_buffers'] = {
                topo_name: {
                    'tree': buf.tree.tree.copy(),
                    'data': buf.tree.data.copy(),
                    'write': buf.tree.write,
                    'n_entries': buf.tree.n_entries,
                    'max_priority': buf.max_priority,
                }
                for topo_name, buf in self.replay_buffer.buffers.items()
            }
            checkpoint['current_topology'] = self.replay_buffer.current_topology
        else:
            # Replay buffer state for resume continuity
            checkpoint.update({
                'replay_tree': self.replay_buffer.tree.tree.copy(),
                'replay_data': self.replay_buffer.tree.data.copy(),
                'replay_write': self.replay_buffer.tree.write,
                'replay_n_entries': self.replay_buffer.tree.n_entries,
                'replay_max_priority': self.replay_buffer.max_priority,
            })
        torch.save(checkpoint, path)
        normalize_artifact_permissions(path, file_mode=0o664)
        log.info(f"Model saved to {path} (with replay buffer: {len(self.replay_buffer)} experiences)")
    
    def load(self, path: str):
        """Load model checkpoint including replay buffer if available."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint.get('step_count', 0)
        self.eps_step_count = checkpoint.get('eps_step_count', self.step_count)
        self.eps = checkpoint.get('eps', EPS_END)
        self.beta = checkpoint.get('beta', PER_BETA_END)
        
        # Load replay buffer if available
        if self.multi_buffer and 'multi_replay_buffers' in checkpoint:
            for topo_name, state in checkpoint['multi_replay_buffers'].items():
                self.replay_buffer.set_topology(topo_name)
                buf = self.replay_buffer.buffers[topo_name]
                buf.tree.tree = state['tree']
                buf.tree.data = state['data']
                buf.tree.write = state['write']
                buf.tree.n_entries = state['n_entries']
                buf.max_priority = state['max_priority']
            current_topology = checkpoint.get('current_topology')
            if current_topology:
                self.replay_buffer.set_topology(current_topology)
            log.info(f"Loaded multi-topology replay buffer with {len(self.replay_buffer)} experiences")
        elif 'replay_tree' in checkpoint and not self.multi_buffer:
            self.replay_buffer.tree.tree = checkpoint['replay_tree']
            self.replay_buffer.tree.data = checkpoint['replay_data']
            self.replay_buffer.tree.write = checkpoint['replay_write']
            self.replay_buffer.tree.n_entries = checkpoint['replay_n_entries']
            self.replay_buffer.max_priority = checkpoint['replay_max_priority']
            log.info(f"Loaded replay buffer with {len(self.replay_buffer)} experiences")
        elif 'replay_tree' in checkpoint and self.multi_buffer:
            log.warning("Checkpoint has single replay buffer but agent uses --multi-buffer; replay not loaded")
        else:
            log.info("No replay buffer in checkpoint, starting fresh")
        
        log.info(f"Model loaded from {path}")
    
    def get_stats(self) -> Dict:
        """Get agent statistics for logging and metrics."""
        return {
            'eps': self.eps,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'last_loss': self.last_loss,
            'q_max': np.mean(self.q_values_max) if self.q_values_max else 0.0,
        }

    # =========================================================================
    #                      EWC (Elastic Weight Consolidation)
    # =========================================================================
    def ewc_penalty(self) -> torch.Tensor:
        """
        Compute EWC regularization penalty.

        Returns penalty term: λ * Σ F_i * (θ_i - θ*_i)²
        where F_i is Fisher information, θ* are optimal params from previous task.
        """
        if self.ewc_fisher is None or self.ewc_lambda == 0.0:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.online_net.named_parameters():
            if n in self.ewc_fisher:
                penalty += (self.ewc_fisher[n] * (p - self.ewc_optimal_params[n]).pow(2)).sum()

        return self.ewc_lambda * penalty

    def compute_fisher_matrix(self, env, num_samples: int = EWC_FISHER_SAMPLES) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix using sampled gradients.
        Call this after training on a topology, before switching.

        The Fisher matrix approximates parameter importance for the current task.
        """
        log.info(f"Computing Fisher Information Matrix ({num_samples} samples)...")
        fisher = {n: torch.zeros_like(p) for n, p in self.online_net.named_parameters()}
        self.online_net.train()

        for i in range(num_samples):
            if i % 50 == 0:
                log.info(f"  Fisher sample {i}/{num_samples}")

            # Get a state from the environment
            state = env.reset(force_reset=True)

            # Forward pass
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)

            # Sample action from policy (softmax over Q-values)
            probs = F.softmax(q_values, dim=1)
            action = torch.multinomial(probs, 1).item()

            # Compute log probability gradient
            log_prob = F.log_softmax(q_values, dim=1)[0, action]

            self.optimizer.zero_grad()
            log_prob.backward()

            # Accumulate squared gradients
            for n, p in self.online_net.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)

        # Average over samples
        for n in fisher:
            fisher[n] /= num_samples

        log.info("Fisher Information Matrix computed.")
        return fisher

    def save_ewc(self, path: str, env, num_samples: int = EWC_FISHER_SAMPLES):
        """
        Save Fisher matrix and optimal parameters for EWC.
        Call after training on a topology to prepare for next topology.
        """
        fisher = self.compute_fisher_matrix(env, num_samples)
        optimal_params = {n: p.data.clone() for n, p in self.online_net.named_parameters()}

        torch.save({
            'fisher': fisher,
            'optimal_params': optimal_params,
        }, path)
        log.info(f"EWC data saved to {path}")

    def load_ewc(self, path: str):
        """
        Load Fisher matrix and optimal parameters for EWC regularization.
        Call before training on a new topology.
        """
        data = torch.load(path, map_location=self.device)
        self.ewc_fisher = {k: v.to(self.device) for k, v in data['fisher'].items()}
        self.ewc_optimal_params = {k: v.to(self.device) for k, v in data['optimal_params'].items()}
        log.info(f"EWC data loaded from {path}")

    def set_topology(self, topology_name: str):
        """Set current topology for multi-buffer replay."""
        if self.multi_buffer:
            self.replay_buffer.set_topology(topology_name)


# =============================================================================
#                           ENVIRONMENT
# =============================================================================
class QoSRoutingEnv:
    """
    Environment for QoS-aware routing optimization using P4 INT metrics.
    
    State: 464 features (8-frame stacking to capture action delay + trends)
      Frame stacking provides velocity/trend information.
      
      Raw observation (50 features):
        - Per queue (16 features × 3 = 48):
            - Basic metrics: lat_ratio, drop_norm, util_norm, sla_met, lat_ema, lat_ema_diff (6)
            - Bottleneck: present, drop, lat, util (4)
            - Alternatives (2 × 3 = 6): available, drop_vs_bn, lat_vs_bn
        - Global (2): max_pressure, steps_since_action (2)
      
    Actions: 8
      - 0: No-op
      - 1-2: Queue 0 -> Alt 0, 1 (Voice)
      - 3-4: Queue 1 -> Alt 0, 1 (Video)
      - 5-6: Queue 7 -> Alt 0, 1 (BE)
      - 7: Multi-queue reroute (all violating queues)
    
    Reward:
      - SLA-based with soft margin, drop penalty, action cost
    """
    
    # Action to (queue, alt_index) mapping
    # alt_index is 0-based index into the available alternatives list
    # 'multi' triggers multi-queue rerouting for all violating queues
    ACTION_MAP = {
        0: None,                         # No-op
        1: (0, 0), 2: (0, 1),            # Voice alts
        3: (1, 0), 4: (1, 1),            # Video alts
        5: (7, 0), 6: (7, 1),            # BE alts
        7: 'multi',                       # Multi-queue reroute
    }
    
    MAX_ALTS = 2
    # Use MAX_SWITCHES from config for topology-agnostic fixed-size state
    # This allows trained models to work on any topology up to this size
    NUM_SWITCHES = MAX_SWITCHES

    def __init__(self, bucket: str, token: str, org: str, url: str,
                 verbose: bool = False, reset_network: bool = True,
                 production_mode: bool = False, topology_builder=None,
                 rules_dir: str = None, config_path: str = None,
                 traffic_seed: Optional[int] = None,
                 telemetry_backend: str = "influx",
                 telemetry_cache_socket: str = DEFAULT_SOCKET_PATH,
                 telemetry_cache_timeout: float = 1.0):
        self.bucket = bucket
        self.org = org
        self.url = url
        self.telemetry_backend = telemetry_backend
        self.telemetry_cache_socket = telemetry_cache_socket
        self.telemetry_cache = (
            LocalTelemetryClient(
                socket_path=telemetry_cache_socket,
                timeout=telemetry_cache_timeout,
            )
            if telemetry_backend in ("cache", "cache-fallback-influx")
            else None
        )

        # InfluxDB client. Pure cache mode can run without host InfluxDB.
        # PHASE 2.4: Reduced timeout from 5000ms to 2000ms for faster failure detection.
        self.client = None
        self.query_api = None
        self.write_api = None
        if token:
            self.client = InfluxDBClient(url=url, token=token, org=org, timeout=2000)
            self.query_api = self.client.query_api()
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        elif telemetry_backend in ("influx", "cache-fallback-influx"):
            log.warning("InfluxDB token missing; Influx telemetry fallback is unavailable")

        log.info(f"Telemetry backend: {self.telemetry_backend}")
        if self.telemetry_cache is not None:
            log.info(f"Local telemetry cache socket: {self.telemetry_cache_socket}")

        # Store topology builder for reference
        self._topology_builder = topology_builder
        self._config_path = config_path

        # Controller for routing changes
        # Pass topology_builder for dynamic role detection if available
        self.controller = Controller(
            verbose=verbose,
            topology_builder=topology_builder,
            rules_dir=rules_dir
        )
        
        # Reset behavior: if True, perform full network reset on each episode
        # If False (production mode), preserve network state across resets
        self.reset_network = reset_network
        self.production_mode = production_mode
        
        # Initialize switch mapping for One-Hot encoding
        # We need a stable mapping of switch IDs to indices 0..N-1
        self.all_switch_ids = self.controller.get_all_switch_ids()
        # The mapping supports up to MAX_SWITCHES for topology-agnostic operation
        # Actual switch count may be less; unused slots are ignored
        self.sid_to_idx = {sid: i for i, sid in enumerate(self.all_switch_ids)}
        log.info(f"Initialized switch mapping ({len(self.all_switch_ids)}/{self.NUM_SWITCHES} max): {self.sid_to_idx}")
        
        # Global step counter (persists across episodes)
        self.global_step = 0
       
        # Episode state
        self.episode_step = 0
        self.sla_streak = 0
        self.last_action_time = 0.0
        self.last_action = 0
        self._last_action_global_step = 0
        
        # Cache snapshots for comparison
        self.last_snapshot = None
        self._last_rerouted_qids = []

        # Frame stacking for velocity/trend detection
        # Stores last STACK_SIZE raw observation states (each 50-dim)
        self.frame_stack: deque = deque(maxlen=STACK_SIZE)
        
        # Action stacking for causality tracking
        # Stores last STACK_SIZE actions as one-hot vectors (each 8-dim)
        self.action_stack: deque = deque(maxlen=STACK_SIZE)

        # CPU Optimization: Preallocated arrays for _build_stacked_state()
        # Avoids repeated allocation every step
        self._frames_buffer = np.zeros((STACK_SIZE, RAW_STATE_DIM), dtype=np.float32)
        self._actions_buffer = np.zeros((STACK_SIZE, ACTION_DIM), dtype=np.float32)

        # EMA smoothed latency ratios for temporal smoothing
        self.lat_ema = {qid: 1.0 for qid in QIDS}
        
        # Traffic manager for dynamic profile changes (training only)
        # In production_mode, traffic is managed externally by ProductionRunner
        self.traffic_manager = (
            TrafficManager(config_path=self._config_path, seed=traffic_seed)
            if (reset_network and not production_mode) else None
        )
        
        # Current traffic profile for logging
        self.current_traffic_profile = ""  # e.g., "light_1", "medium_2", "bursty_be_1"
        self.current_traffic_category = ""  # "light", "medium", "high", "bursty"
        self.traffic_category_weights = None  # Optional: {'light': 0.1, 'medium': 0.2, 'high': 0.3, 'bursty': 0.4}
        self.traffic_profile_weights = None   # Optional: {'light_1': 1.0, 'bursty_vo_1': 2.0, ...}
        self.fixed_traffic_profile = None    # Optional: override to use specific profile for all episodes
        self._required_qids_cache = tuple(QIDS)
        self.telemetry_liveness_enabled = True
        self.telemetry_liveness_retries = TELEMETRY_LIVENESS_RETRIES
        self.telemetry_liveness_interval_seconds = TELEMETRY_LIVENESS_INTERVAL_SECONDS
        self.telemetry_liveness_restarts = TELEMETRY_LIVENESS_RESTARTS
        self.telemetry_liveness_window_seconds = WINDOW_SECONDS
        self._telemetry_epoch_id = None
        self._telemetry_epoch_ns = None
        self._last_telemetry_freshness = None
        self._episode_start_telemetry_valid = True

        # CPU Optimization: Persistent ThreadPoolExecutor for parallel InfluxDB queries
        # Avoids thread creation/destruction overhead on every _collect_snapshot() call
        # 5 workers: 2 main queries + up to 3 parallel retries (lat, drop, util)
        self._query_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="influx_query")

    def _required_qids_for_current_profile(self) -> Tuple[int, ...]:
        """Queues that must have valid telemetry for the active workload."""
        profile = self.current_traffic_profile or ""
        if profile.startswith("bursty_vo_"):
            return (0,)
        if profile.startswith("bursty_vi_"):
            return (1,)
        if profile.startswith("bursty_be_"):
            return (7,)
        return tuple(QIDS)

    def _normalize_optional_queue_telemetry(self, snapshot: Dict[int, Dict]) -> None:
        """Keep non-focused burst queues from poisoning state/reward when idle."""
        required = set(self._required_qids_for_current_profile())
        for qid in QIDS:
            q = snapshot[qid]
            q["profile_required"] = qid in required
            q["telemetry_optional"] = qid not in required
            if qid in required or q.get("data_valid", False):
                continue

            # Class-focused burst profiles intentionally starve the other two
            # queues during low/off phases. Treat missing optional telemetry as
            # neutral state, but do not invent routing context or alternatives.
            q["lat_p95"] = SLA_THRESHOLDS[qid]
            q["drop_p95"] = 0.0
            q["util_p95"] = 0.0
            q["bottleneck_sid"] = None
            q["bottleneck_score"] = 0.0
            q["bottleneck_drop"] = 0.0
            q["bottleneck_lat"] = 0.0
            q["bottleneck_util"] = 0.0
            q["bottleneck_role"] = "optional"
            q["alternatives"] = []
            q["alt_exists"] = False

    def _required_valid_count(self, snapshot: Dict[int, Dict]) -> int:
        required = self._required_qids_for_current_profile()
        return sum(
            1 for qid in required
            if snapshot[qid].get("data_valid", False)
        )

    def _required_invalid_queues(self, snapshot: Dict[int, Dict]) -> List[int]:
        required = self._required_qids_for_current_profile()
        return [
            qid for qid in required
            if not snapshot[qid].get("data_valid", False)
        ]

    def _min_required_valid_count(self) -> int:
        required_count = len(self._required_qids_for_current_profile())
        return required_count if required_count <= 2 else 2

    def _program_baseline_routing(self) -> None:
        """Clear P4 state and restore the OSPF baseline."""
        self.controller.clear_all_tables(verify=True)
        log.info("Cleared all P4 tables")
        self.controller.compute_forwarding_entries()
        log.info("Recomputed OSPF forwarding entries")
        self.controller.program_switches()
        log.info("Programmed switches with baseline routing")

    def _reset_local_telemetry_epoch(self, reason: str) -> None:
        """Clear only the live in-memory cache between traffic profiles."""
        if not self._read_from_cache():
            return
        try:
            response = self._cache_request({
                "kind": "reset_epoch",
                "clear": True,
                "reason": reason,
            })
        except Exception as exc:
            log.warning(f"[Telemetry Gate] Local cache epoch reset failed: {exc}")
            return
        if not response:
            return
        self._telemetry_epoch_id = response.get("epoch_id")
        self._telemetry_epoch_ns = response.get("epoch_started_ns")
        log.info(
            "[Telemetry Gate] Reset live cache epoch "
            f"{self._telemetry_epoch_id} for {reason}"
        )

    def _cache_freshness(
        self,
        *,
        qids: Optional[List[int]] = None,
        start: Optional[str] = None,
        stop: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._read_from_cache():
            return None
        if start is None or stop is None:
            stop_dt = datetime.utcnow() - timedelta(milliseconds=SAFETY_LAG_MS)
            seconds = (
                float(window_seconds)
                if window_seconds is not None
                else float(self.telemetry_liveness_window_seconds)
            )
            start_dt = stop_dt - timedelta(seconds=max(WINDOW_SECONDS, seconds))
            start = start_dt.isoformat() + 'Z'
            stop = stop_dt.isoformat() + 'Z'
        target_qids = list(qids) if qids is not None else list(self._required_qids_for_current_profile())
        try:
            return self._cache_request({
                "kind": "freshness",
                "start_ns": iso_to_ns(start),
                "stop_ns": iso_to_ns(stop),
                "qids": target_qids,
                "required_labels": ["lat", "drop", "util"],
            })
        except Exception as exc:
            log.warning(f"[Telemetry Freshness] local cache freshness failed: {exc}")
            return None

    @staticmethod
    def _ns_to_iso(ns_value: Any) -> str:
        if ns_value is None:
            return "none"
        try:
            return datetime.utcfromtimestamp(int(ns_value) / 1_000_000_000).isoformat() + "Z"
        except Exception:
            return str(ns_value)

    def _freshness_missing_required(
        self,
        report: Optional[Dict[str, Any]],
        required_qids: Optional[List[int]] = None,
    ) -> Dict[int, List[str]]:
        if not report:
            return {qid: ["freshness_query_failed"] for qid in (required_qids or [])}
        required = [int(qid) for qid in (required_qids or self._required_qids_for_current_profile())]
        missing_raw = report.get("missing", {}) or {}
        missing = {}
        for qid in required:
            labels = missing_raw.get(str(qid), [])
            if labels:
                missing[qid] = list(labels)
        return missing

    def _log_freshness_report(
        self,
        report: Optional[Dict[str, Any]],
        required_qids: Optional[List[int]] = None,
        *,
        prefix: str,
        level: int = logging.WARNING,
    ) -> None:
        if not report:
            log.log(level, f"{prefix} freshness unavailable")
            return
        required = [int(qid) for qid in (required_qids or self._required_qids_for_current_profile())]
        window = report.get("window", {}) or {}
        log.log(
            level,
            f"{prefix} window={self._ns_to_iso(window.get('start_ns'))}.."
            f"{self._ns_to_iso(window.get('stop_ns'))} "
            f"epoch={report.get('epoch_id')} complete={report.get('complete')} "
            f"missing={report.get('missing', {})}"
        )
        queues = report.get("queues", {}) or {}
        for qid in required:
            pieces = []
            by_label = queues.get(str(qid), {}) or {}
            for label in ("lat", "drop", "util"):
                item = by_label.get(label, {}) or {}
                pieces.append(
                    f"{label}:count={item.get('window_count', 0)} "
                    f"win_latest={self._ns_to_iso(item.get('window_latest_ns'))} "
                    f"cache_latest={self._ns_to_iso(item.get('cache_latest_ns'))}"
                )
            log.log(level, f"{prefix} Q{qid} " + "; ".join(pieces))

    def _wait_for_required_telemetry(self) -> bool:
        """Gate episode start on fresh cache records for required queues."""
        if not self.telemetry_liveness_enabled or not self._read_from_cache():
            self._episode_start_telemetry_valid = True
            return True

        required_qids = list(self._required_qids_for_current_profile())
        retries = max(1, int(self.telemetry_liveness_retries))
        interval = max(0.0, float(self.telemetry_liveness_interval_seconds))
        for attempt in range(1, retries + 1):
            report = self._cache_freshness(
                qids=required_qids,
                window_seconds=self.telemetry_liveness_window_seconds,
            )
            self._last_telemetry_freshness = report
            missing = self._freshness_missing_required(report, required_qids)
            if not missing:
                self._episode_start_telemetry_valid = True
                self._log_freshness_report(
                    report,
                    required_qids,
                    prefix=f"[Telemetry Gate] ready attempt {attempt}/{retries}",
                    level=logging.INFO,
                )
                return True
            self._episode_start_telemetry_valid = False
            self._log_freshness_report(
                report,
                required_qids,
                prefix=f"[Telemetry Gate] waiting attempt {attempt}/{retries}",
                level=logging.WARNING,
            )
            if attempt < retries and interval > 0:
                time.sleep(interval)

        log.warning(
            "[Telemetry Gate] Required telemetry did not become fresh for "
            f"Q{required_qids} after {retries} attempts"
        )
        return False

    def reset(self, force_reset: Optional[bool] = None,
              cooldown_seconds: Optional[float] = None,
              collect_initial_snapshot: bool = True) -> np.ndarray:
        """Reset episode and return initial stacked state.
        
        Args:
            force_reset: Override reset behavior for this episode.
                - None: use self.reset_network default
                - True: force baseline reset (clear tables, reprogram OSPF)
                - False: warm-start from current state
            cooldown_seconds: Optional explicit stabilization interval. When
                omitted, preserves the training defaults (5s after a baseline
                reset, 2s after a warm start). Benchmark runners set the same
                value for RL, ECMP, and OSPF.
            collect_initial_snapshot: When False, reset counters/stacks without
                issuing the initial InfluxDB telemetry query. Baseline runners
                discard the returned state and use this to avoid holding one
                traffic stage for several query-retry seconds before measured
                step 1.
        
        Curriculum-based training can use this to mix baseline-starts and warm-starts:
        - Early training: mostly baseline-starts (agent learns from clean state)
        - Late training: mostly warm-starts (agent learns stability/recovery)
        """
        # Determine whether to reset this episode
        do_reset = force_reset if force_reset is not None else self.reset_network
        
        # Perform full network reset if requested
        if do_reset:
            log.info("=== BASELINE START: Resetting network to OSPF ===")
            self._program_baseline_routing()
        else:
            log.info("=== WARM START: Continuing from current routing state ===")

        profile_retry_count = 0
        profile_override = None
        baseline_recovery_used = bool(do_reset)
        while True:
            # Start traffic for new episode (training mode only)
            if self.traffic_manager:
                if self.fixed_traffic_profile:
                    profile_name = self.fixed_traffic_profile
                    log.info(f"Starting FIXED traffic profile: {profile_name}")
                    profile_info = self.traffic_manager.start_traffic(profile_name=profile_name)
                elif profile_override:
                    log.info(
                        f"Restarting same traffic profile after telemetry gate "
                        f"failure: {profile_override}"
                    )
                    profile_info = self.traffic_manager.start_traffic(profile_name=profile_override)
                else:
                    log.info("Starting randomized traffic profile...")
                    profile_info = self.traffic_manager.start_traffic(
                        category_weights=self.traffic_category_weights,
                        profile_weights=self.traffic_profile_weights,
                    )

                self.current_traffic_profile = profile_info['profile_name']
                self.current_traffic_category = profile_info['profile_category']
                self._reset_local_telemetry_epoch(
                    f"profile_start:{self.current_traffic_profile}"
                )
                log.info(f"Traffic started: {self.current_traffic_profile} ({self.current_traffic_category})")
                log.info(
                    "Required telemetry queues for this profile: "
                    f"Q{list(self._required_qids_for_current_profile())}"
                )

            # Cool-down period for traffic to stabilize (traffic is now running)
            cooldown = (
                float(cooldown_seconds)
                if cooldown_seconds is not None
                else (5.0 if do_reset else 2.0)
            )
            log.info(f"Traffic running, waiting {cooldown}s for metrics to stabilize...")
            if self.traffic_manager:
                self.traffic_manager.warm_profile_for(cooldown)
                self.traffic_manager.begin_measurement()
            else:
                time.sleep(cooldown)

            if not collect_initial_snapshot or self._wait_for_required_telemetry():
                break

            if (
                self.traffic_manager
                and profile_retry_count < max(0, int(self.telemetry_liveness_restarts))
            ):
                profile_retry_count += 1
                profile_override = self.current_traffic_profile
                log.warning(
                    "[Telemetry Gate] Restarting and re-warming profile "
                    f"{profile_override} ({profile_retry_count}/"
                    f"{self.telemetry_liveness_restarts}) before episode start"
                )
                continue

            if self.traffic_manager and not baseline_recovery_used:
                profile_override = self.current_traffic_profile
                log.warning(
                    "[Telemetry Gate] Forcing baseline reset before retrying "
                    f"profile {profile_override}"
                )
                log.info("=== BASELINE START: Resetting network to OSPF ===")
                self._program_baseline_routing()
                do_reset = True
                baseline_recovery_used = True
                profile_retry_count = 0
                continue

            log.warning(
                "[Telemetry Gate] Proceeding with episode start without fresh "
                "required telemetry; learning will remain disabled until "
                "data_valid=True"
            )
            break

        log.info("Episode start complete")
        
        preserve_action_history = (
            not do_reset
            and self.action_stack
            and len(self.action_stack) == STACK_SIZE
        )

        # Reset episode counters
        self.episode_step = 0
        self.sla_streak = 0
        if do_reset:
            self.last_action = 0
            self.last_action_time = time.monotonic()
            self._last_action_global_step = self.global_step
            self._last_action_step = 0  # legacy/debug compatibility
        
        # Reset EMA state for the new traffic profile; warm-starts preserve
        # routing/action context, but not stale latency trends from a prior
        # workload profile.
        self.lat_ema = {qid: 1.0 for qid in QIDS}

        # Collect initial snapshot (after reset if applicable). Some baseline
        # runners discard the returned state; for them, skipping this query
        # prevents the initial telemetry retry path from becoming unintended
        # traffic preconditioning.
        if collect_initial_snapshot:
            self.last_snapshot = self._collect_snapshot()
            raw_state = self._build_raw_state(self.last_snapshot)
        else:
            self.last_snapshot = None
            raw_state = np.zeros(RAW_STATE_DIM, dtype=np.float32)
        
        # Debug assertion to catch dimension mismatches early
        assert len(raw_state) == RAW_STATE_DIM, f"State dim mismatch: {len(raw_state)} != {RAW_STATE_DIM}"
        
        if preserve_action_history:
            # Warm-start: routing state persists, so keep recent action history
            # while refreshing observation frames for the new traffic profile.
            self.frame_stack.clear()
            for _ in range(STACK_SIZE):
                self.frame_stack.append(raw_state.copy())
        else:
            # Baseline reset or first episode: clean slate.
            self._init_stacks(raw_state)

        return self._build_stacked_state()

    def clear_stacks(self):
        """Clear frame and action stacks without full environment reset.

        PHASE 2.3: Used when detecting model reload in production to prevent
        stale historical frames/actions from contaminating new model's state.

        This method:
        1. Collects fresh snapshot from current network state
        2. Rebuilds frame_stack with current observation replicated
        3. Resets action_stack to no-ops

        Call this when you detect a new model has been loaded in production.
        """
        log.info("[STACK CLEAR] Clearing frame and action stacks for model reload")

        # Collect fresh snapshot and reinitialize stacks
        snapshot = self._collect_snapshot()
        raw_state = self._build_raw_state(snapshot)
        self._init_stacks(raw_state)

        log.info("[STACK CLEAR] Stacks cleared - frame and action history reset")

    def _influx_query_with_retry(self, flux: str, max_retries: int = 3) -> Optional[list]:
        """Execute InfluxDB query with exponential backoff retry.

        PHASE 2.4: Add retry logic with exponential backoff to handle transient failures.
        With 2s timeout, retries at 100ms, 200ms, 400ms give fast recovery.

        Args:
            flux: Flux query string
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Query result tables, or None if all retries failed
        """
        if self.query_api is None:
            log.warning("InfluxDB query requested but InfluxDB client is unavailable")
            return None

        backoff_delays = [0.1, 0.2, 0.4]  # 100ms, 200ms, 400ms
        start_time = time.monotonic()

        for attempt in range(max_retries):
            try:
                result = self.query_api.query(org=self.org, query=flux)
                elapsed_ms = (time.monotonic() - start_time) * 1000
                if elapsed_ms > 500:
                    log.warning(f"[Query Slow] {elapsed_ms:.0f}ms total ({attempt + 1} attempts)")
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = backoff_delays[attempt]
                    log.warning(f"Query failed (attempt {attempt + 1}/{max_retries}), "
                              f"retrying in {delay*1000:.0f}ms: {e}")
                    time.sleep(delay)
                else:
                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    log.error(f"Query failed after {max_retries} attempts ({elapsed_ms:.0f}ms): {e}")
                    return None

    def _read_from_cache(self) -> bool:
        return self.telemetry_cache is not None

    def _allow_influx_fallback(self) -> bool:
        return self.telemetry_backend in ("influx", "cache-fallback-influx")

    def _cache_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.telemetry_cache is None:
            return None
        response = self.telemetry_cache.request(payload)
        self._validate_cache_window(payload, response)
        return response

    def _validate_cache_window(self, payload: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Fail fast if the local cache returns records outside the requested window."""
        window = response.get("window")
        if not window:
            return

        expected_start = int(payload["start_ns"])
        expected_stop = int(payload["stop_ns"])
        kind = payload.get("kind", "unknown")
        errors = []

        actual_start = int(window.get("start_ns", -1))
        actual_stop = int(window.get("stop_ns", -1))
        if actual_start != expected_start or actual_stop != expected_stop:
            errors.append(
                f"request window mismatch expected=[{expected_start},{expected_stop}) "
                f"actual=[{actual_start},{actual_stop})"
            )

        min_ns = window.get("min_ns")
        max_ns = window.get("max_ns")
        if min_ns is not None and int(min_ns) < expected_start:
            errors.append(f"min_ns {min_ns} precedes start_ns {expected_start}")
        if max_ns is not None and int(max_ns) >= expected_stop:
            errors.append(f"max_ns {max_ns} reaches/exceeds stop_ns {expected_stop}")
        if not bool(window.get("within_window", True)):
            errors.append("aggregate window audit reports out-of-window records")

        for measurement, stat in (window.get("by_measurement") or {}).items():
            stat_min = stat.get("min_ns")
            stat_max = stat.get("max_ns")
            if stat_min is not None and int(stat_min) < expected_start:
                errors.append(f"{measurement} min_ns {stat_min} precedes start_ns {expected_start}")
            if stat_max is not None and int(stat_max) >= expected_stop:
                errors.append(f"{measurement} max_ns {stat_max} reaches/exceeds stop_ns {expected_stop}")
            if not bool(stat.get("within_window", True)):
                errors.append(f"{measurement} audit reports out-of-window records")

        now_ns = time.time_ns()
        if expected_stop > now_ns + 50_000_000:
            errors.append(
                f"requested stop_ns {expected_stop} is more than 50ms in the future "
                f"relative to RL clock {now_ns}"
            )

        if errors:
            raise RuntimeError(
                f"local telemetry cache window audit failed for {kind}: "
                + "; ".join(errors)
            )

        count = int(window.get("count", 0) or 0)
        log.debug(
            "[Telemetry Window] kind=%s count=%d start_ns=%d stop_ns=%d "
            "min_ns=%s max_ns=%s stop_age_ms=%.1f",
            kind,
            count,
            expected_start,
            expected_stop,
            min_ns,
            max_ns,
            (now_ns - expected_stop) / 1_000_000.0,
        )

    def _cache_queue_metrics(self, start: str, stop: str) -> Dict:
        response = self._cache_request({
            "kind": "queue_metrics",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "qids": QIDS,
        })
        result = {
            'metrics': {qid: {'lat_p95': None, 'drop_p95': None, 'util_p95': None} for qid in QIDS},
            'metrics_received': {qid: {'lat': False, 'drop': False, 'util': False} for qid in QIDS},
            'counts': {qid: {'lat': 0, 'drop': 0, 'util': 0} for qid in QIDS},
            'window': None,
            'recovered_via_retry': set(),
        }
        if response is None:
            return result

        metrics = response.get("metrics", {})
        received = response.get("metrics_received", {})
        counts = response.get("counts", {})
        result['window'] = response.get("window")
        for qid in QIDS:
            key = str(qid)
            src_metrics = metrics.get(key, {})
            src_received = received.get(key, {})
            src_counts = counts.get(key, {})
            for metric_name in ('lat_p95', 'drop_p95', 'util_p95'):
                value = src_metrics.get(metric_name)
                if value is not None:
                    result['metrics'][qid][metric_name] = float(value)
            for received_name in ('lat', 'drop', 'util'):
                result['metrics_received'][qid][received_name] = bool(src_received.get(received_name, False))
                result['counts'][qid][received_name] = int(src_counts.get(received_name, 0) or 0)
        return result

    def _cache_metric_query(self, start: str, stop: str, target_queues: List[int],
                            measurement: str, metric_name: str) -> Dict[int, float]:
        response = self._cache_request({
            "kind": "queue_metrics",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "qids": target_queues,
        })
        if response is None:
            return {}

        field_by_measurement = {
            "flow_latency": "lat_p95",
            "q_drop_rate_100ms": "drop_p95",
            "tx_utilization": "util_p95",
        }
        field = field_by_measurement[measurement]
        metrics = response.get("metrics", {})
        recovered = {}
        for qid in target_queues:
            value = metrics.get(str(qid), {}).get(field)
            if value is not None:
                recovered[qid] = float(value)
        return recovered

    def _cache_queue_summary(
        self,
        start: str,
        stop: str,
        target_queues: List[int],
    ) -> Tuple[Dict, Dict[int, Tuple[str, str]]]:
        response = self._cache_request({
            "kind": "queue_summary",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "qids": target_queues,
        })
        result = {
            'metrics': {qid: {'lat_p95': None, 'drop_p95': None, 'util_p95': None} for qid in QIDS},
            'metrics_received': {qid: {'lat': False, 'drop': False, 'util': False} for qid in QIDS},
            'counts': {qid: {'lat': 0, 'drop': 0, 'util': 0} for qid in QIDS},
            'window': None,
            'recovered_via_retry': set(),
        }
        demands: Dict[int, Tuple[str, str]] = {}
        if response is None:
            return result, demands

        metrics = response.get("metrics", {})
        received = response.get("metrics_received", {})
        counts = response.get("counts", {})
        result['window'] = response.get("window")
        for qid in QIDS:
            key = str(qid)
            src_metrics = metrics.get(key, {})
            src_received = received.get(key, {})
            src_counts = counts.get(key, {})
            for metric_name in ('lat_p95', 'drop_p95', 'util_p95'):
                value = src_metrics.get(metric_name)
                if value is not None:
                    result['metrics'][qid][metric_name] = float(value)
            for received_name in ('lat', 'drop', 'util'):
                result['metrics_received'][qid][received_name] = bool(
                    src_received.get(received_name, False)
                )
                result['counts'][qid][received_name] = int(
                    src_counts.get(received_name, 0) or 0
                )

        for qid_text, item in response.get("demands", {}).items():
            try:
                qid = int(qid_text)
            except (TypeError, ValueError):
                continue
            src = item.get("src_ip")
            dst = item.get("dst_ip")
            if qid in target_queues and src and dst:
                demands[qid] = (str(src), str(dst))

        return result, demands

    def _cache_hot_demands(self, start: str, stop: str, target_queues: List[int]) -> Dict[int, Tuple[str, str]]:
        response = self._cache_request({
            "kind": "hot_demands",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "qids": target_queues,
        })
        if response is None:
            return {}
        demands = {}
        for qid_text, item in response.get("demands", {}).items():
            try:
                qid = int(qid_text)
            except (TypeError, ValueError):
                continue
            src = item.get("src_ip")
            dst = item.get("dst_ip")
            if qid in target_queues and src and dst:
                demands[qid] = (str(src), str(dst))
        return demands

    def _cache_switch_metrics_for_queue(self, start: str, stop: str,
                                        sw_ids: List[int], qid: int) -> Dict[int, Dict]:
        response = self._cache_request({
            "kind": "switch_metrics",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "switch_ids": sw_ids,
            "qid": qid,
        })
        if response is None:
            return {}
        results = {}
        for sid_text, metrics in response.get("switch_metrics", {}).items():
            try:
                sid = int(sid_text)
            except (TypeError, ValueError):
                continue
            results[sid] = {
                'drop': float(metrics.get('drop', 0) or 0),
                'lat': float(metrics.get('lat', 0) or 0),
                'util': float(metrics.get('util', 0) or 0),
            }
        return results

    def _cache_switch_metrics_multi(
        self,
        start: str,
        stop: str,
        sw_ids_by_qid: Dict[int, List[int]],
    ) -> Dict[int, Dict[int, Dict]]:
        if not sw_ids_by_qid:
            return {}
        response = self._cache_request({
            "kind": "switch_metrics_multi",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "queries": {
                str(int(qid)): [int(sid) for sid in sw_ids]
                for qid, sw_ids in sw_ids_by_qid.items()
                if sw_ids
            },
        })
        if response is None:
            return {}
        parsed: Dict[int, Dict[int, Dict]] = {}
        for qid_text, by_sid in response.get("switch_metrics", {}).items():
            try:
                qid = int(qid_text)
            except (TypeError, ValueError):
                continue
            parsed[qid] = {}
            for sid_text, metrics in by_sid.items():
                try:
                    sid = int(sid_text)
                except (TypeError, ValueError):
                    continue
                parsed[qid][sid] = {
                    'drop': float(metrics.get('drop', 0) or 0),
                    'lat': float(metrics.get('lat', 0) or 0),
                    'util': float(metrics.get('util', 0) or 0),
                }
        return parsed

    def _cache_traffic_count(self, qid: int, start: str, stop: str) -> int:
        response = self._cache_request({
            "kind": "traffic_count",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "qid": qid,
        })
        if response is None:
            return 0
        return int(response.get("count", 0) or 0)

    def _cache_traffic_counts(self, qids: List[int], start: str, stop: str) -> Dict[int, int]:
        if not qids:
            return {}
        response = self._cache_request({
            "kind": "traffic_count_multi",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "qids": qids,
        })
        if response is None:
            return {int(qid): 0 for qid in qids}
        counts = response.get("counts", {})
        return {
            int(qid): int(counts.get(str(int(qid)), 0) or 0)
            for qid in qids
        }

    def _cache_flow_coverage(self, start: str, stop: str) -> Dict[int, set]:
        response = self._cache_request({
            "kind": "flow_coverage",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "qids": QIDS,
        })
        observed = {qid: set() for qid in QIDS}
        if response is None:
            return observed
        for qid_text, flow_ids in response.get("observed", {}).items():
            try:
                qid = int(qid_text)
            except (TypeError, ValueError):
                continue
            if qid in observed:
                observed[qid] = {str(flow_id) for flow_id in flow_ids}
        return observed

    def get_local_egress_observations(
        self,
        window_seconds: float = 5.0,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Return compact INT egress-path and bottleneck evidence from cache."""
        if not self._read_from_cache():
            return {
                "ok": False,
                "error": "local telemetry cache is not enabled",
                "flows": {},
                "top_egresses": [],
            }
        stop_dt = datetime.utcnow()
        start_dt = stop_dt - timedelta(seconds=float(window_seconds))
        start = start_dt.isoformat() + 'Z'
        stop = stop_dt.isoformat() + 'Z'
        return self._cache_request({
            "kind": "egress_observations",
            "start_ns": iso_to_ns(start),
            "stop_ns": iso_to_ns(stop),
            "qids": QIDS,
            "top_n": int(top_n),
        })

    def _time_window(self) -> Tuple[str, str]:
        """Get time window for queries."""
        stop_dt = datetime.utcnow() - timedelta(milliseconds=SAFETY_LAG_MS)
        start_dt = stop_dt - timedelta(seconds=WINDOW_SECONDS)
        return start_dt.isoformat() + 'Z', stop_dt.isoformat() + 'Z'

    def verify_telemetry_flow_coverage(
        self,
        expected_flow_ids,
        window_seconds: float = 5.0,
        retries: int = 12,
        retry_delay: float = 1.0,
        raise_on_error: bool = True,
    ) -> Dict:
        """Verify that every configured demand reports latency in every queue.

        Metric presence alone is insufficient for benchmark validity: a
        partially blackholed ECMP condition can look excellent when percentiles
        contain only surviving flows. This check requires exact flow-ID
        coverage for Q0/Q1/Q7 over a multi-second window. Retries allow the
        INT collector and InfluxDB writer to catch up after a clean traffic
        start without weakening the exact all-flow requirement.
        """
        expected = {str(int(flow_id)) for flow_id in expected_flow_ids}
        observed = {qid: set() for qid in QIDS}
        query_error = None

        for attempt in range(max(1, retries)):
            stop_dt = datetime.utcnow()
            start_dt = stop_dt - timedelta(seconds=float(window_seconds))
            start = start_dt.isoformat() + 'Z'
            stop = stop_dt.isoformat() + 'Z'
            flux = f'''
            from(bucket:"{self.bucket}")
                |> range(start:{start}, stop:{stop})
                |> filter(fn: (r) => r._measurement == "flow_latency")
                |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
                |> group(columns:["queue_id", "flow_id"])
                |> first()
            '''
            try:
                if self._read_from_cache():
                    observed = self._cache_flow_coverage(start, stop)
                else:
                    tables = self.query_api.query(org=self.org, query=flux)
                    observed = {qid: set() for qid in QIDS}
                    for table in tables or []:
                        for record in table.records:
                            try:
                                qid = int(record.values.get('queue_id', -1))
                            except (TypeError, ValueError):
                                continue
                            flow_id = record.values.get('flow_id')
                            if qid in observed and flow_id is not None:
                                observed[qid].add(str(flow_id))
                query_error = None
            except Exception as exc:
                query_error = str(exc)

            if query_error is None and all(
                observed[qid] == expected for qid in QIDS
            ):
                break
            if attempt < max(1, retries) - 1:
                time.sleep(retry_delay)

        per_queue = {}
        errors = []
        for qid in QIDS:
            missing = sorted(expected - observed[qid], key=int)
            unexpected = sorted(observed[qid] - expected, key=int)
            per_queue[qid] = {
                'expected_count': len(expected),
                'observed_count': len(observed[qid]),
                'missing_flow_ids': missing,
                'unexpected_flow_ids': unexpected,
            }
            if missing or unexpected:
                errors.append(
                    f"Q{qid} expected {len(expected)} flows, observed "
                    f"{len(observed[qid])}; missing={missing}; "
                    f"unexpected={unexpected}"
                )
        if query_error:
            source = "local cache" if self._read_from_cache() else "InfluxDB"
            errors.append(f"{source} coverage query failed: {query_error}")

        report = {
            'verified': not errors,
            'window_seconds': float(window_seconds),
            'expected_flow_ids': sorted(expected, key=int),
            'per_queue': per_queue,
            'errors': errors,
        }
        if errors and raise_on_error:
            raise RuntimeError(
                "Telemetry flow-coverage verification failed: "
                + "; ".join(errors)
            )
        return report

    def _query_aggregated_metrics(self, start: str, stop: str, step: int) -> Dict:
        """
        Query aggregated p95 metrics for all queues.
        Returns: Dict with 'tables' and 'metrics_received' keys.

        Args:
            start: Query window start time (ISO format)
            stop: Query window stop time (ISO format)
            step: Global step number for logging (captured at call time)
        """
        if self._read_from_cache():
            try:
                return self._cache_queue_metrics(start, stop)
            except Exception as e:
                log.warning(f"[Query S{step}] local telemetry cache queue_metrics failed: {e}")
                if not self._allow_influx_fallback():
                    return {
                        'metrics': {qid: {'lat_p95': None, 'drop_p95': None, 'util_p95': None} for qid in QIDS},
                        'metrics_received': {qid: {'lat': False, 'drop': False, 'util': False} for qid in QIDS},
                        'counts': {qid: {'lat': 0, 'drop': 0, 'util': 0} for qid in QIDS},
                        'window': None,
                        'recovered_via_retry': set(),
                    }
                log.warning(f"[Query S{step}] falling back to InfluxDB for queue metrics")

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

        result = {
            'metrics': {qid: {'lat_p95': None, 'drop_p95': None, 'util_p95': None} for qid in QIDS},
            'metrics_received': {qid: {'lat': False, 'drop': False, 'util': False} for qid in QIDS}
        }

        tables = self._influx_query_with_retry(flux)
        if tables is not None:
            for table in tables:
                for record in table.records:
                    try:
                        qid = int(record.values.get('queue_id', -1))
                        if qid not in QIDS:
                            continue
                        measurement = record.get_measurement()
                        value = record.get_value()
                        if value is not None:
                            result['metrics'][qid][measurement] = float(value)
                            if measurement == 'lat_p95':
                                result['metrics_received'][qid]['lat'] = True
                            elif measurement == 'drop_p95':
                                result['metrics_received'][qid]['drop'] = True
                            elif measurement == 'util_p95':
                                result['metrics_received'][qid]['util'] = True
                    except (ValueError, TypeError):
                        continue

        # Check if any queue is missing metrics and trigger retries IN PARALLEL
        result['recovered_via_retry'] = set()  # Track queues recovered via retry

        missing_lat_queues = [qid for qid in QIDS if not result['metrics_received'][qid]['lat']]
        missing_drop_queues = [qid for qid in QIDS if not result['metrics_received'][qid]['drop']]
        missing_util_queues = [qid for qid in QIDS if not result['metrics_received'][qid]['util']]

        # Submit all retries in parallel (if needed)
        retry_futures = {}
        if missing_lat_queues:
            log.warning(f"[Query S{step}] flow_latency missing for Q{missing_lat_queues}, triggering retry")
            retry_futures['lat'] = self._query_executor.submit(
                self._retry_metric_query, start, stop, missing_lat_queues, "flow_latency", "flow_latency", step)
        if missing_drop_queues:
            log.warning(f"[Query S{step}] drop_rate missing for Q{missing_drop_queues}, triggering retry")
            retry_futures['drop'] = self._query_executor.submit(
                self._retry_metric_query, start, stop, missing_drop_queues, "q_drop_rate_100ms", "drop", step)
        if missing_util_queues:
            log.warning(f"[Query S{step}] util missing for Q{missing_util_queues}, triggering retry")
            retry_futures['util'] = self._query_executor.submit(
                self._retry_metric_query, start, stop, missing_util_queues, "tx_utilization", "util", step)

        # Collect results from parallel retries (timeout covers all retries: 5s max + 1s buffer)
        retry_timeout = 6.0
        if 'lat' in retry_futures:
            try:
                recovered = retry_futures['lat'].result(timeout=retry_timeout)
                for qid, value in recovered.items():
                    result['metrics'][qid]['lat_p95'] = value
                    result['metrics_received'][qid]['lat'] = True
                    result['recovered_via_retry'].add(qid)
                    log.info(f"[Query S{step}] Recovered lat_p95={value:.2f}ms for Q{qid}")
            except Exception as e:
                log.warning(f"[Query S{step}] lat retry failed: {e}")

        if 'drop' in retry_futures:
            try:
                recovered = retry_futures['drop'].result(timeout=retry_timeout)
                for qid, value in recovered.items():
                    result['metrics'][qid]['drop_p95'] = value
                    result['metrics_received'][qid]['drop'] = True
                    result['recovered_via_retry'].add(qid)
                    log.info(f"[Query S{step}] Recovered drop_p95={value:.4f} for Q{qid}")
            except Exception as e:
                log.warning(f"[Query S{step}] drop retry failed: {e}")

        if 'util' in retry_futures:
            try:
                recovered = retry_futures['util'].result(timeout=retry_timeout)
                for qid, value in recovered.items():
                    result['metrics'][qid]['util_p95'] = value
                    result['metrics_received'][qid]['util'] = True
                    result['recovered_via_retry'].add(qid)
                    log.info(f"[Query S{step}] Recovered util_p95={value:.2f}% for Q{qid}")
            except Exception as e:
                log.warning(f"[Query S{step}] util retry failed: {e}")

        return result

    def _retry_metric_query(self, start: str, stop: str, target_queues: List[int],
                             measurement: str, metric_name: str, step: int) -> Dict[int, float]:
        """
        Retry a metric query with up to 5 attempts, 1 second delay between each.
        Max wait time: 5 seconds before declaring metric missing.

        On each retry, the query window expands by the cumulative wait time to catch
        data that was written during the delay.

        Continues retrying until ALL target_queues are recovered or retries exhausted.

        Args:
            start: Query window start time (ISO format)
            stop: Query window stop time (ISO format)
            target_queues: List of queue IDs that need to be recovered
            measurement: InfluxDB measurement name (e.g., "flow_latency", "q_drop_rate_100ms", "tx_utilization")
            metric_name: Human-readable name for logging (e.g., "flow_latency", "drop", "util")
            step: Global step number for logging (passed from caller)

        Returns: Dict mapping queue_id -> p95 value
        """
        result = {}
        remaining_queues = set(target_queues)
        max_retries = 5
        retry_delay = 1.0  # 1 second

        for attempt in range(max_retries):
            if attempt > 0:
                time.sleep(retry_delay)

            # Expand stop time by cumulative wait (attempt * retry_delay seconds)
            # This catches data written during our retry delays
            window_extension = attempt * retry_delay
            if window_extension > 0:
                stop_dt = datetime.fromisoformat(stop.rstrip('Z')) + timedelta(seconds=window_extension)
                current_stop = stop_dt.isoformat() + 'Z'
            else:
                current_stop = stop

            flux = f'''
            from(bucket:"{self.bucket}")
                |> range(start:{start}, stop:{current_stop})
                |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> toFloat()
                |> group(columns:["queue_id"])
                |> quantile(q:0.95, method:"estimate_tdigest")
            '''

            t0 = time.time()
            try:
                if self._read_from_cache():
                    cache_values = self._cache_metric_query(
                        start,
                        current_stop,
                        list(remaining_queues),
                        measurement,
                        metric_name,
                    )
                    elapsed_ms = (time.time() - t0) * 1000
                    for qid, value in cache_values.items():
                        if qid in remaining_queues:
                            result[qid] = float(value)
                            remaining_queues.discard(qid)
                else:
                    tables = self.query_api.query(org=self.org, query=flux)
                    elapsed_ms = (time.time() - t0) * 1000

                    if tables:
                        for table in tables:
                            for record in table.records:
                                try:
                                    qid = int(record.values.get('queue_id', -1))
                                    value = record.get_value()
                                    if qid in remaining_queues and value is not None:
                                        result[qid] = float(value)
                                        remaining_queues.discard(qid)
                                except (ValueError, TypeError):
                                    continue

                window_info = f"+{window_extension:.0f}s window" if window_extension > 0 else ""
                if not remaining_queues:
                    # All target queues recovered
                    log.info(f"[Query S{step}] {metric_name} retry {attempt+1}/{max_retries} SUCCESS: "
                            f"got ALL Q{list(result.keys())} in {elapsed_ms:.0f}ms {window_info}")
                    return result
                elif result:
                    # Some queues recovered, but not all - continue retrying
                    log.info(f"[Query S{step}] {metric_name} retry {attempt+1}/{max_retries} PARTIAL: "
                            f"got Q{list(result.keys())}, still missing Q{list(remaining_queues)} "
                            f"({elapsed_ms:.0f}ms {window_info})")
                else:
                    log.warning(f"[Query S{step}] {metric_name} retry {attempt+1}/{max_retries}: "
                               f"no data ({elapsed_ms:.0f}ms, {window_info})")

            except Exception as e:
                elapsed_ms = (time.time() - t0) * 1000
                if self._read_from_cache() and self._allow_influx_fallback():
                    log.warning(f"[Query S{step}] {metric_name} local cache retry failed "
                                f"({elapsed_ms:.0f}ms), trying InfluxDB: {e}")
                    try:
                        if self.query_api is None:
                            raise RuntimeError("InfluxDB client is unavailable")
                        tables = self.query_api.query(org=self.org, query=flux)
                        elapsed_ms = (time.time() - t0) * 1000
                        if tables:
                            for table in tables:
                                for record in table.records:
                                    try:
                                        qid = int(record.values.get('queue_id', -1))
                                        value = record.get_value()
                                        if qid in remaining_queues and value is not None:
                                            result[qid] = float(value)
                                            remaining_queues.discard(qid)
                                    except (ValueError, TypeError):
                                        continue
                    except Exception as influx_exc:
                        log.warning(f"[Query S{step}] {metric_name} retry {attempt+1}/{max_retries} FAILED "
                                    f"({elapsed_ms:.0f}ms): {influx_exc}")
                    else:
                        window_info = f"+{window_extension:.0f}s window" if window_extension > 0 else ""
                        if not remaining_queues:
                            log.info(f"[Query S{step}] {metric_name} retry {attempt+1}/{max_retries} SUCCESS: "
                                    f"got ALL Q{list(result.keys())} in {elapsed_ms:.0f}ms {window_info}")
                            return result
                        if result:
                            log.info(f"[Query S{step}] {metric_name} retry {attempt+1}/{max_retries} PARTIAL: "
                                    f"got Q{list(result.keys())}, still missing Q{list(remaining_queues)} "
                                    f"({elapsed_ms:.0f}ms {window_info})")
                else:
                    log.warning(f"[Query S{step}] {metric_name} retry {attempt+1}/{max_retries} FAILED "
                               f"({elapsed_ms:.0f}ms): {e}")

        if result:
            log.warning(f"[Query S{step}] {metric_name} retry exhausted after {max_retries} attempts - "
                       f"recovered Q{list(result.keys())}, missing Q{list(remaining_queues)}")
        else:
            log.warning(f"[Query S{step}] {metric_name} retry exhausted after {max_retries} attempts (5s max)")
        return result

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
        # data_valid starts False, set True when we get real telemetry data
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
            'data_valid': False,  # Track telemetry validity
            'telemetry_counts': {'lat': 0, 'drop': 0, 'util': 0},
        } for qid in QIDS}

        # CPU Optimization: Run two independent queries in parallel using persistent ThreadPoolExecutor
        # This reduces total query time from ~1.5-3s (sequential) to ~0.5-1s (parallel)
        # Using persistent executor avoids thread creation/destruction overhead per step
        aggregated_result = None
        all_hot_demands = {}

        # Capture step number NOW before submitting to thread pool
        # This ensures consistent step logging even if retries run after step increments
        step = self.global_step
        hot_demand_qids = list(self._required_qids_for_current_profile())

        # ThreadPoolExecutor health check
        queue_size = self._query_executor._work_queue.qsize()
        if queue_size > 2:
            log.warning(f"[ThreadPool] Work queue backlog: {queue_size} (expected ~0)")

        if self._read_from_cache():
            try:
                aggregated_result, all_hot_demands = self._cache_queue_summary(
                    start,
                    stop,
                    hot_demand_qids,
                )
            except Exception as e:
                log.warning(f"[Query S{step}] local telemetry cache queue_summary failed: {e}")
                if self._allow_influx_fallback():
                    log.warning(f"[Query S{step}] falling back to InfluxDB for queue summary")
                else:
                    aggregated_result = {
                        'metrics': {
                            qid: {'lat_p95': None, 'drop_p95': None, 'util_p95': None}
                            for qid in QIDS
                        },
                        'metrics_received': {
                            qid: {'lat': False, 'drop': False, 'util': False}
                            for qid in QIDS
                        },
                        'counts': {
                            qid: {'lat': 0, 'drop': 0, 'util': 0}
                            for qid in QIDS
                        },
                        'window': None,
                        'recovered_via_retry': set(),
                    }
                    all_hot_demands = {}

        if aggregated_result is None:
            # InfluxDB path: keep the existing retry behavior as the optional
            # compatibility implementation.
            future_aggregated = self._query_executor.submit(
                self._query_aggregated_metrics,
                start,
                stop,
                step,
            )
            future_hottest = self._query_executor.submit(
                self._get_all_hottest_demands,
                step,
                hot_demand_qids,
            )

            # Timeout covers: initial query (~0.5s) + parallel retries (5s max)
            # + buffer (1.5s). This is intentionally only used off the local
            # cache fast path, where exact-window misses stay invalid.
            try:
                aggregated_result = future_aggregated.result(timeout=7.0)
            except Exception as e:
                log.warning(f"[Parallel Query] Aggregated metrics query failed: {e}")

            # hot_demands has its own retry loop (5s max) + buffer (1s) = 6s
            try:
                all_hot_demands = future_hottest.result(timeout=6.0)
            except Exception as e:
                log.warning(f"[Parallel Query] Hottest demands query failed: {e}")

        # Process aggregated metrics result
        metrics_received = {qid: {'lat': False, 'drop': False, 'util': False} for qid in QIDS}
        metrics_counts = {qid: {'lat': 0, 'drop': 0, 'util': 0} for qid in QIDS}
        metrics_window = None
        recovered_via_retry = set()  # Track queues that were recovered via retry
        if aggregated_result is not None:
            recovered_via_retry = aggregated_result.get('recovered_via_retry', set())
            metrics_counts = aggregated_result.get('counts', metrics_counts)
            metrics_window = aggregated_result.get('window')
            for qid in QIDS:
                metrics = aggregated_result['metrics'].get(qid, {})
                received = aggregated_result['metrics_received'].get(qid, {})
                if metrics.get('lat_p95') is not None:
                    snapshot[qid]['lat_p95'] = metrics['lat_p95']
                    metrics_received[qid]['lat'] = received.get('lat', False)
                if metrics.get('drop_p95') is not None:
                    snapshot[qid]['drop_p95'] = metrics['drop_p95']
                    metrics_received[qid]['drop'] = received.get('drop', False)
                if metrics.get('util_p95') is not None:
                    snapshot[qid]['util_p95'] = metrics['util_p95']
                    metrics_received[qid]['util'] = received.get('util', False)

        # Mark data_valid - if metrics present and sane, data is valid
        # Freshness check removed - if we got data from InfluxDB, it's fresh enough
        for qid in QIDS:
            m = metrics_received[qid]
            metrics_present = m['lat'] and m['drop'] and m['util']
            values_sane = self._metric_sane(snapshot[qid]) if metrics_present else False
            snapshot[qid]['data_valid'] = metrics_present and values_sane
            snapshot[qid]['recovered_via_retry'] = qid in recovered_via_retry
            snapshot[qid]['telemetry_counts'] = metrics_counts.get(
                qid,
                {'lat': 0, 'drop': 0, 'util': 0},
            )

        self._normalize_optional_queue_telemetry(snapshot)

        # Log telemetry status for monitoring
        valid_count = sum(1 for qid in QIDS if snapshot[qid]['data_valid'])
        required_qids = set(self._required_qids_for_current_profile())
        required_valid_count = self._required_valid_count(snapshot)
        required_invalid = self._required_invalid_queues(snapshot)
        if required_invalid:
            reasons = []
            for qid in required_invalid:
                if not metrics_received[qid]['lat'] or not metrics_received[qid]['drop'] or not metrics_received[qid]['util']:
                    reasons.append(f"q{qid}:missing_metrics")
                elif not self._metric_sane(snapshot[qid]):
                    reasons.append(f"q{qid}:insane_values")
            optional_missing = [
                qid for qid in QIDS
                if qid not in required_qids
                and not metrics_received[qid]['lat']
            ]
            optional_note = (
                f"; optional missing Q{optional_missing}"
                if optional_missing else ""
            )
            log.warning(
                f"[Telemetry] Invalid required data for queues {required_invalid} "
                f"({', '.join(reasons)}) - required "
                f"{required_valid_count}/{len(required_qids)} valid, "
                f"overall {valid_count}/{len(QIDS)} valid{optional_note}"
            )
            # Detailed diagnostics for debugging missing metrics
            log.warning(f"[Telemetry Debug] Time window: {start} to {stop}")
            log.warning(f"[Telemetry Debug] Global step: {self.global_step}")
            log.warning(f"[Telemetry Debug] Metrics received: {metrics_received}")
            log.warning(f"[Telemetry Debug] Metric counts: {metrics_counts}")
            if metrics_window:
                log.warning(f"[Telemetry Debug] Cache metric window audit: {metrics_window}")
            log.warning(f"[Telemetry Debug] Hot demands: {all_hot_demands}")
            freshness = self._cache_freshness(
                qids=list(required_qids),
                start=start,
                stop=stop,
            )
            self._last_telemetry_freshness = freshness
            self._log_freshness_report(
                freshness,
                list(required_qids),
                prefix=f"[Telemetry Freshness S{self.global_step}]",
                level=logging.WARNING,
            )

        # 2. Get Path and Bottleneck Info (Queue-Specific)
        # Each queue gets its own bottleneck detection and alternative metrics
        # This ensures accurate per-queue congestion identification
        # Note: all_hot_demands already retrieved from parallel query above

        cache_path_context = {}
        cache_path_metrics = {}
        if self._read_from_cache():
            sw_ids_by_qid = {}
            for qid in QIDS:
                hot = all_hot_demands.get(qid)
                if not hot:
                    continue
                src_ip, dst_ip = hot
                path = self.controller.get_path_by_ips_for_queue(src_ip, dst_ip, qid)
                if not path:
                    continue
                sw_names = [n for n in path if n in self.controller.switch_name_to_id]
                sw_ids = [int(self.controller.switch_name_to_id[n]) for n in sw_names]
                if not sw_ids:
                    continue
                cache_path_context[qid] = {
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'path': list(path),
                    'sw_ids': sw_ids,
                }
                sw_ids_by_qid[qid] = sw_ids

            if sw_ids_by_qid:
                try:
                    cache_path_metrics = self._cache_switch_metrics_multi(
                        start,
                        stop,
                        sw_ids_by_qid,
                    )
                except Exception as e:
                    log.warning(f"[Query S{step}] local telemetry cache switch_metrics_multi failed: {e}")
                    cache_path_metrics = {}

        for qid in QIDS:
            cached_path = cache_path_context.get(qid)
            if cached_path is not None:
                src_ip = cached_path['src_ip']
                dst_ip = cached_path['dst_ip']
                path = cached_path['path']
                sw_ids = cached_path['sw_ids']
            else:
                # Find hottest demand for this queue from batch result
                hot = all_hot_demands.get(qid)
                if not hot:
                    if qid not in required_qids:
                        continue
                    # No cache fallback - retry logic already exhausted in _get_all_hottest_demands
                    log.info(f"[Snapshot] Queue {qid}: No hot demand found after retries, skipping bottleneck detection")
                    continue

                src_ip, dst_ip = hot
                # Get current path for THIS queue (uses queue-specific paths after reroutes)
                path = self.controller.get_path_by_ips_for_queue(src_ip, dst_ip, qid)
                if not path:
                    log.info(f"[Snapshot] Queue {qid}: No path found for ({src_ip}, {dst_ip})")
                    continue

                # --- Step A: Path Metrics (Queue-Specific) ---
                # Query metrics filtered by this queue_id for accurate bottleneck detection
                # Filter to switches only (exclude hosts) - check against known switch names
                sw_names = [n for n in path if n in self.controller.switch_name_to_id]
                sw_ids = [self.controller.switch_name_to_id[n] for n in sw_names]
                sw_ids = [int(s) for s in sw_ids]

            log.debug(f"[Snapshot] Queue {qid}: hot_demand=({src_ip}, {dst_ip})")
            snapshot[qid]['hot_src_ip'] = src_ip
            snapshot[qid]['hot_dst_ip'] = dst_ip
            snapshot[qid]['path_nodes'] = list(path)
            
            if not sw_ids:
                continue
                
            # Query path switches with queue_id filter for accurate per-queue bottleneck
            if qid in cache_path_metrics:
                path_metrics = cache_path_metrics.get(qid, {})
            else:
                path_metrics = self._query_switch_metrics_for_queue(
                    sw_ids,
                    qid,
                    start,
                    stop,
                )
            
            # Identify Bottleneck (SKIP edge switches - they have no alternatives)
            best_sid, best_score = None, -1.0
            for sid in sw_ids:
                # Skip edge switches (leaf/tor/access) - they're at edge and have no alt paths
                if self.controller._is_edge_switch(sid):
                    continue
                
                r = path_metrics.get(sid, {'drop': 0, 'lat': 0})
                drop_norm = min(r['drop'], DROP_CAP) / DROP_CAP
                lat_norm = min(r['lat'], SLA_THRESHOLDS[qid]) / SLA_THRESHOLDS[qid]
                score = 0.6 * drop_norm + 0.4 * lat_norm
                if score > best_score:
                    best_sid, best_score = sid, score
            
            snapshot[qid]['bottleneck_sid'] = best_sid
            # Cache the bottleneck score for later use (CPU optimization - avoid recomputation)
            snapshot[qid]['bottleneck_score'] = best_score

            if best_sid is not None:
                # Store bottleneck stats (queue-specific)
                bm = path_metrics.get(best_sid, {'drop': 0, 'lat': 0, 'util': 0})
                snapshot[qid]['bottleneck_drop'] = bm['drop']
                snapshot[qid]['bottleneck_lat'] = bm['lat']
                snapshot[qid]['bottleneck_util'] = bm['util']
                snapshot[qid]['bottleneck_role'] = self.controller._normalize_role(self.controller._role_of_sid(best_sid))

                # --- Step B: Alternatives (Queue-Specific) ---
                # Get alternatives for this bottleneck and query their queue-specific metrics
                alts = self.controller.find_all_alternates(best_sid, path)
                log.debug(f"[Snapshot] Queue {qid}: bottleneck={best_sid}, role={snapshot[qid]['bottleneck_role']}, alternatives={alts}")

                # Collect valid alt switch IDs for this queue
                valid_alts = []
                alt_sids = []
                for alt_name in alts:
                    alt_sid = self.controller.switch_name_to_id.get(alt_name)
                    if alt_sid is not None:
                        valid_alts.append((alt_name, int(alt_sid)))
                        alt_sids.append(int(alt_sid))

                # Query alternative metrics for THIS queue specifically
                if alt_sids:
                    alt_metrics = self._query_switch_metrics_for_queue(
                        alt_sids,
                        qid,
                        start,
                        stop,
                    )

                    # Use cached bottleneck score for relative comparison (CPU optimization)
                    bn_score = best_score

                    # Score each alternative RELATIVE to bottleneck (higher = better improvement)
                    # Alternatives with NO metrics get max score (best, prioritized)
                    scored_alts = []
                    for name, sid in valid_alts:
                        m = alt_metrics.get(sid)
                        if m is None:
                            # No traffic on this switch = best option (max improvement assumed)
                            rel_score = bn_score  # Highest possible relative score
                            scored_alts.append((rel_score, name, {'drop': 0, 'lat': 0, 'util': 0}))
                        else:
                            # Calculate alternative's absolute score
                            alt_drop_norm = min(m['drop'], DROP_CAP) / DROP_CAP
                            alt_lat_norm = min(m['lat'], SLA_THRESHOLDS[qid]) / SLA_THRESHOLDS[qid]
                            alt_score = 0.6 * alt_drop_norm + 0.4 * alt_lat_norm
                            # Relative score = improvement over bottleneck (positive = better)
                            rel_score = bn_score - alt_score
                            scored_alts.append((rel_score, name, m))

                    # Sort by relative score DESCENDING (higher = more improvement)
                    # Shuffle first for random tie-breaking when scores are equal
                    random.shuffle(scored_alts)
                    scored_alts.sort(key=lambda x: x[0], reverse=True)

                    # Take best MAX_ALTS (2) alternatives
                    final_alts = []
                    for rel_score, name, m in scored_alts[:self.MAX_ALTS]:
                        final_alts.append({
                            'name': name,
                            'drop': m['drop'],
                            'lat': m['lat'],
                            'util': m['util'],
                        })

                    log.debug(f"[Snapshot] Q{qid} scored alts: {[(round(s[0], 3), s[1]) for s in scored_alts]}, selected: {[a['name'] for a in final_alts]}")
                    snapshot[qid]['alternatives'] = final_alts
                    snapshot[qid]['alt_exists'] = bool(final_alts)
                else:
                    snapshot[qid]['alternatives'] = []
                    snapshot[qid]['alt_exists'] = False

        return snapshot

    def _query_switch_metrics_for_queue(
        self,
        sw_ids: List[int],
        qid: int,
        start: Optional[str] = None,
        stop: Optional[str] = None,
    ) -> Dict[int, Dict]:
        """
        Query drop, latency, and utilization for a list of switch IDs,
        filtered by a specific queue_id for accurate per-queue metrics.
        
        Args:
            sw_ids: List of switch IDs to query
            qid: Queue ID to filter metrics by (0=voice, 1=video, 7=BE)
            
        Returns:
            Dict mapping switch_id -> {'drop': float, 'lat': float, 'util': float}
        """
        if not sw_ids:
            return {}
            
        sid_filter = " or ".join([f'r.switch_id == "{sid}"' for sid in sw_ids])
        if start is None or stop is None:
            start, stop = self._time_window()

        if self._read_from_cache():
            try:
                return self._cache_switch_metrics_for_queue(start, stop, sw_ids, qid)
            except Exception as e:
                log.warning(f"Local telemetry cache switch_metrics failed for Q{qid}: {e}")
                if not self._allow_influx_fallback():
                    return {}
                log.warning(f"Falling back to InfluxDB switch metrics for Q{qid}")
        
        # Query metrics WITH queue_id filter for queue-specific accuracy
        flux = f'''
        from(bucket:"{self.bucket}")
            |> range(start:{start}, stop:{stop})
            |> filter(fn: (r) => {sid_filter})
            |> filter(fn: (r) => r.queue_id == "{qid}")
            |> filter(fn: (r) => r._measurement == "q_drop_rate_100ms" or r._measurement == "switch_latency" or r._measurement == "tx_utilization")
            |> toFloat()
            |> group(columns:["switch_id", "_measurement"])
            |> max(column:"_value")
            |> group(columns:["switch_id"])
            |> pivot(rowKey:["switch_id"], columnKey:["_measurement"], valueColumn:"_value")
        '''

        results = {}
        # PHASE 2.4: Use retry wrapper for query
        tables = self._influx_query_with_retry(flux)
        if tables is not None:
            for table in tables:
                for record in table.records:
                    sid = int(record.values.get('switch_id', 0) or 0)
                    results[sid] = {
                        'drop': float(record.values.get('q_drop_rate_100ms', 0) or 0),
                        'lat': float(record.values.get('switch_latency', 0) or 0),
                        'util': float(record.values.get('tx_utilization', 0) or 0),
                    }

        return results

    def _get_all_hottest_demands(
        self,
        step: int,
        target_qids: Optional[List[int]] = None,
    ) -> Dict[int, Tuple[str, str]]:
        """Get the demand with highest latency for ALL queues in one query.

        Uses retry logic similar to flow_latency recovery:
        - Up to 5 retries with 1 second delay between each
        - Query window expands on each retry to catch delayed data
        - Continues until all queues have hot demands or retries exhausted

        Args:
            step: Global step number for logging (passed from caller)
        """
        start, stop = self._time_window()

        requested_qids = [int(qid) for qid in (target_qids or QIDS)]
        results = {}
        remaining_queues = set(requested_qids)
        max_retries = 5
        retry_delay = 1.0
        queue_filter = " or ".join(
            [f'r.queue_id == "{qid}"' for qid in requested_qids]
        )

        for attempt in range(max_retries):
            if attempt > 0:
                time.sleep(retry_delay)

            # Expand stop time by cumulative wait
            window_extension = attempt * retry_delay
            if window_extension > 0:
                stop_dt = datetime.fromisoformat(stop.rstrip('Z')) + timedelta(seconds=window_extension)
                current_stop = stop_dt.isoformat() + 'Z'
            else:
                current_stop = stop

            flux = f'''
            from(bucket:"{self.bucket}")
                |> range(start:{start}, stop:{current_stop})
                |> filter(fn: (r) => r._measurement == "flow_latency")
                |> filter(fn: (r) => {queue_filter})
                |> toFloat()
                |> group(columns:["queue_id", "src_ip", "dst_ip"])
                |> mean(column:"_value")
                |> group(columns:["queue_id"])
                |> sort(columns:["_value"], desc:true)
                |> limit(n:1)
            '''

            t0 = time.time()
            try:
                if self._read_from_cache():
                    cache_results = self._cache_hot_demands(
                        start,
                        current_stop,
                        list(remaining_queues),
                    )
                    elapsed_ms = (time.time() - t0) * 1000
                    for qid, demand in cache_results.items():
                        if qid in remaining_queues:
                            results[qid] = demand
                            remaining_queues.discard(qid)
                else:
                    tables = self.query_api.query(org=self.org, query=flux)
                    elapsed_ms = (time.time() - t0) * 1000

                    if tables:
                        for table in tables:
                            for record in table.records:
                                try:
                                    qid = int(record.values.get('queue_id', -1))
                                    src = record.values.get('src_ip')
                                    dst = record.values.get('dst_ip')
                                    if qid in remaining_queues and src and dst:
                                        results[qid] = (str(src), str(dst))
                                        remaining_queues.discard(qid)
                                except (ValueError, TypeError):
                                    continue

                window_info = f"+{window_extension:.0f}s window" if window_extension > 0 else ""
                if not remaining_queues:
                    # All queues have hot demands
                    if attempt > 0:
                        log.info(f"[Query S{step}] hot_demands retry {attempt+1}/{max_retries} SUCCESS: "
                                f"got ALL Q{list(results.keys())} in {elapsed_ms:.0f}ms {window_info}")
                    return results
                elif results and attempt > 0:
                    # Some found, continue retrying for the rest
                    log.info(f"[Query S{step}] hot_demands retry {attempt+1}/{max_retries} PARTIAL: "
                            f"got Q{list(results.keys())}, still missing Q{list(remaining_queues)} "
                            f"({elapsed_ms:.0f}ms {window_info})")
                elif attempt > 0:
                    log.warning(f"[Query S{step}] hot_demands retry {attempt+1}/{max_retries}: "
                               f"no data ({elapsed_ms:.0f}ms, {window_info})")

            except Exception as e:
                elapsed_ms = (time.time() - t0) * 1000
                if self._read_from_cache() and self._allow_influx_fallback():
                    log.warning(f"[Query S{step}] hot_demands local cache failed "
                                f"({elapsed_ms:.0f}ms), trying InfluxDB: {e}")
                    try:
                        if self.query_api is None:
                            raise RuntimeError("InfluxDB client is unavailable")
                        tables = self.query_api.query(org=self.org, query=flux)
                        elapsed_ms = (time.time() - t0) * 1000
                        if tables:
                            for table in tables:
                                for record in table.records:
                                    try:
                                        qid = int(record.values.get('queue_id', -1))
                                        src = record.values.get('src_ip')
                                        dst = record.values.get('dst_ip')
                                        if qid in remaining_queues and src and dst:
                                            results[qid] = (str(src), str(dst))
                                            remaining_queues.discard(qid)
                                    except (ValueError, TypeError):
                                        continue
                    except Exception as influx_exc:
                        if attempt > 0:
                            log.warning(f"[Query S{step}] hot_demands retry {attempt+1}/{max_retries} FAILED "
                                       f"({elapsed_ms:.0f}ms): {influx_exc}")
                    else:
                        window_info = f"+{window_extension:.0f}s window" if window_extension > 0 else ""
                        if not remaining_queues:
                            if attempt > 0:
                                log.info(f"[Query S{step}] hot_demands retry {attempt+1}/{max_retries} SUCCESS: "
                                        f"got ALL Q{list(results.keys())} in {elapsed_ms:.0f}ms {window_info}")
                            return results
                        if results and attempt > 0:
                            log.info(f"[Query S{step}] hot_demands retry {attempt+1}/{max_retries} PARTIAL: "
                                    f"got Q{list(results.keys())}, still missing Q{list(remaining_queues)} "
                                    f"({elapsed_ms:.0f}ms {window_info})")
                elif attempt > 0:
                    log.warning(f"[Query S{step}] hot_demands retry {attempt+1}/{max_retries} FAILED "
                               f"({elapsed_ms:.0f}ms): {e}")

            # On first attempt, if we got partial results, trigger retry
            if attempt == 0 and remaining_queues:
                log.warning(f"[Query S{step}] hot_demands missing for Q{list(remaining_queues)}, triggering retry")

        if results and remaining_queues:
            log.warning(f"[Query S{step}] hot_demands retry exhausted - "
                       f"got Q{list(results.keys())}, missing Q{list(remaining_queues)}")
        return results
    
    def _metric_sane(self, q: Dict) -> bool:
        """
        Sanity check metric values to catch NaN/inf/stale/wrong-tagged data.
        
        Returns True if values are within reasonable bounds.
        Uses wide bounds since normalization caps are applied later.
        """
        try:
            lat = float(q.get('lat_p95', -1))
            drop = float(q.get('drop_p95', -1))
            util = float(q.get('util_p95', -1))
        except (TypeError, ValueError):
            return False
        
        # Reject NaN/inf explicitly
        if not all(math.isfinite(x) for x in [lat, drop, util]):
            log.debug(f"[Sanity] Non-finite values: lat={lat}, drop={drop}, util={util}")
            return False
        
        # Latency: must be positive and < 10 seconds (10,000 ms)
        lat_ok = 0 < lat < 10_000
        # Drop rate: allow wide range for bursts; normalization caps later
        # (was DROP_CAP*2=10, now 10_000 to avoid rejecting valid burst data)
        drop_ok = 0 <= drop < 10_000
        # Utilization: 0 to 100%
        util_ok = 0 <= util <= 100
        
        if not (lat_ok and drop_ok and util_ok):
            log.debug(f"[Sanity] Out of bounds: lat={lat}({lat_ok}), drop={drop}({drop_ok}), util={util}({util_ok})")
        
        return lat_ok and drop_ok and util_ok
    
    def _build_raw_state(self, snapshot: Dict[int, Dict]) -> np.ndarray:
        """
        Build 52-dimensional raw observation vector with relative metrics encoding.
        (Actions are stacked separately as one-hot vectors)

        Layout per queue (16 features):
          [0-5]: Basic metrics (lat_ratio, drop_norm, util_norm, sla_met, lat_ema, lat_ema_diff)
          [6-9]: Bottleneck info (present, drop, lat, util)
          [10-15]: Alternatives 2 × 3 = 6 (available, drop_vs_bn, lat_vs_bn)

        Global (4):
          - Max pressure
          - Steps since last action (normalized, helps avoid rapid oscillation)
          - is_fat_tree (1.0 or 0.0)
          - is_leaf_spine (1.0 or 0.0)

        Total: 3×16 + 4 = 52 features
        """
        state = np.zeros(RAW_STATE_DIM, dtype=np.float32)
        
        pressures = []
        idx = 0
        
        for qid in QIDS:
            q = snapshot[qid]
            sla = SLA_THRESHOLDS[qid]
            
            # --- 1. Basic Metrics (6) ---
            lat_ratio_raw = q['lat_p95'] / sla
            lat_ratio = min(lat_ratio_raw, LAT_RATIO_CAP)  # Cap to prevent instability
            drop_norm = min(q['drop_p95'], DROP_CAP) / DROP_CAP
            util_norm = min(q['util_p95'], UTIL_CAP) / UTIL_CAP
            sla_met = 1.0 if lat_ratio_raw <= 1.0 else 0.0  # Use raw for accurate SLA check
            
            # Update EMA and compute smoothed features
            prev_ema = self.lat_ema[qid]
            self.lat_ema[qid] = LAT_EMA_ALPHA * lat_ratio + (1 - LAT_EMA_ALPHA) * prev_ema
            lat_ema = min(self.lat_ema[qid], LAT_RATIO_CAP)  # Capped
            lat_ema_diff = max(-LAT_DIFF_CAP, min(lat_ratio - lat_ema, LAT_DIFF_CAP))  # Deviation from EMA
            
            state[idx] = lat_ratio
            state[idx+1] = drop_norm
            state[idx+2] = util_norm
            state[idx+3] = sla_met
            state[idx+4] = lat_ema
            state[idx+5] = lat_ema_diff
            idx += 6
            
            # --- 2. Bottleneck Info (4) ---
            bn_sid = q.get('bottleneck_sid')
            if bn_sid is not None:
                state[idx] = 1.0  # Present
                bn_drop = min(q.get('bottleneck_drop', 0), DROP_CAP) / DROP_CAP
                bn_lat = min(q.get('bottleneck_lat', 0) / sla, LAT_RATIO_CAP)  # Capped
                bn_util = min(q.get('bottleneck_util', 0), UTIL_CAP) / UTIL_CAP
                state[idx+1] = bn_drop
                state[idx+2] = bn_lat
                state[idx+3] = bn_util
            else:
                # No bottleneck identified
                state[idx] = 0.0
                bn_drop = 0.0
                bn_lat = 0.0
                bn_util = 0.0
            
            idx += 4
            
            # --- 3. Alternatives (2 × 3 = 6) ---
            alts = q.get('alternatives', [])
            
            for i in range(self.MAX_ALTS):
                if i < len(alts):
                    alt = alts[i]
                    
                    # Available
                    state[idx] = 1.0
                    
                    # Relative metrics (vs bottleneck)
                    alt_drop = min(alt.get('drop', 0), DROP_CAP) / DROP_CAP
                    alt_lat = min(alt.get('lat', 0) / sla, LAT_RATIO_CAP)  # Capped
                    state[idx+1] = alt_drop - bn_drop  # Negative = better than bottleneck
                    # Cap lat difference to bounded range
                    lat_diff = alt_lat - bn_lat
                    state[idx+2] = max(-LAT_DIFF_CAP, min(lat_diff, LAT_DIFF_CAP))
                else:
                    # No alternative at this index
                    state[idx:idx+3] = 0.0
                
                idx += 3
            
            # Track pressure for global feature
            pressure = 0.5 * lat_ratio + 0.3 * drop_norm + 0.2 * util_norm
            pressures.append(pressure)
        
        # Global max pressure
        state[idx] = max(pressures) if pressures else 0.0
        idx += 1
        
        # Steps since last action (normalized: 0=just acted, 1=10+ steps ago)
        # Helps agent learn to wait for effects before acting again
        steps_since = self.global_step - getattr(self, '_last_action_global_step', 0)
        state[idx] = min(steps_since / 10.0, 1.0)  # Cap at 10 steps
        idx += 1

        # Topology Encoding (2 features): Explicit topology type indicators
        # This helps agent distinguish between fat-tree vs leaf-spine architectures
        # which have different path characteristics and bottleneck patterns
        if self._topology_builder is not None:
            topo_type = self._topology_builder.config.topology.type.value.lower()
            state[idx] = 1.0 if 'fat' in topo_type else 0.0  # is_fat_tree
            state[idx+1] = 1.0 if 'leaf' in topo_type or 'spine' in topo_type else 0.0  # is_leaf_spine
        else:
            state[idx] = 0.0
            state[idx+1] = 0.0

        return state
    
    def _action_to_onehot(self, action: int) -> np.ndarray:
        """Convert action index to one-hot vector."""
        onehot = np.zeros(ACTION_DIM, dtype=np.float32)
        onehot[action] = 1.0
        return onehot

    def _init_stacks(self, raw_state: np.ndarray):
        """Initialize frame and action stacks with given state."""
        self.frame_stack.clear()
        for _ in range(STACK_SIZE):
            self.frame_stack.append(raw_state.copy())
        noop_onehot = self._action_to_onehot(0)
        self.action_stack.clear()
        for _ in range(STACK_SIZE):
            self.action_stack.append(noop_onehot.copy())

    def _is_sla_violated(self, qid: int, snapshot: Dict) -> bool:
        """Check if queue is violating SLA (above margin)."""
        ratio = snapshot[qid]['lat_p95'] / SLA_THRESHOLDS[qid]
        return ratio > SLA_MARGIN_HIGH

    def _build_stacked_state(self) -> np.ndarray:
        """
        Build stacked state with exponential decay weighting.
        Newer frames have more impact than older ones.

        Returns:
            928-dimensional state vector:
              - [0-799]: Stacked observations (50 * 16 = 800 features)
              - [800-927]: Stacked one-hot actions (8 * 16 = 128 features)

        Layout: [obs_t-15, ..., obs_t-1, obs_t,
                 act_t-15, ..., act_t-1, act_t]

        Decay weights (STACK_DECAY=0.95, 16 frames):
          frame 0 (oldest): 0.95^15 = 0.46
          frame 15 (newest): 0.95^0 = 1.0

        This allows the agent to see:
        - Trends: If latency is rising or falling (from observation history)
        - Causality: Full action history as one-hot vectors
        - Recency bias: Recent observations weighted more heavily
        """
        # CPU Optimization: Copy into preallocated buffers instead of allocating new arrays
        # This avoids ~960 element allocation every step
        for i, frame in enumerate(self.frame_stack):
            self._frames_buffer[i] = frame
        for i, action in enumerate(self.action_stack):
            self._actions_buffer[i] = action

        # Broadcast multiply: (STACK_SIZE, dim) * (STACK_SIZE, 1) -> (STACK_SIZE, dim)
        # Uses pre-computed DECAY_WEIGHTS from module constants
        weighted_obs = (self._frames_buffer * DECAY_WEIGHTS[:, np.newaxis]).ravel()
        weighted_actions = (self._actions_buffer * DECAY_WEIGHTS[:, np.newaxis]).ravel()

        return np.concatenate([weighted_obs, weighted_actions])
    
    def _get_valid_actions(self, snapshot: Dict[int, Dict]) -> np.ndarray:
        """
        Get mask of valid actions based on available alternatives.
        Action 7 (multi) is valid if ANY queue has valid alternatives.
        """
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[0] = True  # No-op always valid
        
        now = time.monotonic()
        in_cooldown = (now - self.last_action_time) < COOLDOWN_SECONDS
        
        any_valid_for_multi = False
        
        if not in_cooldown:
            actionable_qids = set(self._required_qids_for_current_profile())
            for action, mapping in self.ACTION_MAP.items():
                if mapping is None:  # Skip no-op
                    continue
                if mapping == 'multi':  # Handle multi-action separately
                    continue
                    
                qid, alt_idx = mapping
                if qid not in actionable_qids:
                    continue
                
                # Check if this queue has enough alternatives
                q = snapshot[qid]
                if q.get('bottleneck_sid') is not None:
                    alts = q.get('alternatives', [])
                    if alt_idx < len(alts):
                        mask[action] = True
                        any_valid_for_multi = True
            
            # Multi-action is valid ONLY if 2+ queues are VIOLATING SLA *and* have alternatives
            # This ensures multi-action is reserved for situations where multiple queues need help
            violating_with_alts = 0
            for qid in actionable_qids:
                q = snapshot[qid]
                # Check soft margin violation (consistent with step logic)
                if self._is_sla_violated(qid, snapshot) and q.get('bottleneck_sid') is not None and len(q.get('alternatives', [])) > 0:
                    violating_with_alts += 1

            if violating_with_alts >= 2:
                mask[7] = True
        
        return mask
    
    def _compute_reward(self, snapshot: Dict[int, Dict]) -> Tuple[float, Dict]:
        """
        Compute reward based on SLA compliance with improved stability.
        
        Key features:
        1. Soft margin around SLA to reduce flip-flopping
        2. Higher drop penalty (drops are most actionable)
        3. Tanh compression to bound extreme values smoothly [-2.5, +2.5]
        
        Note: Action cost is applied separately in step() only for successful actions.
        This prevents penalizing failed reroute attempts.
        
        Returns:
            (reward, info_dict)
        """
        raw_reward = 0.0
        reward_qids = list(self._required_qids_for_current_profile())
        info = {
            'sla_met': [],
            'sla_violated': [],
            'per_queue': {},
            'reward_qids': reward_qids,
            'sla_total': len(reward_qids),
        }
        
        for qid in reward_qids:
            q = snapshot[qid]
            sla = SLA_THRESHOLDS[qid]
            lat = q['lat_p95']
            
            # Calculate ratio with soft margin (0.9-1.1 is neutral zone)
            ratio = lat / sla

            if ratio <= SLA_MARGIN_LOW:
                # Clearly under SLA: positive reward
                # Use sqrt for diminishing returns (don't over-reward very low latency)
                headroom = SLA_MARGIN_LOW - ratio  # How much below margin
                component = REWARD_SLA_MET_SCALE * np.sqrt(headroom / SLA_MARGIN_LOW)
                info['sla_met'].append(qid)
            elif ratio <= SLA_MARGIN_HIGH:
                # Within margin: small neutral reward (avoid flip-flopping)
                # Linear interpolation from +0.1 to -0.1
                t = (ratio - SLA_MARGIN_LOW) / (SLA_MARGIN_HIGH - SLA_MARGIN_LOW)  # 0 to 1
                component = 0.1 * (1.0 - 2.0 * t)  # +0.1 to -0.1
                info['sla_met'].append(qid)  # Still counts as met
            else:
                # Above margin: penalty
                # Use tanh to compress extreme violations smoothly
                excess = ratio - SLA_MARGIN_HIGH  # How much above margin
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
                'drop': q['drop_p95'],
                'util': q['util_p95'],
                'ratio': ratio,
                'component': component,
                'drop_penalty': drop_penalty
            }

        if reward_qids:
            raw_reward *= len(QIDS) / len(reward_qids)
        
        # Stability: Symmetric soft clipping with tanh
        # Maps to [-2.5, +2.5] range - symmetric to reduce bias
        reward = 2.5 * np.tanh(raw_reward / 2.5)
        
        info['raw_reward'] = raw_reward  # Pre-clipping for debugging
        
        return reward, info
    
    def _apply_multi_reroute(self, snapshot: Dict[int, Dict]) -> Tuple[bool, Optional[str], int]:
        """
        Apply multi-queue reroute: reroutes ONLY queues violating SLA.
        
        Returns:
            (success, description, reroute_count) - count of queues rerouted
        """
        rerouted = []
        actionable_qids = set(self._required_qids_for_current_profile())
        
        for qid in QIDS:
            if qid not in actionable_qids:
                continue
            q = snapshot[qid]
            
            # Only reroute if SLA is violated
            if q['lat_p95'] <= SLA_THRESHOLDS[qid]:
                continue
                
            # Need bottleneck and alternatives
            src_ip = q.get('hot_src_ip')
            dst_ip = q.get('hot_dst_ip')
            bottleneck_sid = q.get('bottleneck_sid')
            alts = q.get('alternatives', [])
            
            if not (src_ip and dst_ip and bottleneck_sid and alts):
                continue
            
            # Use first alternative
            alt_name = alts[0]['name']
            
            ok, msg = self.controller.reroute_one_demand_symmetric(
                src_ip=src_ip, dst_ip=dst_ip, qid=qid,
                worst_switch_id=int(bottleneck_sid), alt_switch_name=alt_name
            )
            
            if ok:
                rerouted.append(qid)
                self.controller.track_usage(alt_name)
                self.controller.record_queue_change(qid, self.global_step)
                log.info(f"[MULTI] Rerouted qid={qid} to {alt_name} [bn={bottleneck_sid}]")
        
        self._last_rerouted_qids = list(rerouted)
        if rerouted:
            return True, f"multi:{len(rerouted)}", len(rerouted)
        else:
            return False, None, 0

    def _rollback_failed_reroute(self, qid: int, reason: str) -> bool:
        """Rollback the most recent queue-specific route change after a blackhole check."""
        try:
            reverted = self.controller.revert_last_change_for_qid(int(qid))
        except Exception as exc:
            log.warning(
                f"[Rollback] Failed to revert Q{qid} after {reason}: {exc}"
            )
            return False

        if reverted:
            log.warning(
                f"[Rollback] Reverted last Q{qid} route change after {reason}"
            )
            return True

        log.warning(
            f"[Rollback] No pending Q{qid} route change to revert after {reason}"
        )
        return False

    def _verify_queue_traffic_flowing(self, qid: int, window_seconds: float = 1.5) -> bool:
        """
        Verify that traffic is flowing for a specific queue by checking InfluxDB.

        This is a lightweight check to detect if a reroute broke traffic flow.
        Returns True if data exists, False if no data found.

        Args:
            qid: Queue ID to verify
            window_seconds: Time window to check for data
        """
        try:
            stop_dt = datetime.utcnow()
            start_dt = stop_dt - timedelta(seconds=window_seconds)
            start = start_dt.isoformat() + 'Z'
            stop = stop_dt.isoformat() + 'Z'

            if self._read_from_cache():
                try:
                    return self._cache_traffic_count(qid, start, stop) > 0
                except Exception as e:
                    log.debug(f"[Verify] Local telemetry cache traffic check for Q{qid} failed: {e}")
                    if not self._allow_influx_fallback():
                        return True

            # Simple count query for the specific queue
            flux = f'''
            from(bucket:"{self.bucket}")
                |> range(start:{start}, stop:{stop})
                |> filter(fn: (r) => r._measurement == "flow_latency")
                |> filter(fn: (r) => r.queue_id == "{qid}")
                |> count()
            '''

            tables = self._influx_query_with_retry(flux)

            # Check if we got any data
            count = 0
            for table in tables:
                for record in table.records:
                    count += record.get_value() or 0

            return count > 0

        except Exception as e:
            log.debug(f"[Verify] Traffic check for Q{qid} failed: {e}")
            return True  # Assume OK on error to avoid false alarms

    def _verify_rerouted_traffic_flowing(
        self,
        qids: List[int],
        window_seconds: float = 1.5,
    ) -> Dict[int, bool]:
        """Batch post-reroute flow checks when using the local cache."""
        qids = [int(qid) for qid in qids]
        if not qids:
            return {}

        if self._read_from_cache():
            try:
                stop_dt = datetime.utcnow()
                start_dt = stop_dt - timedelta(seconds=window_seconds)
                start = start_dt.isoformat() + 'Z'
                stop = stop_dt.isoformat() + 'Z'
                counts = self._cache_traffic_counts(qids, start, stop)
                return {qid: counts.get(qid, 0) > 0 for qid in qids}
            except Exception as e:
                log.debug(f"[Verify] Local telemetry cache batched traffic check failed: {e}")
                if not self._allow_influx_fallback():
                    return {qid: True for qid in qids}

        return {
            qid: self._verify_queue_traffic_flowing(qid, window_seconds)
            for qid in qids
        }

    def _apply_action(self, action: int, snapshot: Dict[int, Dict]) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Apply routing action based on explicit alternative selection.
        
        Returns:
            (success, alt_name, alt_idx) - alt_name for logging, alt_idx for InfluxDB
            For multi-action: alt_name is 'multi:N', alt_idx is count of rerouted queues
        """
        self._last_rerouted_qids = []
        if action == 0:
            return False, None, None  # No-op
        
        # Parse action from map
        mapping = self.ACTION_MAP.get(action)
        if not mapping:
            return False, None, None
        
        # Handle multi-queue action
        if mapping == 'multi':
            return self._apply_multi_reroute(snapshot)
        
        # Single queue action
        qid, alt_idx = mapping
        if qid not in set(self._required_qids_for_current_profile()):
            log.debug(
                f"Action {action} targets Q{qid}, which is optional for "
                f"profile {self.current_traffic_profile}"
            )
            return False, None, None
        q = snapshot[qid]
        src_ip = q.get('hot_src_ip')
        dst_ip = q.get('hot_dst_ip')
        bottleneck_sid = q.get('bottleneck_sid')
        alts = q.get('alternatives', [])
        
        if not (src_ip and dst_ip and bottleneck_sid):
            log.warning(f"Cannot apply action {action} (q={qid}): missing path info")
            return False, None, None
            
        # Check if requested alternative index exists
        if alt_idx >= len(alts):
            log.warning(f"Cannot apply action {action}: alt index {alt_idx} out of range ({len(alts)} avail)")
            return False, None, None
            
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
            self._last_rerouted_qids = [qid]
        else:
            log.warning(f"[ACTION {action}] Reroute failed: {msg}")
        
        return ok, alt_name, alt_idx
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action index (0-7)
        
        Returns:
            (next_state, reward, terminated, truncated, info)
            - next_state: 464-dim stacked state
            - terminated: True if absorbing state (always False for this env)
            - truncated: True if horizon reached (episode_step >= MAX_EPISODE_STEPS)
        """
        # Increment global step counter (persists across episodes)
        self.global_step += 1
        self.episode_step += 1

        if self.traffic_manager:
            self.traffic_manager.apply_step_profile(self.episode_step)
        
        # Get current snapshot
        current_snapshot = self.last_snapshot
        
        # Apply action
        action_applied, alt_name, alt_idx = self._apply_action(action, current_snapshot)
        
        # Update last action tracking (only on applied actions for stability feature)
        if action_applied:
            self.last_action_time = time.monotonic()
            self._last_action_step = self.episode_step  # For steps_since_action feature
            self._last_action_global_step = self.global_step
        self.last_action = action
        
        # Wait for network to settle
        delay = DELAY_AFTER_ACTION if action_applied else DELAY_NO_ACTION
        time.sleep(delay)

        # Post-reroute traffic verification: check if traffic is still flowing
        # This detects cases where a reroute broke packet forwarding
        if action_applied and action != 0:
            failed_reroute_qids = []
            mapping = self.ACTION_MAP.get(action)
            reroute_flowing = self._verify_rerouted_traffic_flowing(
                list(getattr(self, "_last_rerouted_qids", []))
            )
            for rerouted_qid in getattr(self, "_last_rerouted_qids", []):
                if reroute_flowing.get(rerouted_qid, True):
                    continue
                action_desc = (
                    "multi-queue reroute"
                    if mapping == 'multi'
                    else f"reroute to {alt_name}"
                )
                log.warning(f"[Step {self.episode_step}] POST-REROUTE VERIFICATION FAILED: "
                           f"No traffic data for Q{rerouted_qid} after {action_desc}")
                log.warning(f"[Step {self.episode_step}] Reroute details: action={action}, "
                           f"qid={rerouted_qid}, alt={alt_name}, "
                           f"bn={current_snapshot[rerouted_qid].get('bottleneck_sid')}")
                self._rollback_failed_reroute(
                    rerouted_qid,
                    f"post-reroute traffic verification failure at step {self.episode_step}",
                )
                failed_reroute_qids.append(rerouted_qid)
            if failed_reroute_qids:
                action_applied = False
                log.warning(
                    f"[Step {self.episode_step}] Marking action {action} invalid "
                    f"after rollback of Q{failed_reroute_qids}"
                )

        # Check if traffic is stable before collecting metrics
        # During traffic transitions (burst start/end), telemetry is unreliable
        if self.traffic_manager and not self.traffic_manager.is_traffic_stable():
            elapsed = self.traffic_manager.get_transition_elapsed()
            log.info(f"[Step {self.episode_step}] Traffic transitioning ({elapsed:.1f}s elapsed), using last snapshot")
            # Use deepcopy to avoid modifying last_snapshot's nested dicts
            # A shallow copy would cause 'transitioning' flag to persist in last_snapshot
            next_snapshot = copy.deepcopy(self.last_snapshot) if self.last_snapshot else self._collect_snapshot()
            # Mark as transitioning for reward calculation
            for qid in QIDS:
                next_snapshot[qid]['transitioning'] = True
        else:
            # Collect new snapshot
            next_snapshot = self._collect_snapshot()
        
        # Compute base reward (action cost applied separately below)
        reward, info = self._compute_reward(next_snapshot)
        
        # Always compute network pressure for state representation and logging
        # Uses same formula as _build_raw_state() for consistency
        pressure = 0.0
        for qid in QIDS:
            q = next_snapshot[qid]
            sla = SLA_THRESHOLDS[qid]
            lat_ratio = q['lat_p95'] / sla
            drop_norm = min(q['drop_p95'], DROP_CAP) / DROP_CAP
            util_norm = min(q['util_p95'], UTIL_CAP) / UTIL_CAP
            q_pressure = 0.5 * lat_ratio + 0.3 * drop_norm + 0.2 * util_norm
            pressure = max(pressure, q_pressure)
        
        info['pressure'] = pressure  # Always available for logging
        
        # Apply queue-specific action cost ONLY for successful actions
        # Cost depends on whether the TARGETED queue's SLA is met
        if action != 0 and action_applied:
            # Get which queue this action targets
            mapping = self.ACTION_MAP.get(action)
            
            if mapping == 'multi':
                # Multi-action: cost based on number of queues rerouted
                # alt_idx contains reroute_count for multi-action
                reroute_count = alt_idx if alt_idx else 0
                # Use sick cost per rerouted queue (they were all violating)
                action_cost = reroute_count * REWARD_ACTION_COST_SICK
                targeted_qid = -1  # Sentinel for multi-action
                info['multi_reroute_count'] = reroute_count
            elif mapping is not None:
                # Single-queue action: extract targeted queue ID
                targeted_qid = mapping[0]
                
                # Check targeted queue's health using TREND-BASED logic
                # We must look at current_snapshot (pre-action), not info/next_snapshot
                q_pre = current_snapshot[targeted_qid]
                sla_pre = SLA_THRESHOLDS[targeted_qid]

                # Use latency trend (ema_diff) to determine if intervention is needed
                # Proactive: low cost if latency trending upward (preventive action encouraged)
                # Conservative: high cost if latency stable/improving (don't disturb)
                ratio_pre = q_pre['lat_p95'] / sla_pre
                lat_ema_diff = q_pre.get('lat_ema_diff', 0.0)  # Normalized latency change

                # Decision logic:
                # 1. If SLA violated (ratio > margin_high) → low cost (must fix)
                # 2. If SLA met but trending bad (ema_diff > 0.05) → low cost (preventive)
                # 3. If SLA met and stable/improving → high cost (don't disturb)
                if ratio_pre > SLA_MARGIN_HIGH:
                    # Violated: must fix
                    action_cost = REWARD_ACTION_COST_SICK
                elif lat_ema_diff > 0.05:  # Latency increasing > 5% of normalized range
                    # Trending bad: preventive action encouraged
                    action_cost = REWARD_ACTION_COST_SICK
                else:
                    # Stable or improving: high cost to avoid unnecessary changes
                    action_cost = REWARD_ACTION_COST_HEALTHY
            else:
                action_cost = REWARD_ACTION_COST_HEALTHY  # Fallback
                targeted_qid = None
            
            # Apply cost and re-clip with tanh
            raw_reward_with_cost = info['raw_reward'] - action_cost
            reward = 2.5 * np.tanh(raw_reward_with_cost / 2.5)
            
            info['action_cost'] = action_cost
            info['action_cost_applied'] = True
            info['targeted_qid'] = targeted_qid
        else:
            info['action_cost'] = 0.0
            info['action_cost_applied'] = False
        
        # Check episode termination
        sla_total = int(info.get('sla_total', len(QIDS)) or len(QIDS))
        all_sla_met = len(info['sla_met']) == sla_total
        if all_sla_met:
            self.sla_streak += 1
        else:
            self.sla_streak = 0
        
        # IMPORTANT: Distinguish truncation (timeout) from termination (absorbing state)
        # This is a continuing task - traffic never "ends". Episodes are training windows only.
        # Truncation should still bootstrap (gamma * next_Q), termination should not.
        terminated = False  # No true terminal states in this environment
        if self.production_mode:
            # Production: never truncate, run continuously
            truncated = False
        else:
            # Training: truncate at horizon
            truncated = self.episode_step >= MAX_EPISODE_STEPS
        
        info['episode_step'] = self.episode_step
        info['sla_streak'] = self.sla_streak
        info['action_applied'] = action_applied
        info['terminated'] = terminated
        info['truncated'] = truncated
        info['pre_action_snapshot'] = current_snapshot
        if alt_name is not None:
            info['alt_used'] = alt_name  # String name for stdout/CSV
            info['alt_idx'] = alt_idx    # Numeric index for InfluxDB
        info['all_sla_met'] = all_sla_met
        
        # DATA VALIDITY CHECK:
        # Key insight: validity must key off action != 0, not action_applied
        # - If action == 0 (noop): relaxed (2/3 queues valid is OK)
        # - If action != 0: strict - ALL 3 queues valid AND action context AND action_applied
        #   (Failed actions should not be stored as they confuse learning)
        
        valid_count = sum(1 for qid in QIDS if next_snapshot[qid].get('data_valid', False))
        invalid_qs = [qid for qid in QIDS if not next_snapshot[qid].get('data_valid', False)]
        required_qids = list(self._required_qids_for_current_profile())
        required_valid_count = self._required_valid_count(next_snapshot)
        required_invalid_qs = self._required_invalid_queues(next_snapshot)
        min_required_valid = self._min_required_valid_count()
        
        # Action context check - for ANY non-zero action (not just applied ones)
        action_context_valid = True
        if action != 0:
            # Get the targeted queue from action mapping
            mapping = self.ACTION_MAP.get(action)
            if mapping == 'multi':
                # Multi-action: require at least one violating queue to have context
                action_context_valid = any(
                    (current_snapshot[qid].get('hot_src_ip') is not None and
                     current_snapshot[qid].get('hot_dst_ip') is not None and
                     current_snapshot[qid].get('bottleneck_sid') is not None)
                    for qid in required_qids
                    if self._is_sla_violated(qid, current_snapshot)
                )
                if not action_context_valid:
                    log.debug(f"[Step {self.episode_step}] Multi-action had no violating queues with context")
            elif mapping is not None:
                # Single-queue action: extract targeted queue ID
                targeted_qid = mapping[0]
                if targeted_qid not in required_qids:
                    action_context_valid = False
                    log.debug(
                        f"[Step {self.episode_step}] Action {action} targeted "
                        f"optional queue {targeted_qid} for profile "
                        f"{self.current_traffic_profile}"
                    )
                else:
                    # Check if the PRE-ACTION snapshot had valid routing context
                    q_pre = current_snapshot[targeted_qid]
                    has_context = (q_pre.get('hot_src_ip') is not None and
                                   q_pre.get('hot_dst_ip') is not None and
                                   q_pre.get('bottleneck_sid') is not None)
                    if not has_context:
                        action_context_valid = False
                        log.debug(f"[Step {self.episode_step}] Action {action} targeted queue {targeted_qid} lacked routing context")
        
        # Determine validity based on action type
        # RELAXED: Allow 2/3 valid queues for all actions (not just noop)
        # This reduces excessive data filtering in early training while maintaining quality
        if action == 0:
            # Noop: enough required queues valid for the active profile
            data_valid = required_valid_count >= min_required_valid
        else:
            # Non-zero action attempted:
            # Relaxed from "all 3 queues" to "at least 2 queues" to reduce sampling bias
            # Still require action context and successful application
            target_post_valid = True
            mapping = self.ACTION_MAP.get(action)
            if isinstance(mapping, tuple):
                target_post_valid = bool(
                    next_snapshot[mapping[0]].get('data_valid', False)
                )
            elif mapping == 'multi':
                rerouted_qids = getattr(self, "_last_rerouted_qids", [])
                target_post_valid = any(
                    next_snapshot[qid].get('data_valid', False)
                    for qid in rerouted_qids
                )
            data_valid = (
                required_valid_count >= min_required_valid
                and target_post_valid
                and action_context_valid
                and action_applied
            )
        
        info['data_valid'] = data_valid
        info['valid_count'] = valid_count
        info['invalid_queues'] = invalid_qs
        info['required_qids'] = required_qids
        info['required_valid_count'] = required_valid_count
        info['required_invalid_queues'] = required_invalid_qs
        info['telemetry_epoch_id'] = self._telemetry_epoch_id
        info['episode_start_telemetry_valid'] = self._episode_start_telemetry_valid
        info['telemetry_liveness_missing'] = (
            self._freshness_missing_required(
                self._last_telemetry_freshness,
                required_qids,
            )
            if required_invalid_qs else {}
        )
        
        if not data_valid:
            reasons = []
            if required_valid_count < min_required_valid:
                reasons.append(
                    f"required telemetry incomplete "
                    f"({required_valid_count}/{len(required_qids)}): "
                    f"queues {required_invalid_qs}"
                )
            elif valid_count < len(QIDS):
                reasons.append(
                    f"optional telemetry missing ({valid_count}/3): "
                    f"queues {invalid_qs}"
                )
            if action != 0 and not action_context_valid:
                reasons.append("action lacked routing context")
            if action != 0 and not action_applied:
                reasons.append("action was not applied (controller rejected)")
            log.warning(f"[Step {self.episode_step}] Invalid transition (action={action}) - {', '.join(reasons)}")
        
        # Build raw observation - ONLY update state stacks if data is valid
        # This prevents corrupting observation history with invalid/stale data
        raw_state = self._build_raw_state(next_snapshot)
        
        if data_valid:
            # Valid transition: update frame and action stacks normally
            self.frame_stack.append(raw_state)
            action_onehot = self._action_to_onehot(action)
            self.action_stack.append(action_onehot)
        else:
            # Invalid transition: keep stacks unchanged to preserve valid history
            # The returned next_state will still be built from current stacks
            log.debug(f"[Step {self.episode_step}] Skipping state stack update due to invalid data")
        
        # Build stacked state (464-dim)
        next_state = self._build_stacked_state()
        info['next_valid_mask'] = self._get_valid_actions(next_snapshot)
        self.last_snapshot = next_snapshot
        
        return next_state, reward, terminated, truncated, info
    
    def get_valid_actions(self) -> np.ndarray:
        """Get valid action mask for current state."""
        return self._get_valid_actions(self.last_snapshot)
    
    def write_training_metrics(self, step: int, agent_stats: Dict,
                                reward: float, action: int, info: Dict,
                                episode: int = 0):
        """
        Write training metrics to InfluxDB for Grafana monitoring.
        Measurement: rl_training

        Minimal progress-monitoring fields only. Detailed training/INT report
        data is written to local durable artifacts instead of InfluxDB to keep
        live database writes light.
        """
        try:
            detail = getattr(self, "training_influx_detail", "minimal")
            if detail == "off":
                return
            if self.write_api is None:
                return

            p = (
                Point("rl_training")
                .field("step", int(step))
                .field("episode", int(episode))
                .field("action", int(action))
                .field("reward", float(reward))
                .field("eps", float(agent_stats['eps']))
                .field("loss", float(agent_stats['avg_loss']))
                .field("buffer_size", int(agent_stats.get('buffer_size', 0)))
                .field("sla_met_count", len(info.get('sla_met', [])))
                .field("data_valid", int(info.get('data_valid', False)))
                .time(datetime.utcnow())
            )

            if 'q_max' in agent_stats:
                p = p.field("q_max", float(agent_stats['q_max']))
            if agent_stats.get('last_loss') is not None:
                p = p.field("last_loss", float(agent_stats['last_loss']))

            # Traffic profile
            if info.get('traffic_profile'):
                p = p.field("traffic_profile", str(info['traffic_profile']))

            self.write_api.write(bucket=self.bucket, org=self.org, record=[p])
        except Exception as e:
            log.debug(f"Failed to write training metrics: {e}")
    
    def write_episode_metrics(self, episode: int, episode_reward: float,
                               rolling_avg_100: float, traffic_profile: str):
        """Write episode summary metrics to InfluxDB.
        Measurement: rl_training (same as steps for unified visualization)

        Minimal fields: episode, episode_reward, reward_rolling_avg, traffic_profile
        """
        try:
            if getattr(self, "training_influx_detail", "minimal") == "off":
                return
            if self.write_api is None:
                return
            p = (
                Point("rl_training")
                .field("episode", int(episode))
                .field("episode_reward", float(episode_reward))
                .field("reward_rolling_avg", float(rolling_avg_100))
                .field("traffic_profile", str(traffic_profile))
                .time(datetime.utcnow())
            )
            self.write_api.write(bucket=self.bucket, org=self.org, record=[p])
        except Exception as e:
            log.debug(f"Failed to write episode metrics: {e}")
    
    def close(self):
        """Clean up resources."""
        # Shutdown query executor gracefully
        if hasattr(self, '_query_executor') and self._query_executor is not None:
            self._query_executor.shutdown(wait=False)
            self._query_executor = None

        # Stop traffic first
        if self.traffic_manager:
            try:
                self.traffic_manager.stop_traffic()
                log.info("Traffic stopped")
            except Exception as e:
                log.warning(f"Error stopping traffic: {e}")

        # Close controller Thrift connections
        if self.controller:
            try:
                self.controller.cleanup()
                log.info("Controller connections closed")
            except Exception as e:
                log.warning(f"Error closing controller: {e}")

        # Close InfluxDB connections
        try:
            if self.write_api is not None:
                self.write_api.close()
            if self.client is not None:
                self.client.close()
            if self.client is not None or self.write_api is not None:
                log.info("InfluxDB connections closed")
        except Exception as e:
            log.warning(f"Error closing InfluxDB: {e}")


# =============================================================================
#                         DURABLE TRAINING ARTIFACTS
# =============================================================================
def _json_safe(value: Any) -> Any:
    """Convert numpy/torch/path/container values into JSON-safe objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
        return [_json_safe(v) for v in value]
    return str(value)


def _file_sha256(path: str) -> Optional[str]:
    """Return SHA256 for a file if it exists."""
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> Dict[str, Any]:
    """Collect best-effort git metadata for reproducible training reports."""
    def run_git(args: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except Exception:
            return None

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
        "remote_origin": run_git(["remote", "get-url", "origin"]),
    }


def _sanitized_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Return CLI args safe for durable metadata without secrets."""
    safe = dict(vars(args))
    for key in ("influx_token",):
        if key in safe:
            safe[key] = "<set>" if safe[key] else None
    return safe


def _replay_write_position(replay_buffer) -> Optional[int]:
    """Return replay write pointer for single-buffer replay, if available."""
    tree = getattr(replay_buffer, "tree", None)
    return getattr(tree, "write", None)


def _replay_max_priority(replay_buffer) -> Optional[float]:
    """Return max replay priority across supported replay-buffer types."""
    if hasattr(replay_buffer, "max_priority"):
        return getattr(replay_buffer, "max_priority")
    buffers = getattr(replay_buffer, "buffers", None)
    if buffers:
        return max(
            (getattr(buf, "max_priority", 0.0) for buf in buffers.values()),
            default=None,
        )
    return None


class TrainingArtifactLogger:
    """Persist enough training data to rebuild a full report without InfluxDB.

    Files written under ``training_files/training_logs/<run_id>/``:
      - metadata.json: command, args, topology, git/source hashes, checkpoints.
      - step_metrics.csv: flat per-step training and INT-derived metrics.
      - int_snapshots.jsonl: full pre/post INT snapshot context per step.
      - episode_metrics.csv: per-episode reward/profile summaries.
      - checkpoints.csv: every checkpoint/sidecar path and agent state.
    """

    STEP_FIELDS = [
        "timestamp_utc", "run_id", "total_step", "agent_step_count",
        "eps_step_count", "episode", "episode_step", "reset_type",
        "reset_prob", "traffic_profile", "traffic_category", "stage_high",
        "profile_step_offset", "load_q0_mbps", "load_q1_mbps",
        "load_q7_mbps", "action", "action_name", "reward", "raw_reward",
        "eps", "beta", "avg_loss", "last_loss", "q_max", "buffer_size",
        "sla_met_count", "sla_violated_count", "sla_streak",
        "all_sla_met", "pressure", "data_valid", "valid_count",
        "invalid_queues", "required_qids", "required_valid_count",
        "required_invalid_queues", "telemetry_epoch_id",
        "episode_start_telemetry_valid", "telemetry_liveness_missing",
        "recovery_break", "action_applied",
        "action_cost", "action_cost_applied", "targeted_qid", "alt_used",
        "alt_idx", "multi_reroute_count", "terminated", "truncated",
    ] + [
        field
        for qid in QIDS
        for field in (
            f"q{qid}_latency_ms", f"q{qid}_drop_p95",
            f"q{qid}_util_pct", f"q{qid}_sla_ratio",
            f"q{qid}_reward_component", f"q{qid}_drop_penalty",
            f"q{qid}_data_valid", f"q{qid}_recovered_via_retry",
            f"q{qid}_telemetry_count_lat",
            f"q{qid}_telemetry_count_drop",
            f"q{qid}_telemetry_count_util",
            f"q{qid}_hot_src_ip", f"q{qid}_hot_dst_ip",
            f"q{qid}_bottleneck_sid", f"q{qid}_bottleneck_score",
            f"q{qid}_bottleneck_drop", f"q{qid}_bottleneck_lat",
            f"q{qid}_bottleneck_util", f"q{qid}_bottleneck_role",
            f"q{qid}_alternatives_count",
        )
    ]

    EPISODE_FIELDS = [
        "timestamp_utc", "run_id", "episode", "reset_type", "reset_prob",
        "start_total_step", "end_total_step", "episode_steps",
        "episode_reward", "rolling_avg_100", "traffic_profile",
        "traffic_category", "valid_learning_steps", "invalid_steps",
        "best_avg_reward",
    ]

    CHECKPOINT_FIELDS = [
        "timestamp_utc", "run_id", "tag", "kind", "checkpoint_path",
        "metadata_path", "sha256", "total_step", "episode",
        "agent_step_count", "eps_step_count", "eps", "beta",
        "replay_entries", "replay_write", "best_avg_reward",
        "rolling_avg_100",
    ]

    def __init__(
        self,
        args: argparse.Namespace,
        run_id: str,
        topology_builder=None,
        rules_dir: Optional[str] = None,
        buffer_capacity: Optional[int] = None,
        lr: Optional[float] = None,
    ):
        self.args = args
        self.run_id = run_id
        base_dir = Path(args.training_log_dir) if args.training_log_dir else (
            Path(args.save_dir) / "training_logs"
        )
        self.run_dir = base_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        normalize_artifact_permissions(self.run_dir, dir_mode=0o775)
        self.flush_every = max(
            1, int(getattr(args, "training_log_flush_every", 25) or 25)
        )
        self.snapshot_every = 1
        self.step_path = self.run_dir / "step_metrics.csv"
        self.snapshot_path = self.run_dir / "int_snapshots.jsonl"
        self.episode_path = self.run_dir / "episode_metrics.csv"
        self.checkpoint_path = self.run_dir / "checkpoints.csv"
        self.metadata_path = self.run_dir / "metadata.json"
        self.started_at_utc = datetime.utcnow().isoformat() + "Z"
        self.checkpoints: List[Dict[str, Any]] = []
        self._rows_since_flush = 0
        self.closed = False

        self._step_file = self.step_path.open("w", newline="", buffering=1024 * 1024)
        normalize_artifact_permissions(self.step_path, file_mode=0o664)
        self._step_writer = csv.DictWriter(
            self._step_file, fieldnames=self.STEP_FIELDS
        )
        self._step_writer.writeheader()

        self._episode_file = self.episode_path.open("w", newline="", buffering=256 * 1024)
        normalize_artifact_permissions(self.episode_path, file_mode=0o664)
        self._episode_writer = csv.DictWriter(
            self._episode_file, fieldnames=self.EPISODE_FIELDS
        )
        self._episode_writer.writeheader()

        self._checkpoint_file = self.checkpoint_path.open("w", newline="", buffering=64 * 1024)
        normalize_artifact_permissions(self.checkpoint_path, file_mode=0o664)
        self._checkpoint_writer = csv.DictWriter(
            self._checkpoint_file, fieldnames=self.CHECKPOINT_FIELDS
        )
        self._checkpoint_writer.writeheader()

        self._snapshot_file = self.snapshot_path.open("w", buffering=1024 * 1024)
        normalize_artifact_permissions(self.snapshot_path, file_mode=0o664)

        self.metadata: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at_utc": self.started_at_utc,
            "run_dir": str(self.run_dir),
            "command": " ".join(sys.argv),
            "argv": sys.argv,
            "args": _sanitized_args(args),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "git": _git_metadata(),
            "topology": {
                "config_path": args.config,
                "rules_dir": str(rules_dir) if rules_dir else None,
                "name": (
                    topology_builder.config.topology.name
                    if topology_builder and getattr(topology_builder, "config", None)
                    else None
                ),
            },
            "influxdb": {
                "url": args.influx_url,
                "org": args.influx_org,
                "bucket": args.influx_bucket,
                "token_recorded": bool(args.influx_token),
            },
            "source_hashes": {
                path: _file_sha256(path)
                for path in (
                    "rl_agent_4.py", "traffic_generator.py", "controller.py",
                    "network.py", "logging_config.py", args.config,
                )
                if path
            },
            "hyperparameters": {
                "learning_rate": lr,
                "gamma": GAMMA,
                "batch_size": BATCH_SIZE,
                "replay_capacity": buffer_capacity,
                "min_replay_size": MIN_REPLAY_SIZE,
                "epsilon_start": EPS_START,
                "epsilon_end": EPS_END,
                "epsilon_decay_steps": EPS_DECAY_STEPS,
                "per_alpha": PER_ALPHA,
                "per_beta_start": PER_BETA_START,
                "per_beta_end": PER_BETA_END,
                "per_beta_steps": PER_BETA_STEPS,
                "target_update_freq": TARGET_UPDATE_FREQ,
                "window_seconds": WINDOW_SECONDS,
                "delay_after_action": DELAY_AFTER_ACTION,
                "delay_no_action": DELAY_NO_ACTION,
                "cooldown_seconds": COOLDOWN_SECONDS,
                "sla_thresholds": SLA_THRESHOLDS,
                "qids": QIDS,
                "action_names": {idx: action_to_name(idx) for idx in range(ACTION_DIM)},
                "reward": {
                    "sla_met_scale": REWARD_SLA_MET_SCALE,
                    "sla_violated_scale": REWARD_SLA_VIOLATED_SCALE,
                    "drop_penalty": REWARD_DROP_PENALTY,
                    "action_cost_healthy": REWARD_ACTION_COST_HEALTHY,
                    "action_cost_sick": REWARD_ACTION_COST_SICK,
                    "sla_margin_low": SLA_MARGIN_LOW,
                    "sla_margin_high": SLA_MARGIN_HIGH,
                },
            },
            "traffic": {
                "weights_raw": args.traffic_weights,
                "profile_weights_raw": args.traffic_profile_weights,
                "fixed_profile": args.traffic_profile,
                "seed": args.seed,
            },
            "artifacts": {
                "step_metrics_csv": str(self.step_path),
                "int_snapshots_jsonl": str(self.snapshot_path),
                "episode_metrics_csv": str(self.episode_path),
                "checkpoints_csv": str(self.checkpoint_path),
            },
            "logging_policy": {
                "local_artifacts": "single durable run directory; no extra Influx stream",
                "flush_every_steps": self.flush_every,
                "int_snapshot_every_steps": 1,
                "influx_detail": getattr(args, "training_influx_detail", "minimal"),
                "checkpoint_sidecars": "one metadata JSON beside each checkpoint",
            },
            "checkpoints": self.checkpoints,
        }
        self._write_json(self.metadata_path, self.metadata)
        command_path = self.run_dir / "command.txt"
        command_path.write_text(self.metadata["command"] + "\n")
        normalize_artifact_permissions(command_path, file_mode=0o664)
        log.info(f"Training artifacts will be written to {self.run_dir}")

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
        normalize_artifact_permissions(path, file_mode=0o664)

    def _flush_if_needed(self, *, force: bool = False) -> None:
        if not force and self._rows_since_flush < self.flush_every:
            return
        for fh in (self._step_file, self._snapshot_file):
            try:
                fh.flush()
            except Exception:
                pass
        self._rows_since_flush = 0

    @staticmethod
    def _snapshot_summary(snapshot: Optional[Dict[int, Dict]]) -> Dict[str, Dict]:
        if not snapshot:
            return {}
        out: Dict[str, Dict] = {}
        for qid in QIDS:
            q = snapshot.get(qid, {})
            out[str(qid)] = {
                key: q.get(key)
                for key in (
                    "lat_p95", "drop_p95", "util_p95", "data_valid",
                    "profile_required", "telemetry_optional",
                    "recovered_via_retry", "telemetry_counts",
                    "hot_src_ip", "hot_dst_ip",
                    "path_nodes", "bottleneck_sid", "bottleneck_score",
                    "bottleneck_drop", "bottleneck_lat", "bottleneck_util",
                    "bottleneck_role", "alternatives", "alt_exists",
                    "lat_ema", "lat_ema_diff", "transitioning",
                )
            }
        return out

    def _traffic_state(self, env: QoSRoutingEnv) -> Dict[str, Any]:
        tm = getattr(env, "traffic_manager", None)
        if tm is None:
            return {}
        return {
            "profile": getattr(tm, "current_profile_name", ""),
            "category": getattr(tm, "current_profile_category", ""),
            "current_load": getattr(tm, "current_load", {}),
            "source_load": getattr(tm, "_source_load", {}),
            "stage_high": getattr(tm, "_shaped_stage_high", None),
            "profile_step_offset": getattr(tm, "_profile_step_offset", None),
        }

    def log_step(
        self,
        *,
        total_step: int,
        episode: int,
        reset_type: str,
        reset_prob: float,
        agent: DQNAgent,
        agent_stats: Dict[str, Any],
        env: QoSRoutingEnv,
        action: int,
        reward: float,
        info: Dict[str, Any],
        valid_mask: np.ndarray,
        loss: Optional[float],
    ) -> None:
        traffic = self._traffic_state(env)
        loads = traffic.get("current_load", {})
        post_snapshot = getattr(env, "last_snapshot", {})
        per_queue = info.get("per_queue", {})
        row: Dict[str, Any] = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "total_step": total_step,
            "agent_step_count": agent.step_count,
            "eps_step_count": agent.eps_step_count,
            "episode": episode,
            "episode_step": info.get("episode_step"),
            "reset_type": reset_type,
            "reset_prob": reset_prob,
            "traffic_profile": traffic.get("profile") or info.get("traffic_profile", ""),
            "traffic_category": traffic.get("category", ""),
            "stage_high": traffic.get("stage_high"),
            "profile_step_offset": traffic.get("profile_step_offset"),
            "load_q0_mbps": loads.get(0, 0.0),
            "load_q1_mbps": loads.get(1, 0.0),
            "load_q7_mbps": loads.get(7, 0.0),
            "action": action,
            "action_name": action_to_name(action),
            "reward": reward,
            "raw_reward": info.get("raw_reward", reward),
            "eps": agent_stats.get("eps"),
            "beta": agent.beta,
            "avg_loss": agent_stats.get("avg_loss"),
            "last_loss": agent_stats.get("last_loss") if agent_stats.get("last_loss") is not None else "",
            "q_max": agent_stats.get("q_max"),
            "buffer_size": agent_stats.get("buffer_size"),
            "sla_met_count": len(info.get("sla_met", [])),
            "sla_violated_count": len(info.get("sla_violated", [])),
            "sla_streak": info.get("sla_streak"),
            "all_sla_met": info.get("all_sla_met"),
            "pressure": info.get("pressure"),
            "data_valid": info.get("data_valid"),
            "valid_count": info.get("valid_count"),
            "invalid_queues": json.dumps(_json_safe(info.get("invalid_queues", []))),
            "required_qids": json.dumps(_json_safe(info.get("required_qids", []))),
            "required_valid_count": info.get("required_valid_count", ""),
            "required_invalid_queues": json.dumps(_json_safe(info.get("required_invalid_queues", []))),
            "telemetry_epoch_id": info.get("telemetry_epoch_id", getattr(env, "_telemetry_epoch_id", "")),
            "episode_start_telemetry_valid": info.get(
                "episode_start_telemetry_valid",
                getattr(env, "_episode_start_telemetry_valid", ""),
            ),
            "telemetry_liveness_missing": json.dumps(_json_safe(info.get("telemetry_liveness_missing", {}))),
            "recovery_break": info.get("recovery_break", False),
            "action_applied": info.get("action_applied"),
            "action_cost": info.get("action_cost"),
            "action_cost_applied": info.get("action_cost_applied"),
            "targeted_qid": info.get("targeted_qid", ""),
            "alt_used": info.get("alt_used", ""),
            "alt_idx": info.get("alt_idx", ""),
            "multi_reroute_count": info.get("multi_reroute_count", 0),
            "terminated": info.get("terminated"),
            "truncated": info.get("truncated"),
        }
        for qid in QIDS:
            q = post_snapshot.get(qid, {}) if post_snapshot else {}
            pq = per_queue.get(qid, {})
            row.update({
                f"q{qid}_latency_ms": q.get("lat_p95", pq.get("lat")),
                f"q{qid}_drop_p95": q.get("drop_p95", pq.get("drop")),
                f"q{qid}_util_pct": q.get("util_p95", pq.get("util")),
                f"q{qid}_sla_ratio": pq.get("ratio"),
                f"q{qid}_reward_component": pq.get("component"),
                f"q{qid}_drop_penalty": pq.get("drop_penalty"),
                f"q{qid}_data_valid": q.get("data_valid"),
                f"q{qid}_recovered_via_retry": q.get("recovered_via_retry"),
                f"q{qid}_telemetry_count_lat": (q.get("telemetry_counts") or {}).get("lat", 0),
                f"q{qid}_telemetry_count_drop": (q.get("telemetry_counts") or {}).get("drop", 0),
                f"q{qid}_telemetry_count_util": (q.get("telemetry_counts") or {}).get("util", 0),
                f"q{qid}_hot_src_ip": q.get("hot_src_ip"),
                f"q{qid}_hot_dst_ip": q.get("hot_dst_ip"),
                f"q{qid}_bottleneck_sid": q.get("bottleneck_sid"),
                f"q{qid}_bottleneck_score": q.get("bottleneck_score"),
                f"q{qid}_bottleneck_drop": q.get("bottleneck_drop"),
                f"q{qid}_bottleneck_lat": q.get("bottleneck_lat"),
                f"q{qid}_bottleneck_util": q.get("bottleneck_util"),
                f"q{qid}_bottleneck_role": q.get("bottleneck_role"),
                f"q{qid}_alternatives_count": len(q.get("alternatives", []) or []),
            })
        self._step_writer.writerow({
            k: _json_safe(row.get(k, "")) for k in self.STEP_FIELDS
        })
        self._rows_since_flush += 1

        snapshot_record = {
            "timestamp_utc": row["timestamp_utc"],
            "run_id": self.run_id,
            "total_step": total_step,
            "episode": episode,
            "episode_step": info.get("episode_step"),
            "traffic": traffic,
            "action": {
                "index": action,
                "name": action_to_name(action),
                "valid_mask": [bool(x) for x in valid_mask.tolist()],
                "applied": info.get("action_applied"),
                "alt_used": info.get("alt_used"),
                "alt_idx": info.get("alt_idx"),
                "targeted_qid": info.get("targeted_qid"),
            },
            "reward": {
                "reward": reward,
                "raw_reward": info.get("raw_reward", reward),
                "pressure": info.get("pressure"),
                "sla_met": info.get("sla_met", []),
                "sla_violated": info.get("sla_violated", []),
                "reward_qids": info.get("reward_qids", []),
                "sla_total": info.get("sla_total"),
                "per_queue": info.get("per_queue", {}),
            },
            "agent": {
                "eps": agent.eps,
                "beta": agent.beta,
                "step_count": agent.step_count,
                "eps_step_count": agent.eps_step_count,
                "buffer_size": len(agent.replay_buffer),
                "loss": loss,
                "stats": agent_stats,
            },
            "data_valid": info.get("data_valid"),
            "valid_count": info.get("valid_count"),
            "invalid_queues": info.get("invalid_queues", []),
            "required_qids": info.get("required_qids", []),
            "required_valid_count": info.get("required_valid_count"),
            "required_invalid_queues": info.get("required_invalid_queues", []),
            "telemetry_epoch_id": info.get("telemetry_epoch_id", getattr(env, "_telemetry_epoch_id", None)),
            "episode_start_telemetry_valid": info.get(
                "episode_start_telemetry_valid",
                getattr(env, "_episode_start_telemetry_valid", None),
            ),
            "telemetry_liveness_missing": info.get("telemetry_liveness_missing", {}),
            "recovery_break": info.get("recovery_break", False),
            "pre_action_snapshot": self._snapshot_summary(
                info.get("pre_action_snapshot")
            ),
            "post_action_snapshot": self._snapshot_summary(post_snapshot),
        }
        self._snapshot_file.write(
            json.dumps(_json_safe(snapshot_record), separators=(",", ":")) + "\n"
        )

        self._flush_if_needed()

    def log_episode(
        self,
        *,
        episode: int,
        reset_type: str,
        reset_prob: float,
        start_total_step: int,
        end_total_step: int,
        episode_steps: int,
        episode_reward: float,
        rolling_avg_100: float,
        env: QoSRoutingEnv,
        valid_learning_steps: int,
        invalid_steps: int,
        best_avg_reward: float,
    ) -> None:
        row = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "episode": episode,
            "reset_type": reset_type,
            "reset_prob": reset_prob,
            "start_total_step": start_total_step,
            "end_total_step": end_total_step,
            "episode_steps": episode_steps,
            "episode_reward": episode_reward,
            "rolling_avg_100": rolling_avg_100,
            "traffic_profile": getattr(env, "current_traffic_profile", ""),
            "traffic_category": getattr(env, "current_traffic_category", ""),
            "valid_learning_steps": valid_learning_steps,
            "invalid_steps": invalid_steps,
            "best_avg_reward": best_avg_reward,
        }
        self._episode_writer.writerow({k: _json_safe(row.get(k, "")) for k in self.EPISODE_FIELDS})
        self._episode_file.flush()
        self._flush_if_needed(force=True)

    def log_checkpoint(
        self,
        *,
        tag: str,
        kind: str,
        checkpoint_path: str,
        total_step: int,
        episode: int,
        agent: DQNAgent,
        best_avg_reward: float,
        rolling_avg_100: Optional[float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata_path = f"{checkpoint_path}.metadata.json"
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "tag": tag,
            "kind": kind,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "total_step": total_step,
            "episode": episode,
            "agent": {
                "step_count": agent.step_count,
                "eps_step_count": agent.eps_step_count,
                "eps": agent.eps,
                "beta": agent.beta,
                "replay_entries": len(agent.replay_buffer),
                "replay_write": _replay_write_position(agent.replay_buffer),
                "replay_max_priority": _replay_max_priority(agent.replay_buffer),
            },
            "training": {
                "best_avg_reward": best_avg_reward,
                "rolling_avg_100": rolling_avg_100,
                "run_metadata_path": str(self.metadata_path),
                "run_dir": str(self.run_dir),
            },
            "extra": extra or {},
        }
        self._write_json(Path(metadata_path), payload)
        row = {
            "timestamp_utc": payload["created_at_utc"],
            "run_id": self.run_id,
            "tag": tag,
            "kind": kind,
            "checkpoint_path": checkpoint_path,
            "metadata_path": metadata_path,
            "sha256": payload["checkpoint_sha256"],
            "total_step": total_step,
            "episode": episode,
            "agent_step_count": agent.step_count,
            "eps_step_count": agent.eps_step_count,
            "eps": agent.eps,
            "beta": agent.beta,
            "replay_entries": len(agent.replay_buffer),
            "replay_write": _replay_write_position(agent.replay_buffer),
            "best_avg_reward": best_avg_reward,
            "rolling_avg_100": rolling_avg_100,
        }
        self._checkpoint_writer.writerow({k: _json_safe(row.get(k, "")) for k in self.CHECKPOINT_FIELDS})
        self._checkpoint_file.flush()
        self.checkpoints.append({
            "tag": tag,
            "kind": kind,
            "checkpoint_path": checkpoint_path,
            "metadata_path": metadata_path,
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "created_at_utc": payload["created_at_utc"],
            "total_step": total_step,
            "episode": episode,
            "agent_step_count": agent.step_count,
        })
        self.metadata["checkpoints"] = self.checkpoints
        self._write_json(self.metadata_path, self.metadata)
        self._flush_if_needed(force=True)

    def finalize(
        self,
        *,
        interrupted: bool,
        agent: Optional[DQNAgent] = None,
        total_steps: Optional[int] = None,
        episodes: Optional[int] = None,
        best_avg_reward: Optional[float] = None,
        final_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.closed:
            return
        self.metadata.update({
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "interrupted": interrupted,
            "total_environment_steps": total_steps,
            "episodes": episodes,
            "best_avg_reward": best_avg_reward,
            "final_stats": final_stats,
        })
        if agent is not None:
            self.metadata["final_agent_state"] = {
                "step_count": agent.step_count,
                "eps_step_count": agent.eps_step_count,
                "eps": agent.eps,
                "beta": agent.beta,
                "replay_entries": len(agent.replay_buffer),
                "replay_write": _replay_write_position(agent.replay_buffer),
            }
        self._write_json(self.metadata_path, self.metadata)
        self._flush_if_needed(force=True)
        for fh in (
            self._step_file, self._episode_file,
            self._checkpoint_file, self._snapshot_file,
        ):
            try:
                fh.close()
            except Exception:
                pass
        self.closed = True


# =============================================================================
#                              TRAINING LOOP
# =============================================================================
def train(args):
    """Main training loop."""
    global MAX_EPISODE_STEPS
    MAX_EPISODE_STEPS = args.max_episode_steps
    
    interrupted = False

    
    # Determine learning rate (override or default)
    lr = args.lr if args.lr is not None else LR
    buffer_capacity = args.buffer_capacity if args.multi_buffer else REPLAY_CAPACITY

    log.info("=" * 60)
    log.info("Starting RL Training - DQN Agent v4 (Stacked Obs + Actions)")
    log.info("=" * 60)
    log.info(f"Configuration:")
    log.info(f"  State dim: {STATE_DIM} ({RAW_STATE_DIM} obs * {STACK_SIZE} + {ACTION_DIM} act * {STACK_SIZE})")
    log.info(f"  Action dim: {ACTION_DIM} (one-hot encoded in state)")
    log.info(f"  Hidden dim: {HIDDEN_DIM}")
    log.info(f"  Learning rate: {lr}, Gamma: {GAMMA}")
    log.info(f"  Batch size: {BATCH_SIZE}, Replay capacity: {buffer_capacity}")
    log.info(f"  Min replay: {MIN_REPLAY_SIZE}")
    log.info(f"  Epsilon: {EPS_START} -> {EPS_END} over {EPS_DECAY_STEPS} steps")
    log.info(f"  Timing: Window={WINDOW_SECONDS}s, Delay={DELAY_AFTER_ACTION}s, Cooldown={COOLDOWN_SECONDS}s")
    log.info(f"  Max steps: {args.steps}, Max episode steps: {MAX_EPISODE_STEPS}")
    if args.multi_buffer:
        log.info(f"  Multi-buffer: enabled (balanced_sampling={args.balanced_sampling})")
    if args.ewc_lambda > 0 or args.ewc_file:
        log.info(f"  EWC: lambda={args.ewc_lambda}, file={args.ewc_file}")
    if args.compute_ewc:
        log.info(f"  EWC: Will compute Fisher matrix after training")
    log.info("=" * 60)
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    # Load topology if config specified
    topology_builder, rules_dir = load_topology(args)
    if rules_dir:
        log.info(f"  Rules dir: {rules_dir}")

    # Initialize environment and agent
    env = QoSRoutingEnv(
        args.influx_bucket, args.influx_token,
        args.influx_org, args.influx_url,
        topology_builder=topology_builder,
        rules_dir=rules_dir,
        config_path=args.config,
        traffic_seed=args.seed,
        telemetry_backend=args.telemetry_backend,
        telemetry_cache_socket=args.telemetry_cache_socket,
        telemetry_cache_timeout=args.telemetry_cache_timeout,
    )
    env.training_influx_detail = args.training_influx_detail
    env.telemetry_liveness_enabled = not args.disable_telemetry_liveness_gate
    env.telemetry_liveness_retries = args.telemetry_liveness_retries
    env.telemetry_liveness_interval_seconds = args.telemetry_liveness_interval
    env.telemetry_liveness_restarts = args.telemetry_liveness_restarts
    agent = DQNAgent(
        STATE_DIM, ACTION_DIM, device,
        lr=lr,
        multi_buffer=args.multi_buffer,
        buffer_capacity=buffer_capacity,
        balanced_sampling=args.balanced_sampling
    )

    # Set topology for multi-buffer mode
    if args.multi_buffer:
        topo_name = topology_builder.config.topology.name if topology_builder else "default"
        agent.set_topology(topo_name)

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = args.resume if args.resume.endswith('.pth') else resolve_checkpoint_path(args.save_dir, args.resume)
        if resume_path and os.path.exists(resume_path):
            agent.load(resume_path)
            log.info(f"Resumed training from {resume_path}")

            # Override epsilon if specified
            if args.resume_eps is not None:
                agent.eps = args.resume_eps
                # Calculate eps_step_count to match the desired starting epsilon
                # eps = EPS_END + (EPS_START - EPS_END) * (1 - progress)
                # Solving for progress: progress = 1 - (eps - EPS_END) / (EPS_START - EPS_END)
                progress = 1.0 - (args.resume_eps - EPS_END) / (EPS_START - EPS_END)
                agent.eps_step_count = int(progress * EPS_DECAY_STEPS)
                log.info(f"Reset epsilon to {agent.eps} for resume training (eps_step_count={agent.eps_step_count}, keeping global step count {agent.step_count})")
        else:
            log.warning(f"Checkpoint not found: {resume_path}, starting fresh")

    # Load EWC data if specified (for multi-topology continual learning)
    if args.ewc_file:
        ewc_path = args.ewc_file
        # Support glob patterns and 'latest' keyword for EWC files
        if not os.path.exists(ewc_path):
            # Try glob pattern for timestamped EWC files
            # Pattern: YYYYMMDD-HHMMSS-ewc.pth or user-provided glob
            if '*' in ewc_path:
                # User provided a glob pattern
                matching_ewc = sorted(glob.glob(ewc_path), reverse=True)
            else:
                # Try standard timestamped pattern in save_dir
                pattern = os.path.join(args.save_dir, "*-ewc.pth")
                matching_ewc = sorted(glob.glob(pattern), reverse=True)

            if matching_ewc:
                ewc_path = matching_ewc[0]
                log.info(f"Found {len(matching_ewc)} EWC files, using latest: {os.path.basename(ewc_path)}")

        if os.path.exists(ewc_path):
            agent.load_ewc(ewc_path)
            agent.ewc_lambda = args.ewc_lambda
            log.info(f"EWC enabled with lambda={args.ewc_lambda}")
        else:
            log.warning(f"EWC file not found: {args.ewc_file}, EWC disabled.")
    elif args.ewc_lambda > 0 and not args.ewc_file:
        log.warning("EWC lambda > 0 but no --ewc-file specified. EWC disabled.")

    # Parse traffic weights if specified
    category_weights = None
    if args.traffic_weights:
        category_weights = {}
        for item in args.traffic_weights.split(','):
            k, v = item.split(':')
            category_weights[k.strip()] = float(v.strip())
        env.traffic_category_weights = category_weights
        log.info(f"Using traffic weights: {category_weights}")

    profile_weights = None
    if args.traffic_profile_weights:
        profile_weights = {}
        for item in args.traffic_profile_weights.split(','):
            k, v = item.split(':')
            profile = k.strip()
            if profile not in TrafficManager.TRAFFIC_PROFILES:
                raise ValueError(
                    f"Unknown traffic profile {profile!r}; valid profiles: "
                    f"{', '.join(TrafficManager.TRAFFIC_PROFILES)}"
                )
            profile_weights[profile] = float(v.strip())
        env.traffic_profile_weights = profile_weights
        if category_weights:
            log.warning(
                "--traffic-profile-weights overrides --traffic-weights for randomized training"
            )
        log.info(f"Using traffic profile weights: {profile_weights}")
    
    # Parse fixed traffic profile if specified (overrides weights)
    if args.traffic_profile:
        env.fixed_traffic_profile = args.traffic_profile
        log.info(f"Using FIXED traffic profile: {args.traffic_profile}")

    # Durable training artifacts for post-hoc reporting even if terminal logs
    # or InfluxDB retention are lost.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_logger = TrainingArtifactLogger(
        args,
        run_id=run_id,
        topology_builder=topology_builder,
        rules_dir=rules_dir,
        buffer_capacity=buffer_capacity,
        lr=lr,
    )
    
    # Training state
    total_steps = 0
    episode = 0
    best_avg_reward = -float('inf')
    episode_rewards = deque(maxlen=100)  # Track episode rewards for rolling average (bounded)
    force_recovery_reset_next = False
    
    # Checkpoints
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_steps = {
        int(args.steps * 0.25): '25pct',
        int(args.steps * 0.50): '50pct',
        int(args.steps * 0.75): '75pct',
    }

    try:
        while total_steps < args.steps:
            episode += 1
            
            # === CURRICULUM-BASED RESET STRATEGY ===
            # Early training: mostly baseline-starts (learn to optimize from clean state)
            # Late training: mostly warm-starts (learn stability/recovery)
            # Schedule: 90% reset at start → 60% midway → 30% at end
            if args.no_warm_start:
                do_reset = True
                reset_prob = 1.0
            else:
                progress = total_steps / args.steps
                reset_prob = (
                    args.reset_prob_start
                    + (args.reset_prob_end - args.reset_prob_start) * progress
                )
                reset_prob = max(0.0, min(1.0, reset_prob))
                do_reset = random.random() < reset_prob

            if force_recovery_reset_next:
                do_reset = True
                reset_prob = 1.0
                force_recovery_reset_next = False
            
            cooldown = (
                args.baseline_cooldown_seconds
                if do_reset else args.warm_cooldown_seconds
            )
            state = env.reset(force_reset=do_reset, cooldown_seconds=cooldown)
            episode_reward = 0.0
            episode_steps = 0
            episode_start_step = total_steps
            episode_valid_learning_steps = 0
            episode_invalid_steps = 0
            episode_invalid_streak = 0
            done = False
            
            reset_type = "BASELINE" if do_reset else "WARM"
            log.info(f"\n{'='*50}")
            log.info(f"Episode {episode} [{reset_type}] (total steps: {total_steps}, reset_prob={reset_prob:.2f})")
            log.info(f"Traffic profile: {env.current_traffic_profile} ({env.current_traffic_category})")
            
            while not done and total_steps < args.steps:
                total_steps += 1
                episode_steps += 1
                
                # Get valid actions and select action
                valid_mask = env.get_valid_actions()
                action = agent.select_action(state, valid_mask, explore=True)
                
                # Take step - returns terminated, truncated separately
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  # For loop control
                
                # Only store experience and update epsilon if telemetry data is valid
                # Invalid data (fallback values: 2x SLA, DROP_CAP, UTIL_CAP) would poison the replay buffer
                # CRITICAL: Store 'terminated' not 'done' - truncation should still bootstrap!
                if info.get('data_valid', False):
                    agent.push_experience(
                        state,
                        action,
                        reward,
                        next_state,
                        terminated,
                        info.get('next_valid_mask'),
                    )
                    agent.update_epsilon()
                    episode_valid_learning_steps += 1
                    episode_invalid_streak = 0
                else:
                    episode_invalid_steps += 1
                    episode_invalid_streak += 1
                    log.debug(f"[Step {total_steps}] Skipping experience storage - invalid telemetry")

                if (
                    episode_invalid_streak >= args.invalid_recovery_streak
                    and args.invalid_recovery_streak > 0
                ):
                    log.warning(
                        f"[Recovery] Ending episode {episode} after "
                        f"{episode_invalid_streak} consecutive invalid "
                        "transitions; next episode will force baseline reset "
                        "and wait for fresh required telemetry"
                    )
                    log.warning(
                        "[Recovery] Telemetry missing at break: "
                        f"{info.get('telemetry_liveness_missing', {})}"
                    )
                    info['recovery_break'] = True
                    info['truncated'] = True
                    done = True
                    force_recovery_reset_next = True
                
                # Always try to train from existing valid experiences in the buffer
                loss = agent.train_step()
                
                episode_reward += reward
                state = next_state
                
                # Logging
                stats = agent.get_stats()
                
                if total_steps % args.log_every == 0:
                    loss_str = f"{stats['last_loss']:.4f}" if stats['last_loss'] is not None else "N/A"
                    log.info(
                        f"[Step {total_steps}] "
                        f"action={action_to_name(action):8s} "
                        f"reward={reward:+.2f} "
                        f"eps={stats['eps']:.3f} "
                        f"buffer={stats['buffer_size']:5d} "
                        f"loss={loss_str:>8} "
                        f"sla={len(info['sla_met'])}/"
                        f"{info.get('sla_total', 3)} "
                        f"streak={info['sla_streak']}"
                    )
                
                # Write metrics to InfluxDB
                info['traffic_profile'] = env.current_traffic_profile
                env.write_training_metrics(total_steps, stats, reward, action, info, episode=episode)
                artifact_logger.log_step(
                    total_step=total_steps,
                    episode=episode,
                    reset_type=reset_type,
                    reset_prob=reset_prob,
                    agent=agent,
                    agent_stats=stats,
                    env=env,
                    action=action,
                    reward=reward,
                    info=info,
                    valid_mask=valid_mask,
                    loss=loss,
                )

                # Save checkpoints
                if total_steps in checkpoint_steps:
                    tag = checkpoint_steps[total_steps]
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = os.path.join(args.save_dir, f"{timestamp}-dqn_v4_{tag}.pth")
                    agent.save(path)
                    artifact_logger.log_checkpoint(
                        tag=tag,
                        kind="scheduled",
                        checkpoint_path=path,
                        total_step=total_steps,
                        episode=episode,
                        agent=agent,
                        best_avg_reward=best_avg_reward,
                        rolling_avg_100=(
                            sum(episode_rewards) / len(episode_rewards)
                            if episode_rewards else None
                        ),
                    )
                    log.info(f"Checkpoint saved: {path}")
            
            # Episode summary - compute episode reward (avg reward per step)
            ep_reward = episode_reward / episode_steps if episode_steps > 0 else 0.0
            episode_rewards.append(ep_reward)
            
            # Rolling 100-episode average (deque is already bounded to 100)
            rolling_100 = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            
            log.info(
                f"Episode {episode} finished: "
                f"steps={episode_steps}, "
                f"episode_reward={ep_reward:.3f}, "
                f"rolling_avg_100={rolling_100:.3f}"
            )
            
            # Write episode metrics to InfluxDB
            env.write_episode_metrics(
                episode, ep_reward, rolling_100, env.current_traffic_profile
            )
            artifact_logger.log_episode(
                episode=episode,
                reset_type=reset_type,
                reset_prob=reset_prob,
                start_total_step=episode_start_step,
                end_total_step=total_steps,
                episode_steps=episode_steps,
                episode_reward=ep_reward,
                rolling_avg_100=rolling_100,
                env=env,
                valid_learning_steps=episode_valid_learning_steps,
                invalid_steps=episode_invalid_steps,
                best_avg_reward=best_avg_reward,
            )
            
            # Track best model (based on rolling 100-episode average)
            if rolling_100 > best_avg_reward and len(episode_rewards) >= 50:
                best_avg_reward = rolling_100
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                new_path = os.path.join(args.save_dir, f"{timestamp}-dqn_v4_best.pth")

                # Find old best checkpoints (to delete after successful save)
                old_bests = glob.glob(os.path.join(args.save_dir, "*-dqn_v4_best.pth"))

                # Save new checkpoint first
                agent.save(new_path)
                artifact_logger.log_checkpoint(
                    tag="best",
                    kind="best",
                    checkpoint_path=new_path,
                    total_step=total_steps,
                    episode=episode,
                    agent=agent,
                    best_avg_reward=best_avg_reward,
                    rolling_avg_100=rolling_100,
                )
                log.info(f"New best model saved: rolling_avg_100={best_avg_reward:.3f}")

                # Delete previous best checkpoint(s) after successful save
                for old_path in old_bests:
                    try:
                        os.remove(old_path)
                        log.info(f"Deleted old best checkpoint: {old_path}")
                        old_meta = f"{old_path}.metadata.json"
                        if os.path.exists(old_meta):
                            os.remove(old_meta)
                            log.info(f"Deleted old best metadata: {old_meta}")
                    except OSError as e:
                        log.warning(f"Failed to delete old checkpoint {old_path}: {e}")
    
    except KeyboardInterrupt:
        log.info("\nTraining interrupted by user")
        interrupted = True
    
    finally:
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(args.save_dir, f"{timestamp}-dqn_v4_final.pth")
        agent.save(path)
        artifact_logger.log_checkpoint(
            tag="final",
            kind="final",
            checkpoint_path=path,
            total_step=total_steps,
            episode=episode,
            agent=agent,
            best_avg_reward=best_avg_reward,
            rolling_avg_100=(
                sum(episode_rewards) / len(episode_rewards)
                if episode_rewards else None
            ),
        )

        # Compute and save EWC Fisher matrix if requested (for next topology)
        if args.compute_ewc and not interrupted:
            ewc_path = os.path.join(args.save_dir, f"{timestamp}-ewc.pth")
            log.info("Computing EWC Fisher matrix for next topology...")
            agent.save_ewc(ewc_path, env, EWC_FISHER_SAMPLES)
            artifact_logger.log_checkpoint(
                tag="ewc",
                kind="ewc",
                checkpoint_path=ewc_path,
                total_step=total_steps,
                episode=episode,
                agent=agent,
                best_avg_reward=best_avg_reward,
                rolling_avg_100=(
                    sum(episode_rewards) / len(episode_rewards)
                    if episode_rewards else None
                ),
                extra={"fisher_samples": EWC_FISHER_SAMPLES},
            )

        env.close()
        artifact_logger.finalize(
            interrupted=interrupted,
            agent=agent,
            total_steps=total_steps,
            episodes=episode,
            best_avg_reward=best_avg_reward,
            final_stats=agent.get_stats(),
        )

    log.info("\nTraining complete!")
    log.info(f"Final stats: {agent.get_stats()}")
    
    if interrupted:
        sys.exit(130)


def evaluate(args):
    """Evaluation loop (no training)."""
    log.info("=" * 60)
    log.info("Starting RL Evaluation - DQN Agent v4")
    log.info("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load topology if config specified
    topology_builder, rules_dir = load_topology(args)

    # Initialize environment and agent
    env = QoSRoutingEnv(
        args.influx_bucket, args.influx_token,
        args.influx_org, args.influx_url,
        verbose=args.verbose,
        topology_builder=topology_builder,
        rules_dir=rules_dir,
        config_path=args.config,
        traffic_seed=args.seed,
        telemetry_backend=args.telemetry_backend,
        telemetry_cache_socket=args.telemetry_cache_socket,
        telemetry_cache_timeout=args.telemetry_cache_timeout,
    )
    env.telemetry_liveness_enabled = not args.disable_telemetry_liveness_gate
    env.telemetry_liveness_retries = args.telemetry_liveness_retries
    env.telemetry_liveness_interval_seconds = args.telemetry_liveness_interval
    env.telemetry_liveness_restarts = args.telemetry_liveness_restarts
    agent = DQNAgent(STATE_DIM, ACTION_DIM, device)

    # Load weights - find latest checkpoint with datetime prefix or fallback to legacy
    weights_path = resolve_checkpoint_path(args.save_dir, args.weights_tag)
    if not weights_path:
        log.error(f"Weights file not found for tag: {args.weights_tag}")
        return
    log.info(f"Using checkpoint: {os.path.basename(weights_path)}")
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
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            sla_met_total += len(info['sla_met'])
            sla_checks += len(QIDS)
            
            if step % args.log_every == 0:
                log.info(
                    f"[Eval Step {step}] "
                    f"action={action_to_name(action):6s} "
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
    parser.add_argument('--max-episode-steps', type=int, default=100,
                        help='Max steps per episode (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-every', type=int, default=10,
                        help='Log frequency')
    parser.add_argument('--reset-prob-start', type=float, default=0.9,
                        help='Probability of baseline reset at the beginning '
                             'of training (default: 0.9)')
    parser.add_argument('--reset-prob-end', type=float, default=0.3,
                        help='Probability of baseline reset near the end of '
                             'training (default: 0.3)')
    parser.add_argument('--baseline-cooldown-seconds', type=float, default=5.0,
                        help='Traffic warmup/cooldown before measured steps '
                             'after a baseline reset (default: 5.0)')
    parser.add_argument('--warm-cooldown-seconds', type=float, default=2.0,
                        help='Traffic warmup/cooldown before measured steps '
                             'after a warm-start episode (default: 2.0)')
    
    # Model
    parser.add_argument('--save-dir', default='training_files',
                        help='Directory to save/load model weights')
    parser.add_argument('--training-log-dir', default=None,
                        help='Directory for durable training report artifacts '
                             '(default: <save-dir>/training_logs)')
    parser.add_argument('--training-log-flush-every', type=int, default=25,
                        help='Flush durable local training logs every N steps '
                             '(default: 25; checkpoints/episodes always flush)')
    parser.add_argument('--invalid-recovery-streak', type=int,
                        default=INVALID_RECOVERY_STREAK,
                        help='End the current episode and force a baseline '
                             'reset after N consecutive invalid transitions '
                             '(default: 8; set 0 to disable)')
    parser.add_argument('--telemetry-liveness-retries', type=int,
                        default=TELEMETRY_LIVENESS_RETRIES,
                        help='Fresh local-cache telemetry checks before an '
                             'episode is allowed to start learning '
                             f'(default: {TELEMETRY_LIVENESS_RETRIES})')
    parser.add_argument('--telemetry-liveness-interval', type=float,
                        default=TELEMETRY_LIVENESS_INTERVAL_SECONDS,
                        help='Seconds between episode-start telemetry '
                             'liveness checks '
                             f'(default: {TELEMETRY_LIVENESS_INTERVAL_SECONDS})')
    parser.add_argument('--telemetry-liveness-restarts', type=int,
                        default=TELEMETRY_LIVENESS_RESTARTS,
                        help='Profile restarts to try when telemetry liveness '
                             'fails before falling back to baseline reset '
                             f'(default: {TELEMETRY_LIVENESS_RESTARTS})')
    parser.add_argument('--disable-telemetry-liveness-gate',
                        action='store_true',
                        help='Disable the local-cache telemetry liveness gate')
    parser.add_argument('--training-influx-detail',
                        choices=['minimal', 'off'],
                        default='off',
                        help='Influx training logging detail: minimal writes only '
                             'progress-monitoring fields, off disables training '
                             'Influx writes (default: off)')
    parser.add_argument('--weights-tag', default='final',
                        help='Weight file tag for evaluation (e.g., final, best, 50pct)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint (e.g., 50pct, best, or path to .pth file)')
    parser.add_argument('--resume-eps', type=float, default=None,
                        help='Reset epsilon to this value when resuming (e.g., 0.10)')
    parser.add_argument('--traffic-weights', type=str, default=None,
                        help='Traffic category weights as "light:0.2,medium:0.3,high:0.5"')
    parser.add_argument('--traffic-profile-weights', type=str, default=None,
                        help='Exact traffic profile weights as '
                             '"light_1:1,medium_2:2,bursty_vo_1:3". '
                             'Overrides --traffic-weights when both are set.')
    parser.add_argument('--traffic-profile', type=str,
                        choices=tuple(TrafficManager.TRAFFIC_PROFILES),
                        default=None,
                        help='Use specific traffic profile for all episodes (overrides --traffic-weights)')

    # Learning rate override
    parser.add_argument('--lr', type=float, default=None,
                        help=f'Override learning rate (default: {LR})')

    # EWC (Elastic Weight Consolidation) for multi-topology training
    parser.add_argument('--ewc-lambda', type=float, default=0.0,
                        help=f'EWC regularization strength (0 to disable, recommended: {EWC_LAMBDA})')
    parser.add_argument('--compute-ewc', action='store_true',
                        help='Compute and save Fisher matrix after training (for next topology)')
    parser.add_argument('--ewc-file', type=str, default=None,
                        help='Path to EWC Fisher matrix file from previous topology')

    # Multi-topology replay buffer
    parser.add_argument('--multi-buffer', action='store_true',
                        help='Use separate replay buffers per topology')
    parser.add_argument('--buffer-capacity', type=int, default=25000,
                        help='Replay buffer capacity per topology (default: 25000)')
    parser.add_argument('--balanced-sampling', action='store_true',
                        help='Balance sampling across topology buffers (requires --multi-buffer)')

    # Topology configuration (optional - uses default if not specified)
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to YAML topology configuration file')
    parser.add_argument('--rules-dir', type=str, default=None,
                        help='Directory containing P4 rule files (auto-detected from config if not specified)')

    # InfluxDB
    parser.add_argument('--influx-url', default='http://192.168.56.1:8086')
    parser.add_argument('--influx-org', default='Research')
    parser.add_argument('--influx-bucket', default='INT')
    parser.add_argument('--influx-token', default=os.environ.get('INFLUX_TOKEN'),
                        help='InfluxDB token (or set INFLUX_TOKEN env var)')
    parser.add_argument('--telemetry-backend',
                        choices=['cache', 'influx', 'cache-fallback-influx'],
                        default='cache',
                        help='Telemetry source for RL observations '
                             '(cache uses the local collector socket; influx preserves legacy queries)')
    parser.add_argument('--telemetry-cache-socket',
                        default=DEFAULT_SOCKET_PATH,
                        help='Unix socket path for local telemetry cache')
    parser.add_argument('--telemetry-cache-timeout',
                        type=float,
                        default=1.0,
                        help='Local telemetry cache request timeout in seconds')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='Console log level (file always logs DEBUG)')
    parser.add_argument('--no-warm-start', action='store_true',
                        help='Force baseline resets (no warm-start episodes)')

    args = parser.parse_args()

    # Setup logging based on log level
    setup_logging(log_level=args.log_level)

    needs_influx = (
        args.telemetry_backend in ('influx', 'cache-fallback-influx')
        or (args.mode == 'train' and args.training_influx_detail != 'off')
    )
    if needs_influx and not args.influx_token:
        log.error("InfluxDB token not configured. Set INFLUX_TOKEN environment variable or use --influx-token argument.")
        sys.exit(1)

    try:
        if args.mode == 'train':
            train(args)
        else:
            evaluate(args)
    except KeyboardInterrupt:
        log.info("\nInterrupted by user, shutting down...")
        sys.exit(130)
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
