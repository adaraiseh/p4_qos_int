#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_agent_4.py - Simplified DQN for per-queue QoS path optimization using P4 INT reports

Key Design Principles:
1. Single centralized DQN agent
# 2. Frame stacking (464 features) - stacked observations + stacked one-hot actions
# 3. Action history as one-hot vectors - agent knows "I caused this" vs "happened naturally"
# 4. Clear SLA-based reward - bounded, no improvement bonus (avoids rewarding noise)
# 5. Focused action space (8 actions) - no-op + 6 single + 1 multi-queue
# 6. Prioritized Experience Replay - learn from rare important events
# 7. Proper episode boundaries - clear termination conditions
# 8. Tuned timing for 100% post-action data capture
# 9. Queue-specific bottleneck detection and alternative metrics
# 10. EMA temporal smoothing for latency trends

# State Composition (464 features):
# - Stacked Observations: 50 metrics * 8 frames = 400 features
# - Stacked Actions (one-hot): 8 * 8 frames = 64 features
# - Total: 464 features

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
import csv
import math
import glob
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Tuple, Optional
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

# Create a custom handler with immediate flushing
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Define setup_logging function to be called by main
def setup_logging(verbose: bool = False):
    """Configure logging with immediate flushing and appropriate level."""
    # Determine level
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[FlushingStreamHandler(sys.stdout)],
        force=True,
    )
    
    # Set level for this module matches root
    log = logging.getLogger(__name__)
    log.setLevel(level)
    
    # Ensure traffic_generator logger matches
    logging.getLogger('traffic_generator').setLevel(level)
    
    # If verbose, set controller logger to DEBUG restricted (or handle elsewhere)
    # The Controller class handles its own verbosity, but we can set the logger level here too
    
log = logging.getLogger(__name__)
# Default to INFO until setup_logging is called
log.setLevel(logging.INFO)

# Now import modules that use logging
from controller import Controller
from traffic_generator import TrafficManager
from config.schema import MAX_SWITCHES

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Ensure traffic_generator logger uses the same level
# Ensure traffic_generator logger uses the same level
# logging.getLogger('traffic_generator').setLevel(logging.INFO)


# =============================================================================
#                           HYPERPARAMETERS
# =============================================================================
# Network
HIDDEN_DIM = 128        # Reduced from 256 for smaller state space
RAW_STATE_DIM = 50      # 3×16 + 2: per-queue features + max_pressure + steps_since_action
STACK_SIZE = 16         # Extended for burst detection (~12.8s history)
ACTION_DIM = 8          # No-op + 6 single (3 queues × 2 alts) + 1 multi
# State composition: stacked observations + stacked one-hot actions
# Observations: 50 metrics * 16 frames = 800
# Actions: 8 (one-hot) * 16 frames = 128
# Total: 800 + 128 = 928
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
EPS_DECAY_STEPS = 30_000  # Decay over 60% of 50K training for thorough exploration

# Target network - Soft updates (Polyak averaging) for smooth Q-value evolution
TAU = 0.005  # Soft update rate: target = TAU * online + (1-TAU) * target

# EWC (Elastic Weight Consolidation) - prevents catastrophic forgetting
EWC_LAMBDA = 5000.0       # Regularization strength (tune: 1000-10000)
EWC_FISHER_SAMPLES = 200  # Samples for Fisher matrix estimation

# Environment timing - tuned for faster training with acceptable data capture
# Based on sync test results: first_change ~0.28s, query RTT ~0.3s
# Formula: DELAY_AFTER_ACTION >= WINDOW + SAFETY_LAG/1000 + first_change + margin
WINDOW_SECONDS = 1.0        # Observation window (reduced from 1.5)
SAFETY_LAG_MS = 100         # 100ms safety lag for InfluxDB
COOLDOWN_SECONDS = 0.0      # No cooldown for faster learning
DELAY_AFTER_ACTION = 0.8    # Wait after action (reduced from 1.5s - data available after ~500ms)
DELAY_NO_ACTION = 0.8       # Match action delay

# Freshness validation - minimum data points required per metric in window
MIN_POINTS_PER_METRIC = 1   # Require at least 1 point (2 is too strict)

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
LAT_RATIO_CAP = 3.0  # Cap latency ratio at 3x SLA (prevents instability)
LAT_DIFF_CAP = 2.0   # Cap lat difference features to [-2, +2]

# Reward weights - TUNED to prevent tanh saturation
REWARD_SLA_MET_SCALE = 0.5         # Reduced from 1.0
REWARD_SLA_VIOLATED_SCALE = 0.1     # Reduced from 0.8
REWARD_DROP_PENALTY = 0.8           # Reduced from 1.5 (with sqrt compression)

# Queue-specific action cost (replaces pressure-based approach)
# Cost depends on whether the TARGETED queue's SLA is met:
#   - If targeting a healthy queue → high cost (risky, don't break what works)
#   - If targeting a sick queue → low cost (encouraged to fix)
REWARD_ACTION_COST_HEALTHY = 0.5   # Cost when targeting a queue with SLA met (reduced from 0.65)
REWARD_ACTION_COST_SICK = 0.10      # Cost when targeting a queue with SLA violated

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
        # max_priority stores RAW priority (|TD| + eps), NOT exponentiated
        # This avoids double-alpha bug when push() applies alpha
        self.max_priority = 1.0
        self.epsilon = 1e-5
    
    def push(self, state, action, reward, next_state, terminated):
        """Add experience with max priority (ensures new samples get sampled)."""
        data = (state, action, reward, next_state, terminated)
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
        
        states, actions, rewards, next_states, terminateds = zip(*samples)
        
        return (
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            terminateds,
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

    def push(self, state, action, reward, next_state, terminated):
        """Add experience to current topology's buffer."""
        if self.current_topology is None:
            raise ValueError("Must call set_topology() before push()")
        self.buffers[self.current_topology].push(state, action, reward, next_state, terminated)

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
        all_next_states, all_terminateds = [], []
        all_indices, all_weights = [], []

        for i, (topo_name, buffer) in enumerate(self.buffers.items()):
            if len(buffer) < samples_per_topo:
                # Not enough samples in this buffer, skip
                continue

            n_samples = samples_per_topo + (1 if i < remainder else 0)
            (states, actions, rewards, next_states, terminateds,
             indices, weights) = buffer.sample(n_samples, beta)

            all_states.append(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_next_states.append(next_states)
            all_terminateds.extend(terminateds)
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
    
    def push_experience(self, state, action, reward, next_state, terminated):
        """Add experience to replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, terminated)
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
             indices, weights) = self.replay_buffer.sample(
                 BATCH_SIZE, self.beta, balance=self.balanced_sampling)
        else:
            (states, actions, rewards, next_states, terminateds,
             indices, weights) = self.replay_buffer.sample(BATCH_SIZE, self.beta)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        terminated_t = torch.BoolTensor(terminateds).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: use online net to select actions, target net to evaluate
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1)
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
        """Save model checkpoint including replay buffer."""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'eps_step_count': self.eps_step_count,
            'eps': self.eps,
            'beta': self.beta,
            # Replay buffer state for resume continuity
            'replay_tree': self.replay_buffer.tree.tree.copy(),
            'replay_data': self.replay_buffer.tree.data.copy(),
            'replay_write': self.replay_buffer.tree.write,
            'replay_n_entries': self.replay_buffer.tree.n_entries,
            'replay_max_priority': self.replay_buffer.max_priority,
        }, path)
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
        if 'replay_tree' in checkpoint:
            self.replay_buffer.tree.tree = checkpoint['replay_tree']
            self.replay_buffer.tree.data = checkpoint['replay_data']
            self.replay_buffer.tree.write = checkpoint['replay_write']
            self.replay_buffer.tree.n_entries = checkpoint['replay_n_entries']
            self.replay_buffer.max_priority = checkpoint['replay_max_priority']
            log.info(f"Loaded replay buffer with {len(self.replay_buffer)} experiences")
        else:
            log.info("No replay buffer in checkpoint, starting fresh")
        
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
            'q_max': np.mean(self.q_values_max) if self.q_values_max else 0.0,
            'q_mean': np.mean(self.q_values_mean) if self.q_values_mean else 0.0,
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
                 rules_dir: str = None):
        self.bucket = bucket
        self.org = org
        self.url = url
        
        # InfluxDB client
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=5000)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

        # Store topology builder for reference
        self._topology_builder = topology_builder

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
        
        # Cache snapshots for comparison
        self.last_snapshot = None
        
        # Frame stacking for velocity/trend detection
        # Stores last STACK_SIZE raw observation states (each 50-dim)
        self.frame_stack: deque = deque(maxlen=STACK_SIZE)
        
        # Action stacking for causality tracking
        # Stores last STACK_SIZE actions as one-hot vectors (each 8-dim)
        self.action_stack: deque = deque(maxlen=STACK_SIZE)
        
        # EMA smoothed latency ratios for temporal smoothing
        self.lat_ema = {qid: 1.0 for qid in QIDS}
        
        # Traffic manager for dynamic profile changes (training only)
        # In production_mode, traffic is managed externally by ProductionRunner
        self.traffic_manager = TrafficManager() if (reset_network and not production_mode) else None
        
        # Current traffic profile for logging
        self.current_traffic_profile = ""  # e.g., "light_1", "medium_2", "bursty_be_1"
        self.current_traffic_category = ""  # "light", "medium", "high", "bursty"
        self.traffic_category_weights = None  # Optional: {'light': 0.1, 'medium': 0.2, 'high': 0.3, 'bursty': 0.4}
        self.fixed_traffic_profile = None    # Optional: override to use specific profile for all episodes
        self.is_bursty_episode = False  # Track if current episode uses bursty profile
    
    def reset(self, force_reset: Optional[bool] = None) -> np.ndarray:
        """Reset episode and return initial stacked state.
        
        Args:
            force_reset: Override reset behavior for this episode.
                - None: use self.reset_network default
                - True: force baseline reset (clear tables, reprogram OSPF)
                - False: warm-start from current state
        
        Curriculum-based training can use this to mix baseline-starts and warm-starts:
        - Early training: mostly baseline-starts (agent learns from clean state)
        - Late training: mostly warm-starts (agent learns stability/recovery)
        """
        # Determine whether to reset this episode
        do_reset = force_reset if force_reset is not None else self.reset_network
        
        # Perform full network reset if requested
        if do_reset:
            log.info("=== BASELINE START: Resetting network to OSPF ===")
            
            # Step 1: Clear all P4 tables
            self.controller.clear_all_tables()
            log.info("Cleared all P4 tables")
            
            # Step 2: Recompute baseline OSPF paths
            self.controller.compute_forwarding_entries()
            log.info("Recomputed OSPF forwarding entries")
            
            # Step 3: Program switches with baseline
            self.controller.program_switches()
            log.info("Programmed switches with baseline routing")
        else:
            log.info("=== WARM START: Continuing from current routing state ===")
        
        # Start traffic for new episode (training mode only)
        if self.traffic_manager:
            if self.fixed_traffic_profile:
                log.info(f"Starting FIXED traffic profile: {self.fixed_traffic_profile}")
                profile_info = self.traffic_manager.start_traffic(profile_name=self.fixed_traffic_profile)
            else:
                log.info("Starting randomized traffic profile...")
                profile_info = self.traffic_manager.start_traffic(category_weights=self.traffic_category_weights)
            
            self.current_traffic_profile = profile_info['profile_name']
            self.current_traffic_category = profile_info['profile_category']
            self.is_bursty_episode = profile_info.get('is_bursty', False)
            log.info(f"Traffic started: {self.current_traffic_profile} ({self.current_traffic_category})")
            if self.is_bursty_episode:
                log.info(f"  [BURSTY] Will cycle bursts during episode")
        
        # Cool-down period for traffic to stabilize (traffic is now running)
        cooldown = 5.0 if do_reset else 2.0  # Less cooldown for warm-start
        log.info(f"Traffic running, waiting {cooldown}s for metrics to stabilize...")
        time.sleep(cooldown)
        log.info("Episode start complete")
        
        # Reset episode counters
        self.episode_step = 0
        self.sla_streak = 0
        self.last_action = 0
        self.last_action_time = time.monotonic()
        self._last_action_step = 0  # For steps_since_action feature
        
        # Reset EMA state
        self.lat_ema = {qid: 1.0 for qid in QIDS}
        
        # Collect initial snapshot (after reset if applicable)
        self.last_snapshot = self._collect_snapshot()
        raw_state = self._build_raw_state(self.last_snapshot)
        
        # Debug assertion to catch dimension mismatches early
        assert len(raw_state) == RAW_STATE_DIM, f"State dim mismatch: {len(raw_state)} != {RAW_STATE_DIM}"
        
        # Initialize frame stack with replicated first observation
        # This provides a clean slate for each episode
        self.frame_stack.clear()
        for _ in range(STACK_SIZE):
            self.frame_stack.append(raw_state.copy())
        
        # Initialize action stack with no-ops
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
        
        # Track which metrics were received per queue
        metrics_received = {qid: {'lat': False, 'drop': False, 'util': False} for qid in QIDS}
        
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
                            # Track which metrics we received (not defaults)
                            if measurement == 'lat_p95':
                                metrics_received[qid]['lat'] = True
                            elif measurement == 'drop_p95':
                                metrics_received[qid]['drop'] = True
                            elif measurement == 'util_p95':
                                metrics_received[qid]['util'] = True
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            log.warning(f"Failed to query metrics: {e}")
        
        # Batch Query: Freshness (One query for all queues)
        freshness_map = self._check_all_queues_freshness()
        
        # Mark data_valid only if ALL 3 metrics were returned, values are sane, AND data is fresh
        for qid in QIDS:
            m = metrics_received[qid]
            metrics_present = m['lat'] and m['drop'] and m['util']
            values_sane = self._metric_sane(snapshot[qid]) if metrics_present else False
            # Check freshness from batch result
            data_fresh = freshness_map.get(qid, False)
            snapshot[qid]['data_valid'] = metrics_present and values_sane and data_fresh
            snapshot[qid]['data_fresh'] = data_fresh  # For debugging
        
        # Log telemetry status for monitoring
        valid_count = sum(1 for qid in QIDS if snapshot[qid]['data_valid'])
        if valid_count < len(QIDS):
            missing = [qid for qid in QIDS if not snapshot[qid]['data_valid']]
            reasons = []
            for qid in missing:
                if not metrics_received[qid]['lat'] or not metrics_received[qid]['drop'] or not metrics_received[qid]['util']:
                    reasons.append(f"q{qid}:missing_metrics")
                elif not self._metric_sane(snapshot[qid]):
                    reasons.append(f"q{qid}:insane_values")
                elif not snapshot[qid].get('data_fresh', True):
                    reasons.append(f"q{qid}:stale_data")
            log.warning(f"[Telemetry] Invalid data for queues {missing} ({', '.join(reasons)}) - only {valid_count}/{len(QIDS)} valid")
        
        # 2. Get Path and Bottleneck Info (Queue-Specific)
        # Each queue gets its own bottleneck detection and alternative metrics
        # This ensures accurate per-queue congestion identification
        
        # Batch Query: Hottest Demands (One query for all queues)
        all_hot_demands = self._get_all_hottest_demands()
        
        for qid in QIDS:
            # Find hottest demand for this queue from batch result
            hot = all_hot_demands.get(qid)
            if not hot:
                log.info(f"[Snapshot] Queue {qid}: No hot demand found, skipping bottleneck detection")
                continue
            
            src_ip, dst_ip = hot
            log.debug(f"[Snapshot] Queue {qid}: hot_demand=({src_ip}, {dst_ip})")
            snapshot[qid]['hot_src_ip'] = src_ip
            snapshot[qid]['hot_dst_ip'] = dst_ip
            
            # Get current path
            path = self.controller.get_path_by_ips(src_ip, dst_ip)
            if not path:
                log.info(f"[Snapshot] Queue {qid}: No path found for ({src_ip}, {dst_ip})")
                continue
            
            snapshot[qid]['path_nodes'] = list(path)
            
            # --- Step A: Path Metrics (Queue-Specific) ---
            # Query metrics filtered by this queue_id for accurate bottleneck detection
            # Filter to switches only (exclude hosts) - check against known switch names
            sw_names = [n for n in path if n in self.controller.switch_name_to_id]
            sw_ids = [self.controller.switch_name_to_id[n] for n in sw_names]
            sw_ids = [int(s) for s in sw_ids]
            
            if not sw_ids:
                continue
                
            # Query path switches with queue_id filter for accurate per-queue bottleneck
            path_metrics = self._query_switch_metrics_for_queue(sw_ids, qid)
            
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
                    alt_metrics = self._query_switch_metrics_for_queue(alt_sids, qid)

                    # Calculate bottleneck score for relative comparison
                    bn_drop_norm = min(bm['drop'], DROP_CAP) / DROP_CAP
                    bn_lat_norm = min(bm['lat'], SLA_THRESHOLDS[qid]) / SLA_THRESHOLDS[qid]
                    bn_score = 0.6 * bn_drop_norm + 0.4 * bn_lat_norm

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

    def _query_switch_metrics_for_queue(self, sw_ids: List[int], qid: int) -> Dict[int, Dict]:
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
        start, stop = self._time_window()
        
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
        try:
            tables = self.query_api.query(org=self.org, query=flux)
            for table in tables or []:
                for record in table.records:
                    sid = int(record.values.get('switch_id', 0) or 0)
                    results[sid] = {
                        'drop': float(record.values.get('q_drop_rate_100ms', 0) or 0),
                        'lat': float(record.values.get('switch_latency', 0) or 0),
                        'util': float(record.values.get('tx_utilization', 0) or 0),
                    }
        except Exception as e:
            log.debug(f"Failed to query switch metrics for queue {qid}: {e}")
            
        return results
    
    def _check_all_queues_freshness(self) -> Dict[int, bool]:
        """
        Check freshness for ALL queues in a single query.
        Returns: Dict[qid, bool]
        """
        start, stop = self._time_window()
        
        # Group by both queue_id and _measurement
        flux = f'''
        from(bucket:"{self.bucket}")
            |> range(start:{start}, stop:{stop})
            |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
            |> filter(fn: (r) => r._measurement == "flow_latency" or r._measurement == "q_drop_rate_100ms" or r._measurement == "tx_utilization")
            |> group(columns:["queue_id", "_measurement"])
            |> count()
        '''
        
        # Initialize counts: qid -> measurement -> count
        counts = {qid: {'flow_latency': 0, 'q_drop_rate_100ms': 0, 'tx_utilization': 0} for qid in QIDS}
        freshness_map = {qid: False for qid in QIDS}
        
        try:
            tables = self.query_api.query(org=self.org, query=flux)
            for table in tables or []:
                for record in table.records:
                    try:
                        qid = int(record.values.get('queue_id', -1))
                        measurement = record.values.get('_measurement')
                        count = record.get_value()
                        
                        if qid in counts and measurement in counts[qid] and count is not None:
                            counts[qid][measurement] = int(count)
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            log.debug(f"Failed to check batch data freshness: {e}")
            return freshness_map  # All False
            
        # Check freshness per queue
        for qid in QIDS:
            c = counts[qid]
            fresh = all(cnt >= MIN_POINTS_PER_METRIC for cnt in c.values())
            freshness_map[qid] = fresh
            if not fresh:
                log.debug(f"[Freshness] Queue {qid} has insufficient points: {c}")
                
        return freshness_map
    
    def _get_all_hottest_demands(self) -> Dict[int, Tuple[str, str]]:
        """Get the demand with highest latency for ALL queues in one query."""
        start, stop = self._time_window()
        flux = f'''
        from(bucket:"{self.bucket}")
            |> range(start:{start}, stop:{stop})
            |> filter(fn: (r) => r._measurement == "flow_latency")
            |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
            |> toFloat()
            |> group(columns:["queue_id", "src_ip", "dst_ip"])
            |> mean(column:"_value")
            |> group(columns:["queue_id"])
            |> sort(columns:["_value"], desc:true)
            |> limit(n:1)
        '''
        results = {}
        try:
            tables = self.query_api.query(org=self.org, query=flux)
            for table in tables or []:
                for record in table.records:
                    try:
                        qid = int(record.values.get('queue_id', -1))
                        src = record.values.get('src_ip')
                        dst = record.values.get('dst_ip')
                        if qid in QIDS and src and dst:
                            results[qid] = (str(src), str(dst))
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            log.debug(f"Failed to get batch hottest demands: {e}")
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
        Build 50-dimensional raw observation vector with relative metrics encoding.
        (Actions are stacked separately as one-hot vectors)
        
        Layout per queue (16 features):
          [0-5]: Basic metrics (lat_ratio, drop_norm, util_norm, sla_met, lat_ema, lat_ema_diff)
          [6-9]: Bottleneck info (present, drop, lat, util)
          [10-15]: Alternatives 2 × 3 = 6 (available, drop_vs_bn, lat_vs_bn)
        
        Global (2):
          - Max pressure
          - Steps since last action (normalized, helps avoid rapid oscillation)
        
        Total: 3×16 + 2 = 50 features
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
        steps_since = self.episode_step - getattr(self, '_last_action_step', 0)
        state[idx] = min(steps_since / 10.0, 1.0)  # Cap at 10 steps
        
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
            464-dimensional state vector:
              - [0-399]: Stacked observations (50 * 8 = 400 features)
              - [400-463]: Stacked one-hot actions (8 * 8 = 64 features)
            
        Layout: [obs_t-7, ..., obs_t-1, obs_t, 
                 act_t-7, ..., act_t-1, act_t]
            
        This allows the agent to see:
        - Trends: If latency is rising or falling (from observation history)
        - Causality: Full action history as one-hot vectors
        - Example: If action_stack = [[1,0,...], [0,0,1,0,...], ...]
                   means: noop -> video-alt-0 -> ...
        """
        # Concatenate observations: oldest first, newest last
        stacked_obs = np.concatenate(list(self.frame_stack), axis=0)
        
        # Concatenate one-hot actions: oldest first, newest last
        stacked_actions = np.concatenate(list(self.action_stack), axis=0)
        
        # Combine: [400 obs features] + [64 action features] = 464 total
        return np.concatenate([stacked_obs, stacked_actions], axis=0)
    
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
            for action, mapping in self.ACTION_MAP.items():
                if mapping is None:  # Skip no-op
                    continue
                if mapping == 'multi':  # Handle multi-action separately
                    continue
                    
                qid, alt_idx = mapping
                
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
            for qid in QIDS:
                q = snapshot[qid]
                # Check soft margin violation (consistent with step logic)
                is_violating = (q['lat_p95'] / SLA_THRESHOLDS[qid]) > (1.0 + SLA_SOFT_MARGIN)
                if is_violating and q.get('bottleneck_sid') is not None and len(q.get('alternatives', [])) > 0:
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
        
        for qid in QIDS:
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
        
        if rerouted:
            return True, f"multi:{len(rerouted)}", len(rerouted)
        else:
            return False, None, 0
    
    def _apply_action(self, action: int, snapshot: Dict[int, Dict]) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Apply routing action based on explicit alternative selection.
        
        Returns:
            (success, alt_name, alt_idx) - alt_name for logging, alt_idx for InfluxDB
            For multi-action: alt_name is 'multi:N', alt_idx is count of rerouted queues
        """
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
        
        # Handle step-based bursts for bursty episodes
        if self.is_bursty_episode and self.traffic_manager:
            burst_msg = self.traffic_manager.check_step_burst(
                self.episode_step, 
                bursty_profile=self.current_traffic_profile
            )
            # Burst state changes are logged inside check_step_burst
        
        # Get current snapshot
        current_snapshot = self.last_snapshot
        
        # Apply action
        action_applied, alt_name, alt_idx = self._apply_action(action, current_snapshot)
        
        # Update last action tracking (only on applied actions for stability feature)
        if action_applied:
            self.last_action_time = time.monotonic()
            self._last_action_step = self.episode_step  # For steps_since_action feature
        self.last_action = action
        
        # Wait for network to settle
        delay = DELAY_AFTER_ACTION if action_applied else DELAY_NO_ACTION
        time.sleep(delay)
        
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
                
                # Check if targeted queue's SLA was met BEFORE the action
                # We must look at current_snapshot (pre-action), not info/next_snapshot
                q_pre = current_snapshot[targeted_qid]
                sla_pre = SLA_THRESHOLDS[targeted_qid]
                
                # Determine health using the same logic as _compute_reward margin
                # If pre-action latency was within "safe zone" (ratio <= 1.2), it was "Healthy"
                ratio_pre = q_pre['lat_p95'] / sla_pre
                margin_high = 1.0 + SLA_SOFT_MARGIN  # 1.2
                
                if ratio_pre <= margin_high:
                    # Targeting a healthy queue → high cost (risky)
                    action_cost = REWARD_ACTION_COST_HEALTHY
                else:
                    # Targeting a sick queue → low cost (encouraged)
                    action_cost = REWARD_ACTION_COST_SICK
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
        all_sla_met = len(info['sla_met']) == len(QIDS)
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
        
        # Action context check - for ANY non-zero action (not just applied ones)
        action_context_valid = True
        if action != 0:
            # Get the targeted queue from action mapping
            mapping = self.ACTION_MAP.get(action)
            if mapping == 'multi':
                # Multi-action: require at least one violating queue to have context
                # Use soft margin (> 1.2) for consistency with reward/cost logic
                action_context_valid = any(
                    (current_snapshot[qid].get('hot_src_ip') is not None and
                     current_snapshot[qid].get('hot_dst_ip') is not None and
                     current_snapshot[qid].get('bottleneck_sid') is not None)
                    for qid in QIDS
                    if (current_snapshot[qid]['lat_p95'] / SLA_THRESHOLDS[qid]) > (1.0 + SLA_SOFT_MARGIN)
                )
                if not action_context_valid:
                    log.debug(f"[Step {self.episode_step}] Multi-action had no violating queues with context")
            elif mapping is not None:
                # Single-queue action: extract targeted queue ID
                targeted_qid = mapping[0]
                # Check if the PRE-ACTION snapshot had valid routing context
                q_pre = current_snapshot[targeted_qid]
                has_context = (q_pre.get('hot_src_ip') is not None and 
                               q_pre.get('hot_dst_ip') is not None and 
                               q_pre.get('bottleneck_sid') is not None)
                if not has_context:
                    action_context_valid = False
                    log.debug(f"[Step {self.episode_step}] Action {action} targeted queue {targeted_qid} lacked routing context")
        
        # Determine validity based on action type (key off action != 0, not action_applied)
        if action == 0:
            # Noop: relaxed - at least 2/3 queues valid
            data_valid = valid_count >= 2
        else:
            # Non-zero action attempted:
            # Strict: all 3 queues valid + action context + action must have succeeded
            # Failed actions (attempted but controller rejected) should NOT be stored
            # as they create confusing transitions ("I tried X but nothing changed")
            data_valid = (valid_count == len(QIDS)) and action_context_valid and action_applied
        
        info['data_valid'] = data_valid
        info['valid_count'] = valid_count
        info['invalid_queues'] = invalid_qs
        
        if not data_valid:
            reasons = []
            if valid_count < len(QIDS):
                reasons.append(f"telemetry incomplete ({valid_count}/3): queues {invalid_qs}")
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
        self.last_snapshot = next_snapshot
        
        return next_state, reward, terminated, truncated, info
    
    def get_valid_actions(self) -> np.ndarray:
        """Get valid action mask for current state."""
        return self._get_valid_actions(self.last_snapshot)
    
    def write_training_metrics(self, step: int, agent_stats: Dict, 
                                reward: float, action: int, prev_action: int, info: Dict,
                                episode: int = 0):
        """
        Write training metrics to InfluxDB for Grafana monitoring.
        Measurement: rl_training
        """
        try:
            p = (
                Point("rl_training")
                .field("step", int(step))
                .field("episode", int(episode))
                .field("episode_step", int(info.get('episode_step', 0)))
                .field("action", int(action))
                .field("reward", float(reward))
                .field("eps", float(agent_stats['eps']))
                .field("beta", float(agent_stats.get('beta', 0.4)))
                .field("loss", float(agent_stats['avg_loss']))  # Renamed from avg_loss
                .field("step_avg_reward_100", float(agent_stats['avg_reward']))  # Renamed
                .field("buffer_size", int(agent_stats['buffer_size']))
                .field("sla_met_count", len(info.get('sla_met', [])))
                .field("sla_streak", int(info.get('sla_streak', 0)))
                .field("pressure", float(info.get('pressure', 0.0)))
                .time(datetime.utcnow())
            )
            
            # Q-value stats from agent
            if 'q_max' in agent_stats:
                p = p.field("q_max", float(agent_stats['q_max']))
            if 'q_mean' in agent_stats:
                p = p.field("q_mean", float(agent_stats['q_mean']))
            
            # Optional fields (useful only)
            if info.get('alt_idx') is not None:
                p = p.field("alt_idx", int(info['alt_idx']))
            
            # Per-queue latencies only (removed redundant sla_met flags)
            per_queue = info.get('per_queue', {})
            for qid in QIDS:
                if qid in per_queue:
                    q_lat = per_queue[qid].get('lat', 0.0)
                    p = p.field(f"queue_{qid}_latency", float(q_lat))
            
            # Data validity metrics (keep for debugging)
            p = p.field("data_valid", int(info.get('data_valid', False)))
            
            # Traffic profile as FIELDS
            if info.get('traffic_profile'):
                p = p.field("traffic_profile", str(info['traffic_profile']))
            if info.get('traffic_category'):
                p = p.field("traffic_category", str(info['traffic_category']))
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=[p])
        except Exception as e:
            log.debug(f"Failed to write training metrics: {e}")
    
    def write_episode_metrics(self, episode: int, episode_steps: int, episode_reward: float,
                               rolling_avg_100: float, traffic_profile: str, traffic_category: str):
        """Write episode summary metrics to InfluxDB.
        Measurement: rl_training (same as steps for unified visualization)
        """
        try:
            p = (
                Point("rl_training")
                .field("episode", int(episode))
                .field("episode_steps", int(episode_steps))
                .field("episode_reward", float(episode_reward))
                .field("reward_rolling_avg", float(rolling_avg_100))  # User requested renaming to keep this clear
                .field("traffic_profile", str(traffic_profile))    # FIELD not tag
                .field("traffic_category", str(traffic_category))  # FIELD not tag
                .time(datetime.utcnow())
            )
            self.write_api.write(bucket=self.bucket, org=self.org, record=[p])
        except Exception as e:
            log.debug(f"Failed to write episode metrics: {e}")
    
    def close(self):
        """Clean up resources."""
        try:
            if self.traffic_manager:
                self.traffic_manager.stop_all()
                log.info("Traffic stopped")
            self.write_api.close()
            self.client.close()
        except Exception:
            pass


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
    topology_builder = None
    rules_dir = args.rules_dir
    if args.config:
        from topology.factory import create_topology
        log.info(f"Loading topology from: {args.config}")
        topology_builder = create_topology(args.config)
        if rules_dir is None:
            # Auto-detect rules dir from topology name
            topo_name = topology_builder.config.topology.name.replace('-', '_')
            rules_dir = f"rules/{topo_name}"
        log.info(f"Topology: {topology_builder.config.topology.name}")
        log.info(f"  Switches: {len(topology_builder.switches)}/{MAX_SWITCHES} max")
        log.info(f"  Rules dir: {rules_dir}")

    # Initialize environment and agent
    env = QoSRoutingEnv(
        args.influx_bucket, args.influx_token,
        args.influx_org, args.influx_url,
        topology_builder=topology_builder,
        rules_dir=rules_dir
    )
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
        resume_path = args.resume
        if not resume_path.endswith('.pth'):
            # Try glob pattern for timestamped checkpoints first
            # Pattern: YYYYMMDD-HHMMSS-dqn_v4_{tag}.pth
            pattern = os.path.join(args.save_dir, f"*-dqn_v4_{args.resume}.pth")
            matching_files = sorted(glob.glob(pattern), reverse=True)

            if matching_files:
                # Use the latest (most recent) checkpoint
                resume_path = matching_files[0]
                log.info(f"Found {len(matching_files)} matching checkpoints, using latest: {os.path.basename(resume_path)}")
            else:
                # Fallback to legacy naming without timestamp
                resume_path = os.path.join(args.save_dir, f"dqn_v4_{args.resume}.pth")

        if os.path.exists(resume_path):
            agent.load(resume_path)
            log.info(f"Resumed training from {resume_path}")

            # Override epsilon if specified
            if args.resume_eps is not None:
                agent.eps = args.resume_eps
                agent.eps_step_count = 0  # Reset ONLY epsilon decay steps
                log.info(f"Reset epsilon to {agent.eps} for resume training (keeping global step count {agent.step_count})")
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
    
    # Parse fixed traffic profile if specified (overrides weights)
    if args.traffic_profile:
        env.fixed_traffic_profile = args.traffic_profile
        log.info(f"Using FIXED traffic profile: {args.traffic_profile}")
    
    # Training state
    total_steps = 0
    episode = 0
    best_avg_reward = -float('inf')
    episode_rewards = []  # Track episode rewards for rolling average
    
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
        'sla_met', 'sla_streak', 'eps', 'loss', 'avg_reward_100', 'alt_used',
        'traffic_profile', 'traffic_category', 'timestamp'
    ])
    log.info(f"Training log: {csv_path}")
    
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
                reset_prob = 0.9 - 0.6 * progress  # 0.9 → 0.3 over training
                do_reset = random.random() < reset_prob
            
            state = env.reset(force_reset=do_reset)
            episode_reward = 0.0
            episode_steps = 0
            done = False
            prev_action = 0  # Track previous action for churn metrics
            
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
                    agent.push_experience(state, action, reward, next_state, terminated)
                    agent.update_epsilon()
                else:
                    log.debug(f"[Step {total_steps}] Skipping experience storage - invalid telemetry")
                
                # Always try to train from existing valid experiences in the buffer
                loss = agent.train_step()
                
                episode_reward += reward
                state = next_state
                
                # Logging
                stats = agent.get_stats()
                
                if total_steps % args.log_every == 0:
                    loss_str = f"{stats['last_loss']:.4f}" if stats['last_loss'] is not None else "N/A"
                    # Map action to readable name
                    if action == 0:
                        action_name = "noop"
                    elif action in (1, 2):
                        action_name = f"v0-alt{action-1}"
                    elif action in (3, 4):
                        action_name = f"v1-alt{action-3}"
                    elif action in (5, 6):
                        action_name = f"be-alt{action-5}"
                    elif action == 7:
                        action_name = "multi"
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
                
                # Write metrics to InfluxDB (includes action churn tracking)
                # Inject traffic profile into info dict for logging
                info['traffic_profile'] = env.current_traffic_profile
                info['traffic_category'] = env.current_traffic_category
                env.write_training_metrics(total_steps, stats, reward, action, prev_action, info)
                prev_action = action  # Update for next step's churn calculation
                
                # Write to local CSV
                csv_writer.writerow([
                    total_steps, episode, action, reward,
                    info.get('raw_reward', 0),
                    len(info['sla_met']), info['sla_streak'],
                    stats['eps'], stats['avg_loss'], stats['avg_reward'],
                    info.get('alt_used', ''),
                    env.current_traffic_profile, env.current_traffic_category,
                    datetime.now().isoformat()
                ])
                csv_file.flush()  # Ensure data is written immediately
                
                # Save checkpoints
                if total_steps in checkpoint_steps:
                    tag = checkpoint_steps[total_steps]
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = os.path.join(args.save_dir, f"{timestamp}-dqn_v4_{tag}.pth")
                    agent.save(path)
                    log.info(f"Checkpoint saved: {path}")
            
            # Episode summary - compute episode reward (avg reward per step)
            ep_reward = episode_reward / episode_steps if episode_steps > 0 else 0.0
            episode_rewards.append(ep_reward)
            
            # Rolling 100-episode average
            rolling_100 = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
            
            log.info(
                f"Episode {episode} finished: "
                f"steps={episode_steps}, "
                f"episode_reward={ep_reward:.3f}, "
                f"rolling_avg_100={rolling_100:.3f}"
            )
            
            # Write episode metrics to InfluxDB
            env.write_episode_metrics(
                episode, episode_steps, ep_reward,
                rolling_100,
                env.current_traffic_profile, env.current_traffic_category
            )
            
            # Track best model (based on rolling 100-episode average)
            if rolling_100 > best_avg_reward and len(episode_rewards) >= 50:
                best_avg_reward = rolling_100
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = os.path.join(args.save_dir, f"{timestamp}-dqn_v4_best.pth")
                agent.save(path)
                log.info(f"New best model saved: rolling_avg_100={best_avg_reward:.3f}")
    
    except KeyboardInterrupt:
        log.info("\nTraining interrupted by user")
        interrupted = True
    
    finally:
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(args.save_dir, f"{timestamp}-dqn_v4_final.pth")
        agent.save(path)

        # Compute and save EWC Fisher matrix if requested (for next topology)
        if args.compute_ewc and not interrupted:
            ewc_path = os.path.join(args.save_dir, f"{timestamp}-ewc.pth")
            log.info("Computing EWC Fisher matrix for next topology...")
            agent.save_ewc(ewc_path, env, EWC_FISHER_SAMPLES)

        env.close()
        csv_file.close()

    log.info("\nTraining complete!")
    log.info(f"Final stats: {agent.get_stats()}")
    log.info(f"Training log saved to: {csv_path}")
    
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
    topology_builder = None
    rules_dir = args.rules_dir
    if args.config:
        from topology.factory import create_topology
        log.info(f"Loading topology from: {args.config}")
        topology_builder = create_topology(args.config)
        if rules_dir is None:
            topo_name = topology_builder.config.topology.name.replace('-', '_')
            rules_dir = f"rules/{topo_name}"
        log.info(f"Topology: {topology_builder.config.topology.name}")
        log.info(f"  Switches: {len(topology_builder.switches)}/{MAX_SWITCHES} max")

    # Initialize environment and agent
    env = QoSRoutingEnv(
        args.influx_bucket, args.influx_token,
        args.influx_org, args.influx_url,
        verbose=args.verbose,
        topology_builder=topology_builder,
        rules_dir=rules_dir
    )
    agent = DQNAgent(STATE_DIM, ACTION_DIM, device)

    # Load weights - find latest checkpoint with datetime prefix
    # Pattern: YYYYMMDD-HHMMSS-dqn_v4_{tag}.pth or legacy dqn_v4_{tag}.pth
    pattern = os.path.join(args.save_dir, f"*-dqn_v4_{args.weights_tag}.pth")
    matching_files = sorted(glob.glob(pattern), reverse=True)

    if matching_files:
        # Use the latest (most recent) checkpoint
        weights_path = matching_files[0]
        log.info(f"Using latest checkpoint: {os.path.basename(weights_path)}")
    else:
        # Fallback to legacy naming without timestamp
        weights_path = os.path.join(args.save_dir, f"dqn_v4_{args.weights_tag}.pth")
        if not os.path.exists(weights_path):
            log.error(f"Weights file not found: {weights_path}")
            log.error(f"Pattern searched: {pattern}")
            return
        log.info(f"Using legacy checkpoint: {os.path.basename(weights_path)}")

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
            
            if action == 0:
                action_name = "noop"
            elif action in (1, 2):
                action_name = f"voice-{action-1}"
            elif action in (3, 4):
                action_name = f"video-{action-3}"
            elif action in (5, 6):
                action_name = f"be-{action-5}"
            elif action == 7:
                action_name = "multi"
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
    parser.add_argument('--max-episode-steps', type=int, default=100,
                        help='Max steps per episode (default: 100)')
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
    parser.add_argument('--resume-eps', type=float, default=None,
                        help='Reset epsilon to this value when resuming (e.g., 0.10)')
    parser.add_argument('--traffic-weights', type=str, default=None,
                        help='Traffic category weights as "light:0.2,medium:0.3,high:0.5"')
    parser.add_argument('--traffic-profile', type=str, default=None,
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
    parser.add_argument('--influx-url', default='http://192.168.201.1:8086')
    parser.add_argument('--influx-org', default='research')
    parser.add_argument('--influx-bucket', default='INT')
    parser.add_argument('--influx-token', 
                        default='0fO0ojKAANp-7aEehJHRDWEKE-cSNoIEHY2aK8dd1KI0VWpmO1GAsMJhRh_B1U8bXDIaozHMDVv1yEkCPm230w==')
    
    # Controller output verbosity
    parser.add_argument('--verbose', action='store_true',
                        help='Show verbose P4 controller output (route add/delete messages)')
    parser.add_argument('--no-warm-start', action='store_true',
                        help='Force baseline resets (no warm-start episodes)')
    
    args = parser.parse_args()
    
    # Setup logging based on verbose flag
    setup_logging(args.verbose)
    
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

