#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_production.py - Production inference script for trained DQN agent

Runs the trained RL model in production mode with comprehensive metrics
logging to InfluxDB. No training or exploration - pure inference.

Key Features:
- Continuous operation without episode boundaries
- Comprehensive RL metrics logging to InfluxDB
- Graceful shutdown on SIGINT/SIGTERM
- Q-value logging for decision debugging
- Rolling statistics for monitoring

Usage:
    python3 rl_production.py --weights-tag best
    python3 rl_production.py --weights-tag final --steps 10000

Author: Research Team
"""

import os
import sys
import time
import signal
import logging
import argparse
import csv
import glob
from datetime import datetime
from collections import deque
from typing import Dict, Optional
import numpy as np

import torch

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Import from the main RL agent module
from rl_agent_4 import (
    DuelingDQN, QoSRoutingEnv, setup_logging,
    STATE_DIM, ACTION_DIM, HIDDEN_DIM, QIDS, SLA_THRESHOLDS
)
from traffic_generator import TrafficManager
from config.schema import MAX_SWITCHES
import rl_agent_4

# =============================================================================
#                     PRODUCTION TIMING OVERRIDES
# =============================================================================
# PHASE 2.1: Align production timing with training to avoid distribution shift
# Previously used 2x training timing which caused train-test mismatch
# Now: production timing matches training exactly
WINDOW_SECONDS = 1.0        # Match training (was 2.0s)
SAFETY_LAG_MS = 0           # Match training (was 200ms, now 0ms)
DELAY_AFTER_ACTION = 1.0    # Match training (was 1.6s, now 1.0s)
DELAY_NO_ACTION = 1.0       # Match training (was 1.6s, now 1.0s)

# Apply overrides to rl_agent_4 module so QoSRoutingEnv uses production timing
rl_agent_4.WINDOW_SECONDS = WINDOW_SECONDS
rl_agent_4.SAFETY_LAG_MS = SAFETY_LAG_MS
rl_agent_4.DELAY_AFTER_ACTION = DELAY_AFTER_ACTION
rl_agent_4.DELAY_NO_ACTION = DELAY_NO_ACTION

# =============================================================================
#                              LOGGING SETUP
# =============================================================================
# Use 'rl_agent_4' logger name to match training - this logger is configured
# by setup_logging() with proper console and file handlers
log = logging.getLogger('rl_agent_4')


# =============================================================================
#                        PRODUCTION AGENT (INFERENCE ONLY)
# =============================================================================
class ProductionAgent:
    """
    Lightweight agent for production inference only.
    No replay buffer, no training, no exploration.
    """
    
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        self.device = device
        self.action_dim = action_dim

        # Network (inference only)
        self.network = DuelingDQN(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.network.eval()

        # Q-value tracking for metrics
        self.last_stats = None

        # PHASE 2.3: Track model checkpoint for reload detection
        self.checkpoint_path = None
        self.checkpoint_mtime = None
    
    def select_action(self, state: np.ndarray, valid_mask: np.ndarray) -> int:
        """
        Select action greedily (no exploration).
        Also stores Q-values for metrics logging.
        """
        valid_actions = np.where(valid_mask)[0]
        
        if len(valid_actions) == 0:
            self.last_stats = None
            return 0  # Default to no-op
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_t).squeeze(0).cpu().numpy()
            
            # Mask invalid actions with -inf
            q_values_masked = q_values.copy()
            q_values_masked[~valid_mask] = -np.inf
            
            # Select action
            action = int(np.argmax(q_values_masked))
            
            # Compute stats on VALID actions only
            valid_q = q_values[valid_mask]
            chosen_q = float(q_values[action])
            
            # Gap: Difference between chosen Q and second best valid Q
            if len(valid_q) > 1:
                # Sort valid Qs descending
                sorted_valid = np.sort(valid_q)[::-1]
                # sorted_valid[0] is chosen_q (since check above ensures we picked max)
                second_best = float(sorted_valid[1])
                q_gap = chosen_q - second_best
            else:
                q_gap = 0.0
                
            self.last_stats = {
                'q_max': float(np.max(valid_q)),
                'q_mean': float(np.mean(valid_q)),
                'q_min': float(np.min(valid_q)),
                'chosen_q': chosen_q,
                'q_gap': q_gap
            }
            
            return action
    
    def get_q_stats(self) -> Dict:
        """Get Q-value statistics from last action selection."""
        if self.last_stats is None:
            return {'q_max': 0.0, 'q_mean': 0.0, 'q_min': 0.0, 'chosen_q': 0.0, 'q_gap': 0.0}
        
        return self.last_stats
    
    def load(self, path: str):
        """Load model weights and track checkpoint for reload detection."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['online_net'])

        # PHASE 2.3: Track checkpoint for reload detection
        self.checkpoint_path = path
        self.checkpoint_mtime = os.path.getmtime(path) if os.path.exists(path) else None

        log.info(f"Model loaded from {path}")

    def check_model_reload(self) -> bool:
        """Check if model checkpoint has been updated (for hot-reload detection).

        Returns:
            True if checkpoint has changed, False otherwise
        """
        if self.checkpoint_path is None or self.checkpoint_mtime is None:
            return False

        if not os.path.exists(self.checkpoint_path):
            return False

        current_mtime = os.path.getmtime(self.checkpoint_path)
        return current_mtime > self.checkpoint_mtime


# =============================================================================
#                        PRODUCTION METRICS WRITER
# =============================================================================
class ProductionMetricsWriter:
    """
    Writes comprehensive RL metrics to InfluxDB for production monitoring.
    """
    
    def __init__(self, bucket: str, org: str, url: str, token: str):
        self.bucket = bucket
        self.org = org
        # PHASE 2.4: Reduced timeout from 5000ms to 2000ms for faster failure detection
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=2000)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

        # Rolling statistics (for get_summary)
        self.rewards = deque(maxlen=100)

        # Cumulative tracking (for get_summary)
        self.cumulative_reward = 0.0
        self.total_sla_checks = 0
        self.total_sla_met = 0
        self.total_steps = 0
        self.start_time = time.time()

        # PHASE 2.2: Circuit breaker for write failures
        self.consecutive_write_failures = 0
        self.max_consecutive_failures = 3
        self.circuit_open = False
    
    def write_metrics(self, step: int, action: int, reward: float,
                      info: Dict, q_stats: Dict):
        """
        Write production metrics to InfluxDB.

        Measurement: rl_production

        Minimal fields: step, action, reward, q_values_max,
        sla_met_count, data_valid, pressure, queue_X_latency, queue_X_drops
        """

        # Update stats for get_summary
        self.total_steps += 1
        self.rewards.append(reward)
        sla_met_count = len(info.get('sla_met', []))
        self.cumulative_reward += reward
        self.total_sla_checks += len(QIDS)
        self.total_sla_met += sla_met_count

        try:
            p = (
                Point("rl_production")
                .field("step", int(step))
                .field("action", int(action))
                .field("reward", float(reward))
                .field("q_values_max", float(q_stats['q_max']))
                .field("sla_met_count", int(sla_met_count))
                .field("data_valid", int(info.get('data_valid', False)))
                .field("pressure", float(info.get('pressure', 0.0)))
                .time(datetime.utcnow())
            )

            # Per-queue latency and drops
            per_queue = info.get('per_queue', {})
            for qid in QIDS:
                if qid in per_queue:
                    p = p.field(f"queue_{qid}_latency", float(per_queue[qid].get('lat', 0)))
                    p = p.field(f"queue_{qid}_drops", float(per_queue[qid].get('drop', 0)))
            
            # PHASE 2.2: Circuit breaker pattern for write failures
            if self.circuit_open:
                log.warning("Circuit breaker OPEN - skipping metric write")
                return

            self.write_api.write(bucket=self.bucket, org=self.org, record=[p])

            # Success - reset failure counter
            if self.consecutive_write_failures > 0:
                log.info(f"Metric write recovered after {self.consecutive_write_failures} failures")
                self.consecutive_write_failures = 0

        except Exception as e:
            self.consecutive_write_failures += 1
            log.error(f"Failed to write production metrics (failure {self.consecutive_write_failures}/{self.max_consecutive_failures}): {e}")

            # Open circuit breaker if max failures reached
            if self.consecutive_write_failures >= self.max_consecutive_failures:
                self.circuit_open = True
                log.critical(f"CIRCUIT BREAKER OPEN: {self.max_consecutive_failures} consecutive metric write failures. "
                           f"Production metrics disabled to prevent cascading failures. "
                           f"Check InfluxDB connectivity and restart production script to recover.")
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_steps': self.total_steps,
            'cumulative_reward': self.cumulative_reward,
            'avg_reward_100': np.mean(self.rewards) if self.rewards else 0.0,
            'overall_sla_compliance': (self.total_sla_met / max(1, self.total_sla_checks)) * 100,
            'uptime_seconds': time.time() - self.start_time,
        }
    
    def close(self):
        """Clean up resources."""
        try:
            self.write_api.close()
            self.client.close()
        except Exception:
            pass


# =============================================================================
#                           PRODUCTION LOOP
# =============================================================================
class ProductionRunner:
    """
    Main production runner with graceful shutdown support.
    """
    
    def __init__(self, args):
        self.args = args
        self.running = True
        self.step = 0
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        log.info(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def run(self):
        """Main production loop."""
        args = self.args
        
        log.info("=" * 60)
        log.info("Starting RL Production - DQN Agent v4")
        log.info("=" * 60)
        log.info(f"Configuration:")
        log.info(f"  Weights: {args.weights_tag}")
        log.info(f"  Max steps: {args.steps} (0 = infinite)")
        log.info(f"  Log frequency: every {args.log_every} steps")
        log.info("=" * 60)
        
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
                topo_name = topology_builder.config.topology.name.replace('-', '_')
                rules_dir = f"rules/{topo_name}"
            log.info(f"Topology: {topology_builder.config.topology.name}")
            log.info(f"  Switches: {len(topology_builder.switches)}/{MAX_SWITCHES} max")
            log.info(f"  Rules dir: {rules_dir}")

        # Initialize components
        env = QoSRoutingEnv(
            args.influx_bucket, args.influx_token,
            args.influx_org, args.influx_url,
            verbose=False,  # Controller verbosity disabled (logs go to file)
            reset_network=True,  # Reset network at start for clean baseline
            production_mode=True,  # Production: continuous operation
            topology_builder=topology_builder,
            rules_dir=rules_dir
        )
        agent = ProductionAgent(STATE_DIM, ACTION_DIM, device)
        metrics = ProductionMetricsWriter(
            args.influx_bucket, args.influx_org,
            args.influx_url, args.influx_token
        )
        
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
        
        # Traffic generation (optional) - fixed profile for entire run
        traffic_manager = None
        if args.generate_traffic:
            traffic_manager = TrafficManager(config_path=args.config)
            log.info(f"Traffic generation enabled with profile: {args.traffic_profile}")
            
            # Start traffic with fixed profile
            profile_info = traffic_manager.start_traffic(profile_name=args.traffic_profile)
            log.info(f"Started traffic: {profile_info['profile_name']} ({profile_info['profile_category']})")
            log.info("Waiting 5s for traffic to stabilize...")
            time.sleep(5.0)
        
        # CSV logging
        os.makedirs('data', exist_ok=True)
        csv_path = f"data/production_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'step', 'action', 'action_name', 'reward', 'raw_reward',
            'q_max', 'q_mean', 'chosen_q', 'q_gap',
            'sla_met_count', 'sla_streak', 'action_applied', 'pressure', 'timestamp'
        ])
        log.info(f"Production log: {csv_path}")
        
        # Initialize environment
        state = env.reset()
        log.info("Environment initialized, starting production loop...")
        
        try:
            while self.running:
                self.step += 1
                
                # Check step limit
                if args.steps > 0 and self.step > args.steps:
                    log.info(f"Reached step limit ({args.steps}), stopping...")
                    break
                
                # Check for bursty mode traffic changes
                if traffic_manager and args.bursty_mode:
                    burst_msg = traffic_manager.check_burst(
                        burst_interval=30.0,    # Every 30 seconds
                        min_duration=5.0,       # Minimum 5 seconds
                        max_duration=120.0,     # Maximum 2 minutes
                        base_profile=args.traffic_profile,
                        burst_profile=args.burst_profile
                    )
                    if burst_msg:
                        log.info(f"[Step {self.step}] {burst_msg}")

                # PHASE 2.3: Check for model reload (hot-reload support)
                # If checkpoint file is updated, reload model and clear stacks
                if agent.check_model_reload():
                    log.warning(f"[Step {self.step}] Model checkpoint updated, reloading...")
                    agent.load(agent.checkpoint_path)
                    env.clear_stacks()
                    log.info(f"[Step {self.step}] Model reloaded and stacks cleared")

                # Get valid actions and select action (no exploration)
                valid_mask = env.get_valid_actions()
                action = agent.select_action(state, valid_mask)
                q_stats = agent.get_q_stats()
                
                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)

                # Console logging (matches training format) - placed immediately after env.step
                action_names = {
                    0: "noop",
                    1: "vo-alt0", 2: "vo-alt1",           # Voice
                    3: "vi-alt0", 4: "vi-alt1",           # Video
                    5: "be-alt0", 6: "be-alt1",           # BE
                    7: "multi",                            # Multi-queue
                }
                action_name = action_names.get(action, str(action))
                log.info(f"[Step {self.step}] action={action_name:8s} reward={reward:+.2f} sla={len(info['sla_met'])}/3 streak={info['sla_streak']}")

                # Write metrics to InfluxDB
                metrics.write_metrics(self.step, action, reward, info, q_stats)
                
                # CSV logging
                csv_writer.writerow([
                    self.step, action, action_name, reward,
                    info.get('raw_reward', reward),
                    q_stats['q_max'], q_stats['q_mean'],
                    q_stats.get('chosen_q', 0.0), q_stats.get('q_gap', 0.0),
                    len(info['sla_met']), info.get('sla_streak', 0),
                    int(info.get('action_applied', False)),
                    info.get('pressure', 0),
                    datetime.now().isoformat()
                ])
                csv_file.flush()
                
                # Update state (production mode never terminates/truncates)
                state = next_state
        
        except Exception as e:
            log.error(f"Production loop error: {e}", exc_info=True)
        
        finally:
            # Graceful shutdown
            log.info("\nShutting down gracefully...")
            
            # Print summary
            summary = metrics.get_summary()
            log.info("=" * 60)
            log.info("Production Run Summary:")
            log.info(f"  Total steps: {self.step}")
            log.info(f"  Cumulative reward: {summary['cumulative_reward']:.2f}")
            log.info(f"  Avg reward (last 100): {summary['avg_reward_100']:.2f}")
            log.info(f"  SLA compliance: {summary['overall_sla_compliance']:.1f}%")
            log.info(f"  Uptime: {summary['uptime_seconds']:.1f}s")
            log.info("=" * 60)
            
            # Clean up
            csv_file.close()
            metrics.close()
            env.close()
            if traffic_manager:
                traffic_manager.stop_traffic()
                log.info("Stopped traffic generation")
            
            log.info(f"Production log saved to: {csv_path}")
            log.info("Shutdown complete.")


# =============================================================================
#                                 MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DQN Agent v4 - Production Inference")
    
    # Model
    parser.add_argument('--save-dir', default='training_files',
                        help='Directory containing model weights')
    parser.add_argument('--weights-tag', default='best',
                        help='Weight file tag (e.g., final, best, 50pct)')
    
    # Operation
    parser.add_argument('--steps', type=int, default=0,
                        help='Max steps (0 = infinite, run until interrupted)')
    parser.add_argument('--log-every', type=int, default=1,
                        help='Console log frequency')
    
    # InfluxDB
    parser.add_argument('--influx-url', default='http://192.168.56.1:8086')
    parser.add_argument('--influx-org', default='Research')
    parser.add_argument('--influx-bucket', default='INT')
    parser.add_argument('--influx-token', default=os.environ.get('INFLUX_TOKEN'),
                        help='InfluxDB token (or set INFLUX_TOKEN env var)')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='Console log level (file always logs DEBUG)')

    # Topology configuration
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to YAML topology configuration file')
    parser.add_argument('--rules-dir', type=str, default=None,
                        help='Directory containing P4 rule files')

    # Traffic generation
    parser.add_argument('--generate-traffic', action='store_true',
                        help='Generate traffic with a fixed profile')
    parser.add_argument('--traffic-profile', type=str, default='high_1',
                        help='Traffic profile name: light_1, light_2, medium_1, medium_2, high_1, high_2, test_*')
    parser.add_argument('--bursty-mode', action='store_true',
                        help='Enable periodic bursts every 30s with random duration (5s-2min)')
    parser.add_argument('--burst-profile', type=str, default='bursty_be_2',
                        help='Traffic profile to use during bursts (default: bursty_be_2)')
    
    args = parser.parse_args()

    # Setup logging (matches training: INFO console, DEBUG file with 50MB rotation)
    setup_logging(log_level=args.log_level)

    if not args.influx_token:
        log.error("InfluxDB token not configured. Set INFLUX_TOKEN environment variable or use --influx-token argument.")
        sys.exit(1)

    runner = ProductionRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
