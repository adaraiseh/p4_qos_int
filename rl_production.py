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
from datetime import datetime
from collections import deque
from typing import Dict, Optional
import numpy as np

import torch

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Import from the main RL agent module
from rl_agent_4 import (
    DuelingDQN, QoSRoutingEnv,
    STATE_DIM, ACTION_DIM, HIDDEN_DIM, QIDS, SLA_THRESHOLDS
)
from traffic_generator import TrafficManager

# =============================================================================
#                              LOGGING SETUP
# =============================================================================
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
        """Load model weights."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['online_net'])
        log.info(f"Model loaded from {path}")


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
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=5000)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        # Rolling statistics
        self.rewards = deque(maxlen=100)
        self.sla_met_history = deque(maxlen=100)
        self.action_history = deque(maxlen=100)
        
        # Cumulative tracking
        self.cumulative_reward = 0.0
        self.total_sla_checks = 0
        self.total_sla_met = 0
        self.total_steps = 0
        self.start_time = time.time()
    
    def write_metrics(self, step: int, action: int, reward: float, 
                      info: Dict, q_stats: Dict):
        """
        Write comprehensive production metrics to InfluxDB.
        
        Measurement: rl_production
        """

        # Update rolling stats
        self.total_steps += 1
        self.rewards.append(reward)
        sla_met_count = len(info.get('sla_met', []))
        self.sla_met_history.append(sla_met_count == len(QIDS))
        self.action_history.append(action)
        
        # Update cumulative
        self.cumulative_reward += reward
        self.total_sla_checks += len(QIDS)
        self.total_sla_met += sla_met_count
        
        # Calculate derived metrics
        avg_reward_100 = np.mean(self.rewards) if self.rewards else 0.0
        sla_compliance_rate = (sum(self.sla_met_history) / len(self.sla_met_history) * 100) if self.sla_met_history else 0.0
        action_rate = sum(1 for a in self.action_history if a != 0) / len(self.action_history) if self.action_history else 0.0
        uptime_seconds = time.time() - self.start_time
        
        # Action name mapping
        action_names = {
            0: "noop",
            1: "v0-alt0", 2: "v0-alt1", 3: "v0-alt2",
            4: "v1-alt0", 5: "v1-alt1", 6: "v1-alt2",
            7: "be-alt0", 8: "be-alt1", 9: "be-alt2",
        }
        action_name = action_names.get(action, f"unk-{action}")
        
        try:
            p = (
                Point("rl_production")
                # Core metrics
                .field("step", int(step))
                .field("action", int(action))
                .field("action_name", action_name)  # Field, not tag - keeps single series
                .field("reward", float(reward))
                .field("raw_reward", float(info.get('raw_reward', reward)))
                
                .field("q_values_max", float(q_stats['q_max']))
                .field("q_values_mean", float(q_stats['q_mean']))
                .field("chosen_q", float(q_stats.get('chosen_q', 0.0)))
                .field("q_gap", float(q_stats.get('q_gap', 0.0)))
                
                # SLA metrics
                .field("sla_met_count", int(sla_met_count))
                .field("sla_streak", int(info.get('sla_streak', 0)))
                .field("sla_compliance_rate", float(sla_compliance_rate))
                
                # Action details
                .field("action_applied", int(info.get('action_applied', False)))
                .field("action_cost", float(info.get('action_cost', 0.0)))
                
                # Network state
                .field("pressure", float(info.get('pressure', 0.0)))
                
                # Rolling/cumulative metrics
                .field("avg_reward_100", float(avg_reward_100))
                .field("cumulative_reward", float(self.cumulative_reward))
                .field("action_rate", float(action_rate))
                
                # Operational
                .field("uptime_seconds", float(uptime_seconds))
                .field("data_valid", int(info.get('data_valid', False)))
                
                .time(datetime.utcnow())
            )
            
            # Per-queue SLA tags
            sla_met_list = info.get('sla_met', [])
            for qid in QIDS:
                p = p.field(f"queue_{qid}_sla_met", int(qid in sla_met_list))
            
            # Per-queue latency (from per_queue info if available)
            per_queue = info.get('per_queue', {})
            for qid in QIDS:
                if qid in per_queue:
                    p = p.field(f"queue_{qid}_latency", float(per_queue[qid].get('lat', 0)))
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=[p])
            
        except Exception as e:
            log.warning(f"Failed to write production metrics: {e}")
    
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
        
        # Initialize components
        env = QoSRoutingEnv(
            args.influx_bucket, args.influx_token,
            args.influx_org, args.influx_url,
            verbose=args.verbose,

            reset_network=False,  # Production: never reset network state
            production_mode=True  # Production: continuous operation
        )
        agent = ProductionAgent(STATE_DIM, ACTION_DIM, device)
        metrics = ProductionMetricsWriter(
            args.influx_bucket, args.influx_org,
            args.influx_url, args.influx_token
        )
        
        # Load weights
        weights_path = os.path.join(args.save_dir, f"dqn_v4_{args.weights_tag}.pth")
        if not os.path.exists(weights_path):
            log.error(f"Weights file not found: {weights_path}")
            return
        agent.load(weights_path)
        
        # Traffic generation (optional) - fixed profile for entire run
        traffic_manager = None
        if args.generate_traffic:
            traffic_manager = TrafficManager()
            log.info(f"Traffic generation enabled with profile: {args.traffic_profile}")
            
            # Start traffic with fixed profile
            profile_info = traffic_manager.start_traffic(profile_name=args.traffic_profile)
            log.info(f"Started traffic: {profile_info['profile_name']} ({profile_info['profile_category']})")
            log.info("Waiting 20s for traffic to stabilize...")
            time.sleep(20.0)
        
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
                    burst_msg = traffic_manager.check_burst()
                    if burst_msg:
                        log.info(f"[Step {self.step}] {burst_msg}")
                
                # Get valid actions and select action (no exploration)
                valid_mask = env.get_valid_actions()
                action = agent.select_action(state, valid_mask)
                q_stats = agent.get_q_stats()
                
                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Write metrics to InfluxDB
                metrics.write_metrics(self.step, action, reward, info, q_stats)
                
                # Action name for logging
                action_names = {
                    0: "noop",
                    1: "v0-alt0", 2: "v0-alt1", 3: "v0-alt2",
                    4: "v1-alt0", 5: "v1-alt1", 6: "v1-alt2",
                    7: "be-alt0", 8: "be-alt1", 9: "be-alt2",
                }
                action_name = action_names.get(action, str(action))
                
                # Console logging
                if self.step % args.log_every == 0:
                    log.info(
                        f"[Step {self.step:5d}] "
                        f"action={action_name:8s} "
                        f"reward={reward:+.2f} "
                        f"Q_max={q_stats['q_max']:+.2f} "
                        f"sla={len(info['sla_met'])}/3 "
                        f"streak={info['sla_streak']}"
                    )
                
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
    parser.add_argument('--influx-url', default='http://192.168.201.1:8086')
    parser.add_argument('--influx-org', default='research')
    parser.add_argument('--influx-bucket', default='INT')
    parser.add_argument('--influx-token',
                        default='0fO0ojKAANp-7aEehJHRDWEKE-cSNoIEHY2aK8dd1KI0VWpmO1GAsMJhRh_B1U8bXDIaozHMDVv1yEkCPm230w==')
    
    # Controller
    parser.add_argument('--verbose', action='store_true',
                        help='Show verbose P4 controller output')
    
    # Traffic generation
    parser.add_argument('--generate-traffic', action='store_true',
                        help='Generate traffic with a fixed profile')
    parser.add_argument('--traffic-profile', type=str, default='high_1',
                        help='Traffic profile name: light_1, light_2, medium_1, medium_2, high_1, high_2, test_*')
    parser.add_argument('--bursty-mode', action='store_true',
                        help='Enable periodic BE bursts every 60s with random duration (10s-5min)')
    
    args = parser.parse_args()
    
    runner = ProductionRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
