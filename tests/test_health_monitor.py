#!/usr/bin/env python3
"""
Test script for TrafficManager health monitoring.

This script:
1. Starts traffic
2. Waits for health monitor to confirm processes are running
3. Kills some iperf processes to simulate a crash
4. Verifies health monitor detects and restarts them

Run with: sudo python3 test_health_monitor.py --config config/topologies/fat_tree_k4.yaml
"""
import os
import sys
import time
import subprocess
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_generator import TrafficManager
from logging_config import setup_unified_logging

log = logging.getLogger(__name__)


def count_iperf_processes():
    """Count running iperf3 processes."""
    try:
        result = subprocess.run(['pgrep', '-c', 'iperf3'],
                                capture_output=True, text=True, timeout=5)
        return int(result.stdout.strip()) if result.returncode == 0 else 0
    except Exception:
        return 0


def kill_random_iperf(count: int = 10):
    """Kill some random iperf processes to simulate crash."""
    try:
        result = subprocess.run(['pgrep', 'iperf3'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return 0

        pids = result.stdout.strip().split('\n')[:count]
        killed = 0
        for pid in pids:
            try:
                subprocess.run(['kill', '-9', pid], capture_output=True)
                killed += 1
            except Exception:
                pass
        return killed
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Test health monitoring")
    parser.add_argument('--config', '-c', type=str,
                        default='config/topologies/fat_tree_k4.yaml',
                        help='Topology config file')
    parser.add_argument('--profile', '-p', type=str, default='medium_2',
                        help='Traffic profile to use')
    parser.add_argument('--duration', '-d', type=int, default=120,
                        help='Test duration in seconds')
    args = parser.parse_args()

    setup_unified_logging(__name__, log_level="debug")

    if os.geteuid() != 0:
        log.error("Must run with sudo!")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Health Monitor Test")
    log.info("=" * 60)

    # Create TrafficManager
    tm = TrafficManager(config_path=args.config)

    log.info(f"\nStarting traffic with profile: {args.profile}")
    info = tm.start_traffic(profile_name=args.profile)
    log.info(f"Profile: {info['profile_name']} ({info['profile_category']})")
    log.info(f"Loads: Q0={info['loads'][0]:.2f}, Q1={info['loads'][1]:.2f}, Q7={info['loads'][7]:.2f}")

    # Wait for traffic to stabilize
    log.info("\nWaiting 15s for traffic to stabilize...")
    time.sleep(15)

    initial_count = count_iperf_processes()
    log.info(f"Initial iperf process count: {initial_count}")

    if initial_count == 0:
        log.error("No iperf processes running! Is the network up?")
        tm.stop_traffic()
        sys.exit(1)

    # Test loop
    test_start = time.time()
    crash_simulated = False
    restart_detected = False

    while time.time() - test_start < args.duration:
        elapsed = time.time() - test_start
        current_count = count_iperf_processes()

        # After 30 seconds, simulate a crash if not done yet
        if elapsed > 30 and not crash_simulated:
            log.info("\n" + "=" * 40)
            log.info("SIMULATING CRASH: Killing 50% of iperf processes")
            log.info("=" * 40)

            to_kill = max(1, initial_count // 2)
            killed = kill_random_iperf(to_kill)
            log.info(f"Killed {killed} iperf processes")
            crash_simulated = True
            time.sleep(2)
            post_kill = count_iperf_processes()
            log.info(f"Processes after kill: {post_kill}")
            continue

        # Check if restart was detected
        if crash_simulated and not restart_detected:
            if current_count >= initial_count * 0.9:
                log.info("\n" + "=" * 40)
                log.info("SUCCESS: Health monitor restarted traffic!")
                log.info(f"Processes restored: {current_count}")
                log.info("=" * 40)
                restart_detected = True

        # Log status every 10 seconds
        if int(elapsed) % 10 == 0:
            log.info(f"[{elapsed:.0f}s] iperf count: {current_count}, "
                    f"restart_count: {tm._restart_count}")

        time.sleep(1)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("TEST SUMMARY")
    log.info("=" * 60)
    log.info(f"Duration: {args.duration}s")
    log.info(f"Initial processes: {initial_count}")
    log.info(f"Final processes: {count_iperf_processes()}")
    log.info(f"Total restarts by health monitor: {tm._restart_count}")
    log.info(f"Crash simulated: {crash_simulated}")
    log.info(f"Restart detected: {restart_detected}")

    if crash_simulated and restart_detected:
        log.info("RESULT: PASS - Health monitor working correctly")
    elif crash_simulated and not restart_detected:
        log.info("RESULT: FAIL - Health monitor did not restart traffic")
    else:
        log.info("RESULT: INCOMPLETE - Crash not simulated")

    log.info("\nStopping traffic...")
    tm.stop_traffic()
    log.info("Done")


if __name__ == "__main__":
    main()
