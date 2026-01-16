#!/usr/bin/env python3
"""
stress_test_traffic.py - Stress test for traffic generator reliability.

Runs rapid start/stop cycles to validate TaskServer thread handling and
detect resource exhaustion issues before they occur in production.

Test Parameters:
- 3600 cycles (1 hour at 1 second per cycle, or configurable)
- Random profile selection per cycle
- 10-second traffic duration per cycle (configurable)
- Comprehensive statistics collection

Usage:
    sudo python3 stress_test_traffic.py --cycles 3600 --duration 10 --config config/topologies/fat_tree_k4.yaml
"""

import os
import sys
import time
import random
import argparse
import subprocess
import threading
import csv
import json
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path (parent of test/ directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging BEFORE importing other modules to prevent duplicate handlers
import logging
from logging_config import setup_unified_logging, set_console_level

# Initialize logging early - INFO on console, DEBUG to file
setup_unified_logging("stress_test", log_level="debug")
set_console_level(logging.INFO)

log = logging.getLogger(__name__)

# Now import TrafficManager (after logging is configured)
from traffic_generator import TrafficManager


@dataclass
class CycleStats:
    """Statistics for a single start/stop cycle."""
    cycle_num: int
    profile_name: str
    start_time: float
    stop_time: float = 0.0

    # Process counts
    expected_processes: int = 0
    processes_after_start: int = 0
    processes_after_duration: int = 0  # Count after running for duration
    processes_after_stop: int = 0
    degraded: bool = False  # True if processes died during traffic phase

    # Timing
    start_duration_ms: float = 0.0
    stop_duration_ms: float = 0.0

    # Status
    start_success: bool = False
    stop_success: bool = False
    verification_passed: bool = False

    # Debug info
    error_message: str = ""
    taskserver_status: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TestSummary:
    """Aggregate statistics for the entire test run."""
    total_cycles: int = 0
    successful_starts: int = 0
    successful_stops: int = 0
    verification_passes: int = 0

    # Process tracking
    min_processes_seen: int = 999999
    max_processes_seen: int = 0
    avg_processes_after_start: float = 0.0

    # Timing
    total_start_time_ms: float = 0.0
    total_stop_time_ms: float = 0.0
    max_start_time_ms: float = 0.0
    max_stop_time_ms: float = 0.0

    # Failures
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0
    failure_cycles: List[int] = field(default_factory=list)
    degraded_cycles: int = 0  # Cycles where processes died during traffic phase

    # TaskServer health
    taskserver_restarts: int = 0
    unresponsive_hosts: List[str] = field(default_factory=list)


class StressTest:
    """Stress test runner for traffic generator."""

    def __init__(self, config_path: str, cycles: int = 3600,
                 duration: float = 10.0, verify_threshold: float = 0.7):
        """Initialize stress test.

        Args:
            config_path: Path to topology YAML config
            cycles: Number of start/stop cycles to run
            duration: Seconds to run traffic per cycle
            verify_threshold: Minimum fraction of expected processes (0.0-1.0)
        """
        self.config_path = config_path
        self.cycles = cycles
        self.duration = duration
        self.verify_threshold = verify_threshold

        # Initialize traffic manager
        self.tm = TrafficManager(config_path=config_path)

        # Calculate expected processes
        # Each traffic pair has 3 queues, each queue has server + client = 6 processes per pair
        self.expected_processes = len(self.tm.traffic_pairs) * len([0, 1, 7]) * 2
        log.info(f"[StressTest] Expected processes per cycle: {self.expected_processes}")
        log.info(f"[StressTest] Traffic pairs: {len(self.tm.traffic_pairs)}")
        log.info(f"[StressTest] Hosts: {self.tm.traffic_hosts}")
        log.info(f"[StressTest] Host IPs: {self.tm.hosts_ips}")
        log.info(f"[StressTest] Senders: {self.tm.senders}")
        log.info(f"[StressTest] Receivers: {self.tm.receivers}")

        # Log first few traffic pairs for debugging
        for i, (src, dst, flow_id) in enumerate(self.tm.traffic_pairs[:5]):
            src_ip = self.tm.hosts_ips.get(src, 'UNKNOWN')
            dst_ip = self.tm.hosts_ips.get(dst, 'UNKNOWN')
            log.info(f"[StressTest] Pair {i}: {src}({src_ip}) -> {dst}({dst_ip}) flow_id={flow_id}")

        # Statistics
        self.summary = TestSummary()
        self.cycle_stats: List[CycleStats] = []

        # Only use medium and high profiles for stress test (generate meaningful CPU load)
        self.all_profiles = ['medium_1', 'medium_2', 'high_1', 'high_2']

        # Output directory
        self.output_dir = Path("log/stress_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test state
        self._running = True
        self._start_time = 0.0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        log.warning(f"[StressTest] Received signal {signum}, stopping...")
        self._running = False

    def _count_iperf_processes(self) -> int:
        """Count running iperf3 processes."""
        try:
            result = subprocess.run(
                ['pgrep', '-c', 'iperf3'],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception as e:
            log.warning(f"[StressTest] Error counting processes: {e}")
            return -1

    def _check_taskserver_sockets(self) -> Dict[str, bool]:
        """Check if TaskServer sockets exist for all hosts."""
        status = {}
        for host in self.tm.traffic_hosts:
            socket_path = f"/tmp/{host}_socket"
            status[host] = os.path.exists(socket_path)
        return status

    def _verify_traffic_started(self, timeout: float = 8.0) -> Tuple[bool, int]:
        """Verify traffic processes actually started.

        Args:
            timeout: Max time to wait for processes

        Returns:
            Tuple of (success, process_count)
        """
        threshold = int(self.expected_processes * self.verify_threshold)
        start_time = time.monotonic()
        last_count = 0

        while time.monotonic() - start_time < timeout:
            count = self._count_iperf_processes()
            if count >= threshold:
                return True, count
            if count != last_count:
                log.debug(f"[StressTest] Waiting: {count}/{self.expected_processes} (need {threshold})")
                last_count = count
            time.sleep(0.5)

        return False, last_count

    def _verify_traffic_stopped(self, timeout: float = 5.0) -> Tuple[bool, int]:
        """Verify traffic processes actually stopped.

        Args:
            timeout: Max time to wait for processes to stop

        Returns:
            Tuple of (success, remaining_process_count)
        """
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            count = self._count_iperf_processes()
            if count == 0:
                return True, 0
            time.sleep(0.3)

        # Still some processes running
        count = self._count_iperf_processes()
        return count == 0, count

    def _run_cycle(self, cycle_num: int) -> CycleStats:
        """Run a single start/stop cycle.

        Args:
            cycle_num: Current cycle number (1-indexed)

        Returns:
            CycleStats for this cycle
        """
        # Select random profile
        profile = random.choice(self.all_profiles)

        stats = CycleStats(
            cycle_num=cycle_num,
            profile_name=profile,
            start_time=time.time(),
            expected_processes=self.expected_processes
        )

        # Check TaskServer sockets before start
        stats.taskserver_status = self._check_taskserver_sockets()
        missing_sockets = [h for h, exists in stats.taskserver_status.items() if not exists]
        socket_count = sum(1 for exists in stats.taskserver_status.values() if exists)

        # === CYCLE START BANNER ===
        log.info(f"")
        log.info(f"{'='*60}")
        log.info(f"[Cycle {cycle_num}/{self.cycles}] STARTING - Profile: {profile}")
        log.info(f"{'='*60}")
        log.info(f"[Cycle {cycle_num}] TaskServer sockets: {socket_count}/{len(self.tm.traffic_hosts)} ready")
        if missing_sockets:
            log.warning(f"[Cycle {cycle_num}] Missing TaskServer sockets: {missing_sockets}")

        # === START TRAFFIC ===
        log.info(f"[Cycle {cycle_num}] Starting traffic with profile '{profile}'...")
        start_t0 = time.monotonic()
        try:
            self.tm.start_traffic(profile_name=profile)
            stats.start_duration_ms = (time.monotonic() - start_t0) * 1000

            # Verify processes started
            log.info(f"[Cycle {cycle_num}] Verifying traffic processes started...")
            stats.verification_passed, stats.processes_after_start = self._verify_traffic_started()
            stats.start_success = stats.verification_passed

            if not stats.verification_passed:
                stats.error_message = f"Only {stats.processes_after_start}/{self.expected_processes} processes started"
                log.error(f"[Cycle {cycle_num}] START FAILED: {stats.error_message}")
            else:
                log.info(f"[Cycle {cycle_num}] START OK: {stats.processes_after_start}/{self.expected_processes} "
                        f"processes in {stats.start_duration_ms:.0f}ms")

        except Exception as e:
            stats.start_duration_ms = (time.monotonic() - start_t0) * 1000
            stats.error_message = str(e)
            stats.start_success = False
            log.error(f"[Cycle {cycle_num}] START EXCEPTION: {e}")

        # === TRAFFIC RUNNING PHASE ===
        if stats.start_success:
            log.info(f"[Cycle {cycle_num}] Traffic running for {self.duration}s...")
            time.sleep(self.duration)

            # Check process count during running phase
            running_count = self._count_iperf_processes()
            stats.processes_after_duration = running_count
            threshold = int(self.expected_processes * self.verify_threshold)
            log.info(f"[Cycle {cycle_num}] After {self.duration}s: {running_count}/{self.expected_processes} processes running")
            if running_count < threshold:
                stats.degraded = True
                log.warning(f"[Cycle {cycle_num}] DEGRADED: Only {running_count}/{self.expected_processes} "
                           f"processes (threshold: {threshold}) - processes died during traffic phase!")

        # === STOP TRAFFIC ===
        log.info(f"[Cycle {cycle_num}] Stopping traffic...")
        stop_t0 = time.monotonic()
        try:
            self.tm.stop_traffic()
            stats.stop_duration_ms = (time.monotonic() - stop_t0) * 1000

            # Verify processes stopped
            stop_ok, remaining = self._verify_traffic_stopped()
            stats.processes_after_stop = remaining
            stats.stop_success = stop_ok

            if not stop_ok:
                log.warning(f"[Cycle {cycle_num}] STOP INCOMPLETE: {remaining} processes still running")
            else:
                log.info(f"[Cycle {cycle_num}] STOP OK: All processes terminated in {stats.stop_duration_ms:.0f}ms")

        except Exception as e:
            stats.stop_duration_ms = (time.monotonic() - stop_t0) * 1000
            stats.stop_success = False
            stats.error_message += f" | Stop error: {e}"
            log.error(f"[Cycle {cycle_num}] STOP EXCEPTION: {e}")

        stats.stop_time = time.time()
        cycle_duration = stats.stop_time - stats.start_time

        # === CYCLE END SUMMARY ===
        status = "SUCCESS" if (stats.start_success and stats.stop_success and not stats.degraded) else "FAILED"
        if stats.degraded:
            status = "DEGRADED"
        log.info(f"[Cycle {cycle_num}] COMPLETED ({status}) - Duration: {cycle_duration:.1f}s")
        log.info(f"[Cycle {cycle_num}] Summary: start={stats.start_duration_ms:.0f}ms, "
                f"stop={stats.stop_duration_ms:.0f}ms, "
                f"procs_start={stats.processes_after_start}, procs_end={stats.processes_after_duration}")
        log.info(f"{'='*60}")
        log.info(f"")

        return stats

    def _update_summary(self, stats: CycleStats):
        """Update aggregate summary with cycle stats."""
        self.summary.total_cycles += 1

        if stats.start_success:
            self.summary.successful_starts += 1
            self.summary.consecutive_failures = 0
        else:
            self.summary.consecutive_failures += 1
            self.summary.max_consecutive_failures = max(
                self.summary.max_consecutive_failures,
                self.summary.consecutive_failures
            )
            self.summary.failure_cycles.append(stats.cycle_num)

        if stats.stop_success:
            self.summary.successful_stops += 1

        if stats.verification_passed:
            self.summary.verification_passes += 1

        if stats.degraded:
            self.summary.degraded_cycles += 1

        # Process tracking
        if stats.processes_after_start > 0:
            self.summary.min_processes_seen = min(
                self.summary.min_processes_seen,
                stats.processes_after_start
            )
            self.summary.max_processes_seen = max(
                self.summary.max_processes_seen,
                stats.processes_after_start
            )

        # Timing
        self.summary.total_start_time_ms += stats.start_duration_ms
        self.summary.total_stop_time_ms += stats.stop_duration_ms
        self.summary.max_start_time_ms = max(self.summary.max_start_time_ms, stats.start_duration_ms)
        self.summary.max_stop_time_ms = max(self.summary.max_stop_time_ms, stats.stop_duration_ms)

        # Calculate running average
        if self.summary.successful_starts > 0:
            total_processes = sum(s.processes_after_start for s in self.cycle_stats if s.start_success)
            self.summary.avg_processes_after_start = total_processes / self.summary.successful_starts

    def _print_progress(self, cycle_num: int, stats: CycleStats):
        """Print progress update."""
        elapsed = time.time() - self._start_time
        rate = cycle_num / elapsed if elapsed > 0 else 0
        eta_seconds = (self.cycles - cycle_num) / rate if rate > 0 else 0
        eta_mins = eta_seconds / 60

        success_rate = self.summary.successful_starts / self.summary.total_cycles * 100

        status = "OK" if stats.start_success else "FAIL"
        log.info(f"[Progress] Cycle {cycle_num}/{self.cycles} ({status}) | "
                f"Success: {success_rate:.1f}% | "
                f"Rate: {rate:.2f}/s | "
                f"ETA: {eta_mins:.1f}m")

    def _save_results(self):
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save cycle-by-cycle CSV
        csv_path = self.output_dir / f"stress_test_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'cycle', 'profile', 'start_time', 'stop_time',
                'expected_procs', 'procs_after_start', 'procs_after_stop',
                'start_ms', 'stop_ms', 'start_ok', 'stop_ok', 'verified',
                'error'
            ])
            for s in self.cycle_stats:
                writer.writerow([
                    s.cycle_num, s.profile_name,
                    datetime.fromtimestamp(s.start_time).isoformat(),
                    datetime.fromtimestamp(s.stop_time).isoformat() if s.stop_time else '',
                    s.expected_processes, s.processes_after_start, s.processes_after_stop,
                    f"{s.start_duration_ms:.1f}", f"{s.stop_duration_ms:.1f}",
                    s.start_success, s.stop_success, s.verification_passed,
                    s.error_message
                ])
        log.info(f"[StressTest] Saved cycle data to {csv_path}")

        # Save summary JSON
        summary_path = self.output_dir / f"stress_test_summary_{timestamp}.json"
        summary_dict = {
            'test_params': {
                'config_path': self.config_path,
                'cycles_requested': self.cycles,
                'cycles_completed': self.summary.total_cycles,
                'duration_per_cycle': self.duration,
                'verify_threshold': self.verify_threshold,
                'expected_processes': self.expected_processes,
            },
            'timing': {
                'start_time': datetime.fromtimestamp(self._start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_runtime_seconds': time.time() - self._start_time,
            },
            'results': {
                'successful_starts': self.summary.successful_starts,
                'successful_stops': self.summary.successful_stops,
                'verification_passes': self.summary.verification_passes,
                'start_success_rate': self.summary.successful_starts / max(1, self.summary.total_cycles),
                'stop_success_rate': self.summary.successful_stops / max(1, self.summary.total_cycles),
            },
            'process_stats': {
                'min_processes_seen': self.summary.min_processes_seen if self.summary.min_processes_seen < 999999 else 0,
                'max_processes_seen': self.summary.max_processes_seen,
                'avg_processes_after_start': self.summary.avg_processes_after_start,
            },
            'timing_stats': {
                'avg_start_time_ms': self.summary.total_start_time_ms / max(1, self.summary.total_cycles),
                'avg_stop_time_ms': self.summary.total_stop_time_ms / max(1, self.summary.total_cycles),
                'max_start_time_ms': self.summary.max_start_time_ms,
                'max_stop_time_ms': self.summary.max_stop_time_ms,
            },
            'failures': {
                'max_consecutive_failures': self.summary.max_consecutive_failures,
                'failure_cycles': self.summary.failure_cycles[:50],  # First 50 failures
                'total_failures': len(self.summary.failure_cycles),
                'degraded_cycles': self.summary.degraded_cycles,
            }
        }
        with open(summary_path, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        log.info(f"[StressTest] Saved summary to {summary_path}")

    def _print_summary(self):
        """Print final test summary."""
        runtime = time.time() - self._start_time
        runtime_mins = runtime / 60

        print("\n" + "="*70)
        print("STRESS TEST SUMMARY")
        print("="*70)
        print(f"Runtime: {runtime_mins:.1f} minutes ({self.summary.total_cycles} cycles)")
        print(f"Expected processes: {self.expected_processes}")
        print()
        print("SUCCESS RATES:")
        print(f"  Start success:  {self.summary.successful_starts}/{self.summary.total_cycles} "
              f"({self.summary.successful_starts/max(1,self.summary.total_cycles)*100:.1f}%)")
        print(f"  Stop success:   {self.summary.successful_stops}/{self.summary.total_cycles} "
              f"({self.summary.successful_stops/max(1,self.summary.total_cycles)*100:.1f}%)")
        print(f"  Verified:       {self.summary.verification_passes}/{self.summary.total_cycles} "
              f"({self.summary.verification_passes/max(1,self.summary.total_cycles)*100:.1f}%)")
        print()
        print("PROCESS COUNTS:")
        min_seen = self.summary.min_processes_seen if self.summary.min_processes_seen < 999999 else 0
        print(f"  Min seen:       {min_seen}")
        print(f"  Max seen:       {self.summary.max_processes_seen}")
        print(f"  Avg after start: {self.summary.avg_processes_after_start:.1f}")
        print()
        print("TIMING:")
        avg_start = self.summary.total_start_time_ms / max(1, self.summary.total_cycles)
        avg_stop = self.summary.total_stop_time_ms / max(1, self.summary.total_cycles)
        print(f"  Avg start time: {avg_start:.0f}ms (max: {self.summary.max_start_time_ms:.0f}ms)")
        print(f"  Avg stop time:  {avg_stop:.0f}ms (max: {self.summary.max_stop_time_ms:.0f}ms)")
        print()
        print("FAILURES:")
        print(f"  Start failures: {len(self.summary.failure_cycles)}")
        print(f"  Degraded cycles: {self.summary.degraded_cycles} (processes died during traffic phase)")
        print(f"  Max consecutive: {self.summary.max_consecutive_failures}")
        if self.summary.failure_cycles:
            print(f"  First failures at cycles: {self.summary.failure_cycles[:10]}")
        print("="*70)

        # Final verdict
        success_rate = self.summary.successful_starts / max(1, self.summary.total_cycles)
        if success_rate >= 0.99:
            print("RESULT: PASS - Excellent reliability (>=99%)")
        elif success_rate >= 0.95:
            print("RESULT: PASS - Good reliability (>=95%)")
        elif success_rate >= 0.90:
            print("RESULT: WARNING - Marginal reliability (90-95%)")
        else:
            print(f"RESULT: FAIL - Poor reliability ({success_rate*100:.1f}%)")
        print("="*70 + "\n")

    def run(self):
        """Run the stress test."""
        log.info(f"[StressTest] Starting {self.cycles} cycles, {self.duration}s per cycle")
        log.info(f"[StressTest] Config: {self.config_path}")
        log.info(f"[StressTest] Verify threshold: {self.verify_threshold*100:.0f}%")

        self._start_time = time.time()

        try:
            for cycle in range(1, self.cycles + 1):
                if not self._running:
                    log.warning(f"[StressTest] Interrupted at cycle {cycle}")
                    break

                # Run cycle
                stats = self._run_cycle(cycle)
                self.cycle_stats.append(stats)
                self._update_summary(stats)

                # Progress update every 10 cycles
                if cycle % 10 == 0:
                    self._print_progress(cycle, stats)

                # Check for catastrophic failure (5+ consecutive failures)
                if self.summary.consecutive_failures >= 5:
                    log.error(f"[StressTest] ABORT: {self.summary.consecutive_failures} consecutive failures")
                    break

                # Brief pause between cycles (let OS cleanup)
                time.sleep(0.5)

        except Exception as e:
            log.error(f"[StressTest] Unexpected error: {e}")

        finally:
            # Ensure traffic is stopped
            try:
                self.tm.stop_traffic()
            except:
                pass

            # Save results
            self._save_results()
            self._print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Stress test for traffic generator reliability"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/topologies/fat_tree_k4.yaml',
        help='Path to topology YAML config'
    )
    parser.add_argument(
        '--cycles', '-n',
        type=int,
        default=3600,
        help='Number of start/stop cycles (default: 3600 for 1 hour)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=10.0,
        help='Seconds to run traffic per cycle (default: 10)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.7,
        help='Verification threshold 0.0-1.0 (default: 0.7 = 70%%)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test: 100 cycles, 5s duration'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging (default: enabled)'
    )

    args = parser.parse_args()

    # Check root
    if os.geteuid() != 0:
        print("ERROR: Must run with sudo!")
        print("Usage: sudo python3 stress_test_traffic.py [options]")
        sys.exit(1)

    # Logging already configured at module load time (see top of file)

    # Quick test mode
    if args.quick:
        args.cycles = 100
        args.duration = 5.0
        log.info("[StressTest] Quick mode: 100 cycles, 5s each")

    # Run test
    test = StressTest(
        config_path=args.config,
        cycles=args.cycles,
        duration=args.duration,
        verify_threshold=args.threshold
    )
    test.run()


if __name__ == "__main__":
    main()
