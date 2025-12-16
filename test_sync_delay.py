#!/usr/bin/env python3
"""
test_sync_delay.py - Measure synchronization delay between route changes and InfluxDB metrics

This script measures the time it takes for route changes to be reflected in InfluxDB metrics.
It helps fine-tune the timing parameters in rl_agent_4.py (DELAY_AFTER_ACTION, WINDOW_SECONDS, etc.)

Usage:
    python3 test_sync_delay.py [--iterations N] [--poll-interval-ms MS]

The test performs:
1. Captures baseline metrics from InfluxDB
2. Applies a significant route change
3. Rapidly polls InfluxDB to detect when metrics change
4. Measures time to first change, time to stabilization
5. Reverts the route change
6. Repeats for multiple iterations to get statistical measures

Author: Test script for rl_agent_4.py timing calibration
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple, Optional
import statistics
import json

import numpy as np
from influxdb_client import InfluxDBClient

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)

# Import controller for route changes
from controller import Controller


def _silent_init_controller(verbose: bool = False) -> Controller:
    """Initialize controller silently to avoid p4utils verbose output."""
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        return Controller(verbose=verbose)


# =============================================================================
#                              CONFIGURATION
# =============================================================================
# InfluxDB default settings (same as rl_agent_4.py)
DEFAULT_INFLUX_URL = 'http://192.168.201.1:8086'
DEFAULT_INFLUX_ORG = 'research'
DEFAULT_INFLUX_BUCKET = 'INT'
DEFAULT_INFLUX_TOKEN = '0fO0ojKAANp-7aEehJHRDWEKE-cSNoIEHY2aK8dd1KI0VWpmO1GAsMJhRh_B1U8bXDIaozHMDVv1yEkCPm230w=='

# Queue IDs to monitor (same as RL agent)
QIDS = (0, 1, 7)

# SLA thresholds for context
SLA_THRESHOLDS = {
    0: 100.0,   # Voice
    1: 150.0,   # Video
    7: 200.0,   # Best-effort
}

# Metrics to monitor
METRICS = ['flow_latency', 'q_drop_rate_100ms', 'tx_utilization', 'switch_latency']


# =============================================================================
#                           METRICS COLLECTOR
# =============================================================================
class MetricsSnapshot:
    """A snapshot of metrics at a point in time."""
    
    def __init__(self):
        self.timestamp = time.monotonic()
        self.wall_time = datetime.utcnow()
        self.data: Dict[int, Dict[str, float]] = {qid: {} for qid in QIDS}
        self.path_info: Dict[int, List[str]] = {}  # qid -> path nodes
        
    def __repr__(self):
        return f"Snapshot(t={self.timestamp:.3f}, data={self.data})"
    
    def get_signature(self) -> str:
        """Get a string signature for comparison."""
        parts = []
        for qid in sorted(self.data.keys()):
            for metric, value in sorted(self.data[qid].items()):
                parts.append(f"{qid}:{metric}:{value:.2f}")
        return "|".join(parts)


class SyncTester:
    """Test synchronization delays between route changes and metric updates."""
    
    def __init__(self, bucket: str, token: str, org: str, url: str):
        self.bucket = bucket
        self.org = org
        self.url = url
        
        # InfluxDB client
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=5000)
        self.query_api = self.client.query_api()
        
        # Controller for routing changes (silent init to suppress p4utils output)
        log.info("Initializing P4 controller (this may take a moment)...")
        self.controller = _silent_init_controller(verbose=False)
        
        # Results storage
        self.results: List[Dict] = []
        
        log.info(f"SyncTester initialized")
        log.info(f"  InfluxDB: {url}")
        log.info(f"  Bucket: {bucket}")
        log.info(f"  Monitoring queues: {QIDS}")
    
    def _time_window(self, window_seconds: float = 1.0, 
                      safety_lag_ms: float = 100.0) -> Tuple[str, str]:
        """Get time window for queries with minimal lag for fast detection."""
        stop_dt = datetime.utcnow() - timedelta(milliseconds=safety_lag_ms)
        start_dt = stop_dt - timedelta(seconds=window_seconds)
        return start_dt.isoformat() + 'Z', stop_dt.isoformat() + 'Z'
    
    def collect_snapshot(self, window_seconds: float = 1.0) -> MetricsSnapshot:
        """Collect a snapshot of current metrics."""
        snapshot = MetricsSnapshot()
        start, stop = self._time_window(window_seconds)
        
        # Query all metrics at once
        flux = f'''
        base = from(bucket:"{self.bucket}")
            |> range(start:{start}, stop:{stop})
            |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
            |> toFloat()
        
        lat_mean = base
            |> filter(fn: (r) => r._measurement == "flow_latency")
            |> group(columns:["queue_id"])
            |> mean(column:"_value")
            |> set(key:"_measurement", value:"flow_latency")
        
        lat_p95 = base
            |> filter(fn: (r) => r._measurement == "flow_latency")
            |> group(columns:["queue_id"])
            |> quantile(q:0.95, method:"estimate_tdigest")
            |> set(key:"_measurement", value:"flow_latency_p95")
        
        drop_mean = base
            |> filter(fn: (r) => r._measurement == "q_drop_rate_100ms")
            |> group(columns:["queue_id"])
            |> mean(column:"_value")
            |> set(key:"_measurement", value:"drop_rate")
        
        util_mean = base
            |> filter(fn: (r) => r._measurement == "tx_utilization")
            |> group(columns:["queue_id"])
            |> mean(column:"_value")
            |> set(key:"_measurement", value:"tx_util")
        
        switch_lat = base
            |> filter(fn: (r) => r._measurement == "switch_latency")
            |> group(columns:["queue_id"])
            |> mean(column:"_value")
            |> set(key:"_measurement", value:"switch_latency")
        
        union(tables:[lat_mean, lat_p95, drop_mean, util_mean, switch_lat])
        '''
        
        try:
            tables = self.query_api.query(org=self.org, query=flux)
            for table in tables or []:
                for record in table.records:
                    try:
                        qid = int(record.values.get('queue_id', -1))
                        if qid not in snapshot.data:
                            continue
                        measurement = record.get_measurement()
                        value = record.get_value()
                        if value is not None:
                            snapshot.data[qid][measurement] = float(value)
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            log.warning(f"Failed to query metrics: {e}")
        
        return snapshot
    
    def find_active_flow(self, qid: int) -> Optional[Tuple[str, str, List[str], int]]:
        """
        Find an active flow for the given queue ID.
        
        Returns:
            (src_ip, dst_ip, path, bottleneck_sid) or None
        """
        start, stop = self._time_window(2.0)  # Slightly longer window to find flows
        
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
                        src_ip, dst_ip = str(src), str(dst)
                        
                        # Get path for this flow
                        path = self.controller.get_path_by_ips(src_ip, dst_ip)
                        if not path:
                            log.warning(f"No path found for {src_ip} -> {dst_ip}")
                            continue
                        
                        # Find switches on path
                        sw_names = [n for n in path if isinstance(n, str) and n[0] in ('t', 'a', 'c')]
                        
                        # Find a switch that has an alternate (agg or core)
                        for sw in sw_names:
                            sid = self.controller.switch_name_to_id.get(sw)
                            if sid is None:
                                continue
                            role = self.controller.switch_id_role.get(sid, 'other')
                            if role in ('agg', 'core'):
                                alt = self.controller.find_alternate_for_worst(sid, list(path))
                                if alt:
                                    log.info(f"Found flow qid={qid}: {src_ip}->{dst_ip}")
                                    log.info(f"  Path: {' -> '.join(path)}")
                                    log.info(f"  Bottleneck candidate: {sw} (sid={sid}, role={role})")
                                    log.info(f"  Alternate available: {alt}")
                                    return src_ip, dst_ip, list(path), sid
                        
                        log.warning(f"No alternate found for flow {src_ip} -> {dst_ip}")
        except Exception as e:
            log.warning(f"Failed to find active flow for qid={qid}: {e}")
        
        return None
    
    def detect_change(self, baseline: MetricsSnapshot, current: MetricsSnapshot,
                      threshold_pct: float = 10.0) -> Dict[int, Dict[str, float]]:
        """
        Detect changes between baseline and current snapshot.
        
        Returns:
            Dict of qid -> metric -> change_pct
        """
        changes = {}
        
        for qid in QIDS:
            qid_changes = {}
            for metric in baseline.data[qid]:
                if metric not in current.data[qid]:
                    continue
                
                old_val = baseline.data[qid][metric]
                new_val = current.data[qid][metric]
                
                if abs(old_val) < 0.001:
                    # Avoid division by zero
                    if abs(new_val) > 0.001:
                        change_pct = 100.0
                    else:
                        change_pct = 0.0
                else:
                    change_pct = abs((new_val - old_val) / old_val) * 100.0
                
                if change_pct >= threshold_pct:
                    qid_changes[metric] = change_pct
            
            if qid_changes:
                changes[qid] = qid_changes
        
        return changes
    
    def run_single_test(self, qid: int, poll_interval_ms: int = 100,
                        max_wait_seconds: float = 15.0,
                        stabilization_checks: int = 3) -> Optional[Dict]:
        """
        Run a single synchronization test.
        
        Args:
            qid: Queue ID to test
            poll_interval_ms: How often to poll InfluxDB (ms)
            max_wait_seconds: Maximum time to wait for changes
            stabilization_checks: Number of consistent readings to consider stable
        
        Returns:
            Test result dict or None if test failed
        """
        log.info(f"\n{'='*60}")
        log.info(f"Running sync test for qid={qid}")
        log.info(f"{'='*60}")
        
        # Step 1: Find an active flow for this queue
        flow_info = self.find_active_flow(qid)
        if not flow_info:
            log.error(f"No suitable flow found for qid={qid}")
            return None
        
        src_ip, dst_ip, path, bottleneck_sid = flow_info
        
        # Find alternate switch
        alt_switch = self.controller.find_alternate_for_worst(bottleneck_sid, path)
        if not alt_switch:
            log.error(f"No alternate switch found for bottleneck sid={bottleneck_sid}")
            return None
        
        # Step 2: Capture baseline metrics (average over multiple samples)
        log.info("\nCapturing baseline metrics...")
        baseline_samples = []
        for i in range(5):
            snapshot = self.collect_snapshot(window_seconds=0.5)
            baseline_samples.append(snapshot)
            time.sleep(0.2)
        
        # Use last snapshot as baseline (most recent)
        baseline = baseline_samples[-1]
        log.info(f"Baseline for qid={qid}: {baseline.data[qid]}")
        
        # Record the exact time of route change
        route_change_time = time.monotonic()
        route_change_wall = datetime.utcnow()
        
        # Step 3: Apply route change
        log.info(f"\nApplying route change at t=0.000s...")
        log.info(f"  Rerouting qid={qid}: {src_ip} -> {dst_ip}")
        log.info(f"  Bottleneck: sid={bottleneck_sid}, Alt: {alt_switch}")
        
        ok, msg = self.controller.reroute_one_demand_symmetric(
            src_ip=src_ip,
            dst_ip=dst_ip,
            qid=qid,
            worst_switch_id=bottleneck_sid,
            alt_switch_name=alt_switch
        )
        
        if not ok:
            log.error(f"Route change failed: {msg}")
            return None
        
        log.info(f"Route change applied successfully")
        
        # Step 4: Poll for changes
        log.info(f"\nPolling for metric changes (interval={poll_interval_ms}ms, max_wait={max_wait_seconds}s)...")
        
        poll_interval = poll_interval_ms / 1000.0
        first_change_time = None
        first_changes = None
        stable_readings = []
        all_snapshots = []
        
        start_poll = time.monotonic()
        last_significant_change = None
        
        while (time.monotonic() - route_change_time) < max_wait_seconds:
            current = self.collect_snapshot(window_seconds=0.5)
            elapsed = time.monotonic() - route_change_time
            all_snapshots.append((elapsed, current))
            
            # Check for changes from baseline
            changes = self.detect_change(baseline, current, threshold_pct=5.0)
            
            if qid in changes:
                if first_change_time is None:
                    first_change_time = elapsed
                    first_changes = changes[qid]
                    log.info(f"  t={elapsed:.3f}s: FIRST CHANGE DETECTED!")
                    log.info(f"    Changes: {changes[qid]}")
                
                last_significant_change = elapsed
                stable_readings = []  # Reset stability counter
            else:
                if first_change_time is not None:
                    # Track stability after first change
                    stable_readings.append(current)
                    if len(stable_readings) >= stabilization_checks:
                        log.info(f"  t={elapsed:.3f}s: Metrics stabilized after {stabilization_checks} consistent readings")
                        break
            
            # Log progress every second
            if int(elapsed) > int(elapsed - poll_interval):
                log.info(f"  t={elapsed:.1f}s: qid={qid} lat={current.data[qid].get('flow_latency', -1):.1f}ms")
            
            time.sleep(poll_interval)
        
        # Step 5: Revert the route change
        log.info(f"\nReverting route change...")
        reverted = self.controller.revert_last_change_for_qid(qid)
        log.info(f"Revert {'successful' if reverted else 'failed'}")
        
        # Step 6: Compile results
        stabilization_time = None
        if last_significant_change is not None and stable_readings:
            stabilization_time = last_significant_change
        
        result = {
            'qid': qid,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'bottleneck_sid': bottleneck_sid,
            'alt_switch': alt_switch,
            'route_change_success': ok,
            'first_change_time_s': first_change_time,
            'first_changes': first_changes,
            'stabilization_time_s': stabilization_time,
            'baseline_metrics': dict(baseline.data[qid]),
            'final_metrics': dict(all_snapshots[-1][1].data[qid]) if all_snapshots else {},
            'num_snapshots': len(all_snapshots),
            'poll_interval_ms': poll_interval_ms,
            'revert_success': reverted,
        }
        
        # Log summary
        log.info(f"\n{'='*60}")
        log.info(f"TEST RESULT for qid={qid}:")
        log.info(f"  First change detected: {first_change_time:.3f}s" if first_change_time else "  No change detected!")
        if stabilization_time:
            log.info(f"  Stabilization time: {stabilization_time:.3f}s")
        log.info(f"  First changes: {first_changes}")
        log.info(f"{'='*60}")
        
        return result
    
    def run_full_test(self, iterations: int = 3, poll_interval_ms: int = 100,
                      inter_test_delay: float = 5.0) -> List[Dict]:
        """
        Run full synchronization test across all queues.
        
        Args:
            iterations: Number of test iterations per queue
            poll_interval_ms: Polling interval in milliseconds
            inter_test_delay: Delay between tests to let network settle
        
        Returns:
            List of all test results
        """
        log.info("\n" + "="*70)
        log.info("STARTING FULL SYNCHRONIZATION DELAY TEST")
        log.info("="*70)
        log.info(f"Configuration:")
        log.info(f"  Iterations per queue: {iterations}")
        log.info(f"  Poll interval: {poll_interval_ms}ms")
        log.info(f"  Inter-test delay: {inter_test_delay}s")
        log.info(f"  Queues to test: {QIDS}")
        log.info("="*70)
        
        all_results = []
        
        for qid in QIDS:
            log.info(f"\n\n{'#'*70}")
            log.info(f"TESTING QUEUE {qid} (SLA threshold: {SLA_THRESHOLDS[qid]}ms)")
            log.info(f"{'#'*70}")
            
            qid_results = []
            
            for i in range(iterations):
                log.info(f"\n>>> Iteration {i+1}/{iterations} for qid={qid}")
                
                result = self.run_single_test(
                    qid=qid,
                    poll_interval_ms=poll_interval_ms
                )
                
                if result:
                    all_results.append(result)
                    qid_results.append(result)
                
                # Wait between tests
                if i < iterations - 1:
                    log.info(f"\nWaiting {inter_test_delay}s before next test...")
                    time.sleep(inter_test_delay)
            
            # Summarize results for this queue
            if qid_results:
                first_times = [r['first_change_time_s'] for r in qid_results if r['first_change_time_s'] is not None]
                stable_times = [r['stabilization_time_s'] for r in qid_results if r['stabilization_time_s'] is not None]
                
                log.info(f"\n{'='*60}")
                log.info(f"SUMMARY for qid={qid}:")
                if first_times:
                    log.info(f"  First change detection:")
                    log.info(f"    Mean: {statistics.mean(first_times):.3f}s")
                    log.info(f"    Min:  {min(first_times):.3f}s")
                    log.info(f"    Max:  {max(first_times):.3f}s")
                    if len(first_times) > 1:
                        log.info(f"    Std:  {statistics.stdev(first_times):.3f}s")
                if stable_times:
                    log.info(f"  Stabilization time:")
                    log.info(f"    Mean: {statistics.mean(stable_times):.3f}s")
                    log.info(f"    Min:  {min(stable_times):.3f}s")
                    log.info(f"    Max:  {max(stable_times):.3f}s")
                log.info(f"{'='*60}")
        
        return all_results
    
    def print_final_report(self, results: List[Dict]):
        """Print final summary report with recommendations."""
        log.info("\n\n" + "="*70)
        log.info("FINAL SYNCHRONIZATION DELAY REPORT")
        log.info("="*70)
        
        if not results:
            log.warning("No test results to report!")
            return
        
        # Aggregate statistics
        all_first_times = [r['first_change_time_s'] for r in results if r['first_change_time_s'] is not None]
        all_stable_times = [r['stabilization_time_s'] for r in results if r['stabilization_time_s'] is not None]
        
        log.info("\nOVERALL STATISTICS:")
        log.info("-" * 50)
        
        if all_first_times:
            log.info(f"First Change Detection (across all queues):")
            log.info(f"  Count:  {len(all_first_times)} tests")
            log.info(f"  Mean:   {statistics.mean(all_first_times):.3f}s")
            log.info(f"  Median: {statistics.median(all_first_times):.3f}s")
            log.info(f"  Min:    {min(all_first_times):.3f}s")
            log.info(f"  Max:    {max(all_first_times):.3f}s")
            if len(all_first_times) > 1:
                log.info(f"  Std:    {statistics.stdev(all_first_times):.3f}s")
            
            p95 = np.percentile(all_first_times, 95) if len(all_first_times) >= 2 else max(all_first_times)
            log.info(f"  P95:    {p95:.3f}s")
        
        if all_stable_times:
            log.info(f"\nStabilization Time:")
            log.info(f"  Count:  {len(all_stable_times)} tests")
            log.info(f"  Mean:   {statistics.mean(all_stable_times):.3f}s")
            log.info(f"  Median: {statistics.median(all_stable_times):.3f}s")
            log.info(f"  Min:    {min(all_stable_times):.3f}s")
            log.info(f"  Max:    {max(all_stable_times):.3f}s")
        
        # Recommendations
        log.info("\n" + "="*70)
        log.info("RECOMMENDATIONS FOR rl_agent_4.py")
        log.info("="*70)
        
        if all_first_times:
            # Use P95 of first change time + buffer for DELAY_AFTER_ACTION
            p95_first = np.percentile(all_first_times, 95) if len(all_first_times) >= 2 else max(all_first_times)
            recommended_delay = max(2.0, p95_first * 1.5)  # Add 50% safety margin
            
            mean_first = statistics.mean(all_first_times)
            max_first = max(all_first_times)
            
            log.info(f"\nCurrent settings in rl_agent_4.py:")
            log.info(f"  DELAY_AFTER_ACTION = 2.5  # seconds")
            log.info(f"  WINDOW_SECONDS = 2        # query window")
            log.info(f"  SAFETY_LAG_MS = 500       # query safety lag")
            log.info(f"  COOLDOWN_SECONDS = 1.0    # between actions")
            
            log.info(f"\nMeasured delays:")
            log.info(f"  Mean first change:  {mean_first:.3f}s")
            log.info(f"  Max first change:   {max_first:.3f}s")
            log.info(f"  P95 first change:   {p95_first:.3f}s")
            
            log.info(f"\nRECOMMENDED values:")
            log.info(f"  DELAY_AFTER_ACTION = {recommended_delay:.1f}  # P95 + 50% margin")
            
            # Window should cover stabilization time
            if all_stable_times:
                stable_p95 = np.percentile(all_stable_times, 95) if len(all_stable_times) >= 2 else max(all_stable_times)
                recommended_window = max(2.0, stable_p95 * 0.8)
                log.info(f"  WINDOW_SECONDS = {recommended_window:.1f}      # Based on stabilization")
            
            # Cooldown should be less than first change to allow quick re-evaluation
            recommended_cooldown = max(0.5, mean_first * 0.5)
            log.info(f"  COOLDOWN_SECONDS = {recommended_cooldown:.1f}  # Half of mean delay")
            
            log.info(f"\nTo update rl_agent_4.py, modify these lines around line 94-98:")
            window_val = recommended_window if all_stable_times else 2.0
            log.info(f"  WINDOW_SECONDS = {window_val:.1f}")
            log.info(f"  SAFETY_LAG_MS = 500")
            log.info(f"  COOLDOWN_SECONDS = {recommended_cooldown:.1f}")
            log.info(f"  DELAY_AFTER_ACTION = {recommended_delay:.1f}")
        else:
            log.warning("No changes detected during tests - cannot make recommendations")
            log.warning("This may indicate:")
            log.warning("  1. No traffic is flowing through the tested paths")
            log.warning("  2. Route changes are not affecting measured flows")
            log.warning("  3. InfluxDB is not receiving INT reports")
        
        log.info("\n" + "="*70)
    
    def close(self):
        """Clean up resources."""
        try:
            self.client.close()
        except Exception:
            pass


# =============================================================================
#                               QUICK TEST
# =============================================================================
def quick_latency_test(tester: SyncTester, samples: int = 10):
    """Quick test to measure query latency to InfluxDB."""
    log.info("\n" + "="*60)
    log.info("QUICK LATENCY TEST: Measuring InfluxDB query round-trip time")
    log.info("="*60)
    
    times = []
    for i in range(samples):
        start = time.monotonic()
        snapshot = tester.collect_snapshot(window_seconds=0.5)
        elapsed = (time.monotonic() - start) * 1000  # ms
        times.append(elapsed)
        log.info(f"  Query {i+1}/{samples}: {elapsed:.1f}ms")
    
    log.info(f"\nQuery latency statistics:")
    log.info(f"  Mean:   {statistics.mean(times):.1f}ms")
    log.info(f"  Min:    {min(times):.1f}ms")
    log.info(f"  Max:    {max(times):.1f}ms")
    if len(times) > 1:
        log.info(f"  Std:    {statistics.stdev(times):.1f}ms")
    
    return times


# =============================================================================
#                                 MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Test synchronization delay between route changes and InfluxDB metrics"
    )
    
    # Test parameters
    parser.add_argument('--iterations', type=int, default=2,
                        help='Number of test iterations per queue (default: 2)')
    parser.add_argument('--poll-interval-ms', type=int, default=100,
                        help='Polling interval in milliseconds (default: 100)')
    parser.add_argument('--inter-test-delay', type=float, default=5.0,
                        help='Delay between tests in seconds (default: 5.0)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Just run query latency test, skip route change tests')
    parser.add_argument('--single-queue', type=int, choices=[0, 1, 7], default=None,
                        help='Test only a single queue (0=voice, 1=video, 7=best-effort)')
    
    # InfluxDB settings
    parser.add_argument('--influx-url', default=DEFAULT_INFLUX_URL)
    parser.add_argument('--influx-org', default=DEFAULT_INFLUX_ORG)
    parser.add_argument('--influx-bucket', default=DEFAULT_INFLUX_BUCKET)
    parser.add_argument('--influx-token', default=DEFAULT_INFLUX_TOKEN)
    
    # Output
    parser.add_argument('--output-json', type=str, default=None,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SyncTester(
        bucket=args.influx_bucket,
        token=args.influx_token,
        org=args.influx_org,
        url=args.influx_url
    )
    
    try:
        # Quick query latency test
        quick_latency_test(tester)
        
        if args.quick_test:
            log.info("\nQuick test mode - skipping route change tests")
            return
        
        # Modify QIDS if single queue specified
        global QIDS
        if args.single_queue is not None:
            QIDS = (args.single_queue,)
            log.info(f"\nTesting single queue: {args.single_queue}")
        
        # Run full test
        results = tester.run_full_test(
            iterations=args.iterations,
            poll_interval_ms=args.poll_interval_ms,
            inter_test_delay=args.inter_test_delay
        )
        
        # Print report
        tester.print_final_report(results)
        
        # Save to JSON if requested
        if args.output_json and results:
            # Convert any non-serializable items
            for r in results:
                for key, val in r.items():
                    if isinstance(val, np.floating):
                        r[key] = float(val)
                    elif isinstance(val, np.integer):
                        r[key] = int(val)
            
            with open(args.output_json, 'w') as f:
                json.dump({
                    'test_time': datetime.now().isoformat(),
                    'config': {
                        'iterations': args.iterations,
                        'poll_interval_ms': args.poll_interval_ms,
                    },
                    'results': results
                }, f, indent=2)
            log.info(f"\nResults saved to: {args.output_json}")
    
    except KeyboardInterrupt:
        log.info("\n\nTest interrupted by user")
    
    finally:
        tester.close()


if __name__ == '__main__':
    main()

