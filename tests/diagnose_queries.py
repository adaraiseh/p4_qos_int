#!/usr/bin/env python3
"""
diagnose_queries.py - Diagnose InfluxDB query issues in real-time.

This script queries InfluxDB with timing to identify:
- Query latency issues
- Missing data per queue
- Data point counts vs expectations
- Time window alignment issues

Usage:
    python3 diagnose_queries.py [--loops N]

Author: Research Team (Diagnostic Tool)
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, Tuple

from influxdb_client import InfluxDBClient

# Configuration (match rl_agent_4.py)
INFLUX_URL = "http://192.168.56.1:8086"
INFLUX_ORG = "Research"
INFLUX_BUCKET = "INT"
WINDOW_SECONDS = 1.0
SAFETY_LAG_MS = 0
QIDS = (0, 1, 7)

def get_time_window() -> Tuple[str, str]:
    """Get time window matching rl_agent_4.py logic."""
    stop_dt = datetime.utcnow() - timedelta(milliseconds=SAFETY_LAG_MS)
    start_dt = stop_dt - timedelta(seconds=WINDOW_SECONDS)
    return start_dt.isoformat() + 'Z', stop_dt.isoformat() + 'Z'

def query_metrics_counts(query_api, bucket: str, org: str, start: str, stop: str) -> Dict[int, Dict[str, int]]:
    """Query data point counts per queue per measurement."""
    flux = f'''
    from(bucket:"{bucket}")
        |> range(start:{start}, stop:{stop})
        |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
        |> filter(fn: (r) => r._measurement == "flow_latency" or r._measurement == "q_drop_rate_100ms" or r._measurement == "tx_utilization")
        |> group(columns:["queue_id", "_measurement"])
        |> count()
    '''

    counts = {qid: {'flow_latency': 0, 'q_drop_rate_100ms': 0, 'tx_utilization': 0} for qid in QIDS}

    start_time = time.monotonic()
    try:
        tables = query_api.query(org=org, query=flux)
        elapsed_ms = (time.monotonic() - start_time) * 1000

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

        return counts, elapsed_ms, None
    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return counts, elapsed_ms, str(e)

def query_p95_metrics(query_api, bucket: str, org: str, start: str, stop: str) -> Dict[int, Dict[str, float]]:
    """Query P95 aggregated metrics per queue."""
    flux = f'''
    base = from(bucket:"{bucket}")
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

    metrics = {qid: {'lat_p95': None, 'drop_p95': None, 'util_p95': None} for qid in QIDS}

    start_time = time.monotonic()
    try:
        tables = query_api.query(org=org, query=flux)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        for table in tables or []:
            for record in table.records:
                try:
                    qid = int(record.values.get('queue_id', -1))
                    if qid not in QIDS:
                        continue
                    measurement = record.get_measurement()
                    value = record.get_value()
                    if value is not None:
                        metrics[qid][measurement] = float(value)
                except (ValueError, TypeError):
                    continue

        return metrics, elapsed_ms, None
    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return metrics, elapsed_ms, str(e)

def main():
    parser = argparse.ArgumentParser(description='Diagnose InfluxDB query issues')
    parser.add_argument('--loops', type=int, default=10, help='Number of diagnostic loops (0=infinite)')
    parser.add_argument('--interval', type=float, default=2.0, help='Interval between checks (seconds)')
    args = parser.parse_args()

    token = os.environ.get('INFLUX_TOKEN')
    if not token:
        print("ERROR: INFLUX_TOKEN environment variable not set")
        sys.exit(1)

    client = InfluxDBClient(url=INFLUX_URL, token=token, org=INFLUX_ORG, timeout=2000)
    query_api = client.query_api()

    print(f"=== InfluxDB Query Diagnostics ===")
    print(f"URL: {INFLUX_URL}")
    print(f"Bucket: {INFLUX_BUCKET}")
    print(f"Window: {WINDOW_SECONDS}s, Safety lag: {SAFETY_LAG_MS}ms")
    print(f"Queues: {QIDS}")
    print("=" * 60)

    loop = 0
    consecutive_failures = 0
    last_failure = None

    try:
        while args.loops == 0 or loop < args.loops:
            loop += 1
            start, stop = get_time_window()

            # Query counts
            counts, counts_time, counts_error = query_metrics_counts(
                query_api, INFLUX_BUCKET, INFLUX_ORG, start, stop
            )

            # Query P95 metrics
            metrics, metrics_time, metrics_error = query_p95_metrics(
                query_api, INFLUX_BUCKET, INFLUX_ORG, start, stop
            )

            # Analyze results
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            total_time = counts_time + metrics_time

            # Check for issues
            issues = []
            for qid in QIDS:
                c = counts[qid]
                m = metrics[qid]
                if c['flow_latency'] == 0:
                    issues.append(f"Q{qid}:no_latency")
                if c['q_drop_rate_100ms'] == 0:
                    issues.append(f"Q{qid}:no_drop")
                if c['tx_utilization'] == 0:
                    issues.append(f"Q{qid}:no_util")
                if m['lat_p95'] is None:
                    issues.append(f"Q{qid}:null_lat_p95")
                if m['drop_p95'] is None:
                    issues.append(f"Q{qid}:null_drop_p95")
                if m['util_p95'] is None:
                    issues.append(f"Q{qid}:null_util_p95")

            # Print status
            if issues:
                consecutive_failures += 1
                status = "ISSUE"
                print(f"\n[{timestamp}] {status} (#{consecutive_failures}) - {total_time:.0f}ms")
                print(f"  Window: {start} to {stop}")
                print(f"  Issues: {', '.join(issues)}")
                for qid in QIDS:
                    c = counts[qid]
                    m = metrics[qid]
                    lat_str = f"{m['lat_p95']:.2f}" if m['lat_p95'] is not None else 'N/A'
                    drop_str = f"{m['drop_p95']:.2f}" if m['drop_p95'] is not None else 'N/A'
                    util_str = f"{m['util_p95']:.1f}" if m['util_p95'] is not None else 'N/A'
                    print(f"  Q{qid}: lat={c['flow_latency']} drop={c['q_drop_rate_100ms']} util={c['tx_utilization']} | "
                          f"p95: lat={lat_str} drop={drop_str} util={util_str}")
                last_failure = timestamp
            else:
                if consecutive_failures > 0:
                    print(f"\n[{timestamp}] RECOVERED after {consecutive_failures} failures")
                consecutive_failures = 0
                status = "OK"
                q0_cnt = sum(counts[0].values())
                q1_cnt = sum(counts[1].values())
                q7_cnt = sum(counts[7].values())
                print(f"[{timestamp}] {status} - {total_time:.0f}ms | Q0:{q0_cnt} Q1:{q1_cnt} Q7:{q7_cnt}")

            if counts_error:
                print(f"  Count query error: {counts_error}")
            if metrics_error:
                print(f"  Metrics query error: {metrics_error}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted after {loop} loops")
        if last_failure:
            print(f"Last failure at: {last_failure}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
