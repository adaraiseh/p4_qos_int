"""Low-overhead local telemetry cache for RL observations.

The collector already builds Influx line-protocol records for every INT report.
This module keeps a short in-memory ring of those same records and serves the
query shapes used by ``rl_agent_4.py`` over a Unix domain socket.
"""

import json
import logging
import math
import os
import queue
import socket
import socketserver
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from logging_config import normalize_artifact_permissions


log = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = "/tmp/p4_qos_int_telemetry.sock"
DEFAULT_TRAINING_STATE_PATH = "/tmp/p4_qos_int_training_state.json"
RELEVANT_MEASUREMENTS = {
    "flow_telemetry_seen",
    "flow_latency",
    "q_drop_rate_100ms",
    "tx_utilization",
    "switch_latency",
}


class LocalTelemetryCache:
    """In-memory time-window cache of collector line-protocol records."""

    def __init__(
        self,
        retention_seconds: float = 20.0,
        max_points_per_measurement: int = 120_000,
    ):
        self.retention_ns = int(max(1.0, retention_seconds) * 1_000_000_000)
        self.max_points_per_measurement = max(1000, int(max_points_per_measurement))
        self.scan_slack_ns = 2_000_000_000
        self._points: Dict[str, Deque[Tuple[int, float, Dict[str, str], int]]] = {
            measurement: deque() for measurement in RELEVANT_MEASUREMENTS
        }
        self._lock = threading.RLock()
        self._last_prune_ns = 0
        self._records_seen = 0
        self._records_cached = 0
        self._epoch_id = 0
        self._epoch_started_ns = time.time_ns()

    def add_line_points(self, lines: Iterable[str]) -> None:
        """Cache relevant records from the exact line protocol sent to Influx."""
        now_ns = time.time_ns()
        seen = 0
        points_by_measurement = defaultdict(list)
        cached_count = 0
        for line in lines:
            seen += 1
            parsed = _parse_line_protocol(line)
            if parsed is None:
                continue
            measurement, tags, value, ts_ns = parsed
            if measurement not in RELEVANT_MEASUREMENTS:
                continue
            points_by_measurement[measurement].append((ts_ns, value, tags, now_ns))
            cached_count += 1

        with self._lock:
            for measurement, points in points_by_measurement.items():
                self._points[measurement].extend(points)

            self._records_seen += seen
            self._records_cached += cached_count
            if now_ns - self._last_prune_ns >= 500_000_000:
                self._prune_locked(now_ns)
                self._last_prune_ns = now_ns

    def add_preparsed_points(
        self,
        records: Iterable[Tuple[str, Dict[str, Any], Any, int]],
    ) -> None:
        """Cache records already built by the collector.

        This is the low-CPU path for live training. The collector still writes
        the exact line-protocol strings to the durable spool/Influx path; this
        method avoids parsing those same strings a second time for the in-memory
        cache.
        """
        now_ns = time.time_ns()
        seen = 0
        points_by_measurement = defaultdict(list)
        cached_count = 0
        for measurement, tags, value, ts_ns in records:
            seen += 1
            if measurement not in RELEVANT_MEASUREMENTS:
                continue
            try:
                value = float(value)
                ts_ns = int(ts_ns)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            points_by_measurement[str(measurement)].append((
                ts_ns,
                value,
                {str(key): str(val) for key, val in tags.items()},
                now_ns,
            ))
            cached_count += 1

        with self._lock:
            for measurement, points in points_by_measurement.items():
                self._points[measurement].extend(points)

            self._records_seen += seen
            self._records_cached += cached_count
            if now_ns - self._last_prune_ns >= 500_000_000:
                self._prune_locked(now_ns)
                self._last_prune_ns = now_ns

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        kind = request.get("kind")
        try:
            if kind == "queue_summary":
                return self._queue_summary(request)
            if kind == "queue_metrics":
                return self._queue_metrics(request)
            if kind == "hot_demands":
                return self._hot_demands(request)
            if kind == "switch_metrics":
                return self._switch_metrics(request)
            if kind == "switch_metrics_multi":
                return self._switch_metrics_multi(request)
            if kind == "traffic_count_multi":
                return self._traffic_count_multi(request)
            if kind == "traffic_count":
                return self._traffic_count(request)
            if kind == "flow_coverage":
                return self._flow_coverage(request)
            if kind == "egress_observations":
                return self._egress_observations(request)
            if kind == "freshness":
                return self._freshness(request)
            if kind == "reset_epoch":
                return self._reset_epoch(request)
            if kind == "health":
                return self._health()
            return {"ok": False, "error": f"unknown request kind: {kind}"}
        except Exception as exc:
            log.debug("local telemetry cache request failed", exc_info=True)
            return {"ok": False, "error": str(exc)}

    def _queue_metrics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_ns, stop_ns = _request_range_ns(request)
        qids = {str(int(qid)) for qid in request.get("qids", [0, 1, 7])}
        metric_specs = (
            ("flow_latency", "lat_p95", "lat"),
            ("q_drop_rate_100ms", "drop_p95", "drop"),
            ("tx_utilization", "util_p95", "util"),
        )

        metrics = {
            qid: {"lat_p95": None, "drop_p95": None, "util_p95": None}
            for qid in qids
        }
        received = {
            qid: {"lat": False, "drop": False, "util": False}
            for qid in qids
        }
        counts = {
            qid: {"lat": 0, "drop": 0, "util": 0}
            for qid in qids
        }
        window = _new_window_audit(start_ns, stop_ns)

        with self._lock:
            for measurement, output_key, received_key in metric_specs:
                by_qid: Dict[str, List[float]] = defaultdict(list)
                for ts_ns, value, tags in self._iter_window_locked(
                    measurement,
                    start_ns,
                    stop_ns,
                ):
                    qid = tags.get("queue_id")
                    if qid in qids:
                        by_qid[qid].append(value)
                        _record_window_audit(window, measurement, ts_ns)

                for qid, values in by_qid.items():
                    quantile = _quantile(values, 0.95)
                    if quantile is not None:
                        metrics[qid][output_key] = quantile
                        received[qid][received_key] = True
                        counts[qid][received_key] = len(values)

        return {
            "ok": True,
            "metrics": metrics,
            "metrics_received": received,
            "counts": counts,
            "window": window,
        }

    def _queue_summary(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Return queue metrics and hottest demands from one exact window scan."""
        start_ns, stop_ns = _request_range_ns(request)
        qids = {str(int(qid)) for qid in request.get("qids", [0, 1, 7])}
        metrics = {
            qid: {"lat_p95": None, "drop_p95": None, "util_p95": None}
            for qid in qids
        }
        received = {
            qid: {"lat": False, "drop": False, "util": False}
            for qid in qids
        }
        counts = {
            qid: {"lat": 0, "drop": 0, "util": 0}
            for qid in qids
        }
        grouped_demands: Dict[str, Dict[Tuple[str, str], List[float]]] = {
            qid: {} for qid in qids
        }
        values_by_metric = {
            "lat": defaultdict(list),
            "drop": defaultdict(list),
            "util": defaultdict(list),
        }
        window = _new_window_audit(start_ns, stop_ns)

        with self._lock:
            flow_points = list(
                self._iter_window_locked("flow_latency", start_ns, stop_ns)
            )
            drop_points = list(
                self._iter_window_locked("q_drop_rate_100ms", start_ns, stop_ns)
            )
            util_points = list(
                self._iter_window_locked("tx_utilization", start_ns, stop_ns)
            )

        for ts_ns, value, tags in flow_points:
            qid = tags.get("queue_id")
            if qid not in qids:
                continue
            values_by_metric["lat"][qid].append(value)
            src = tags.get("src_ip")
            dst = tags.get("dst_ip")
            if src and dst:
                bucket = grouped_demands[qid].setdefault((src, dst), [0.0, 0])
                bucket[0] += value
                bucket[1] += 1
            _record_window_audit(window, "flow_latency", ts_ns)

        for ts_ns, value, tags in drop_points:
            qid = tags.get("queue_id")
            if qid in qids:
                values_by_metric["drop"][qid].append(value)
                _record_window_audit(window, "q_drop_rate_100ms", ts_ns)

        for ts_ns, value, tags in util_points:
            qid = tags.get("queue_id")
            if qid in qids:
                values_by_metric["util"][qid].append(value)
                _record_window_audit(window, "tx_utilization", ts_ns)

        output_keys = {
            "lat": "lat_p95",
            "drop": "drop_p95",
            "util": "util_p95",
        }
        for label, by_qid in values_by_metric.items():
            output_key = output_keys[label]
            for qid, values in by_qid.items():
                quantile = _quantile(values, 0.95)
                if quantile is None:
                    continue
                metrics[qid][output_key] = quantile
                received[qid][label] = True
                counts[qid][label] = len(values)

        demands: Dict[str, Dict[str, Any]] = {}
        for qid, groups in grouped_demands.items():
            best_pair = None
            best_mean = -math.inf
            for pair, (total, count) in groups.items():
                if count <= 0:
                    continue
                mean_value = total / count
                if mean_value > best_mean:
                    best_pair = pair
                    best_mean = mean_value
            if best_pair is not None:
                demands[qid] = {
                    "src_ip": best_pair[0],
                    "dst_ip": best_pair[1],
                    "mean_latency": best_mean,
                }

        return {
            "ok": True,
            "metrics": metrics,
            "metrics_received": received,
            "counts": counts,
            "demands": demands,
            "window": window,
        }

    def _freshness(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_ns, stop_ns = _request_range_ns(request)
        qids = {str(int(qid)) for qid in request.get("qids", [0, 1, 7])}
        measurement_labels = {
            "flow_latency": "lat",
            "q_drop_rate_100ms": "drop",
            "tx_utilization": "util",
        }
        required_labels = set(
            str(item) for item in request.get(
                "required_labels",
                ("lat", "drop", "util"),
            )
        )
        now_ns = time.time_ns()
        queues = {
            qid: {
                label: {
                    "window_count": 0,
                    "window_latest_ns": None,
                    "window_latest_age_ms": None,
                    "cache_latest_ns": None,
                    "cache_latest_ingest_ns": None,
                    "cache_latest_age_ms": None,
                    "cache_latest_ingest_age_ms": None,
                }
                for label in measurement_labels.values()
            }
            for qid in qids
        }
        window = _new_window_audit(start_ns, stop_ns)

        with self._lock:
            for measurement, label in measurement_labels.items():
                latest_needed = set(qids)
                scan_cutoff_ns = start_ns - self.scan_slack_ns
                for ts_ns, _value, tags, ingest_ns in reversed(self._points[measurement]):
                    qid = tags.get("queue_id")
                    if qid in qids and qid in latest_needed:
                        stat = queues[qid][label]
                        stat["cache_latest_ns"] = int(ts_ns)
                        stat["cache_latest_ingest_ns"] = int(ingest_ns)
                        stat["cache_latest_age_ms"] = (
                            (now_ns - int(ts_ns)) / 1_000_000.0
                        )
                        stat["cache_latest_ingest_age_ms"] = (
                            (now_ns - int(ingest_ns)) / 1_000_000.0
                        )
                        latest_needed.discard(qid)

                    if ts_ns >= stop_ns or ts_ns < start_ns:
                        if ingest_ns < scan_cutoff_ns and ts_ns < start_ns:
                            break
                        continue

                    if qid in qids:
                        stat = queues[qid][label]
                        stat["window_count"] += 1
                        latest = stat["window_latest_ns"]
                        if latest is None or ts_ns > int(latest):
                            stat["window_latest_ns"] = int(ts_ns)
                            stat["window_latest_age_ms"] = (
                                (now_ns - int(ts_ns)) / 1_000_000.0
                            )
                        _record_window_audit(window, measurement, ts_ns)

                    if ingest_ns < scan_cutoff_ns and ts_ns < start_ns:
                        break

        missing = {}
        for qid, by_label in queues.items():
            absent = [
                label
                for label in sorted(required_labels)
                if by_label.get(label, {}).get("window_count", 0) <= 0
            ]
            if absent:
                missing[qid] = absent

        return {
            "ok": True,
            "epoch_id": self._epoch_id,
            "epoch_started_ns": self._epoch_started_ns,
            "window": window,
            "queues": queues,
            "missing": missing,
            "complete": not missing,
        }

    def _reset_epoch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        clear = bool(request.get("clear", True))
        reason = str(request.get("reason", "unspecified"))
        now_ns = time.time_ns()
        with self._lock:
            previous_sizes = {
                measurement: len(points)
                for measurement, points in self._points.items()
            }
            self._epoch_id += 1
            self._epoch_started_ns = now_ns
            if clear:
                for points in self._points.values():
                    points.clear()
        log.info(
            "Local telemetry cache epoch reset: epoch=%s clear=%s reason=%s "
            "previous_sizes=%s",
            self._epoch_id,
            clear,
            reason,
            previous_sizes,
        )
        return {
            "ok": True,
            "epoch_id": self._epoch_id,
            "epoch_started_ns": self._epoch_started_ns,
            "cleared": clear,
            "previous_sizes": previous_sizes,
        }

    def _hot_demands(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_ns, stop_ns = _request_range_ns(request)
        qids = {str(int(qid)) for qid in request.get("qids", [0, 1, 7])}
        grouped: Dict[str, Dict[Tuple[str, str], List[float]]] = {
            qid: {} for qid in qids
        }
        window = _new_window_audit(start_ns, stop_ns)

        with self._lock:
            for ts_ns, value, tags in self._iter_window_locked(
                "flow_latency",
                start_ns,
                stop_ns,
            ):
                qid = tags.get("queue_id")
                src = tags.get("src_ip")
                dst = tags.get("dst_ip")
                if qid not in qids or not src or not dst:
                    continue
                _record_window_audit(window, "flow_latency", ts_ns)
                bucket = grouped[qid].setdefault((src, dst), [0.0, 0])
                bucket[0] += value
                bucket[1] += 1

        demands: Dict[str, Dict[str, Any]] = {}
        for qid, groups in grouped.items():
            best_pair = None
            best_mean = -math.inf
            for pair, (total, count) in groups.items():
                if count <= 0:
                    continue
                mean_value = total / count
                if mean_value > best_mean:
                    best_pair = pair
                    best_mean = mean_value
            if best_pair is not None:
                demands[qid] = {
                    "src_ip": best_pair[0],
                    "dst_ip": best_pair[1],
                    "mean_latency": best_mean,
                }

        return {"ok": True, "demands": demands, "window": window}

    def _switch_metrics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_ns, stop_ns = _request_range_ns(request)
        qid = str(int(request["qid"]))
        switch_ids = {str(int(sid)) for sid in request.get("switch_ids", [])}
        measurement_fields = {
            "q_drop_rate_100ms": "drop",
            "switch_latency": "lat",
            "tx_utilization": "util",
        }
        results = {
            sid: {"drop": 0.0, "lat": 0.0, "util": 0.0}
            for sid in switch_ids
        }
        window = _new_window_audit(start_ns, stop_ns)

        with self._lock:
            for measurement, field in measurement_fields.items():
                for ts_ns, value, tags in self._iter_window_locked(
                    measurement,
                    start_ns,
                    stop_ns,
                ):
                    if tags.get("queue_id") != qid:
                        continue
                    sid = tags.get("switch_id")
                    if sid in switch_ids:
                        results[sid][field] = max(results[sid][field], value)
                        _record_window_audit(window, measurement, ts_ns)

        return {"ok": True, "switch_metrics": results, "window": window}

    def _switch_metrics_multi(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_ns, stop_ns = _request_range_ns(request)
        queries = request.get("queries", {}) or {}
        qid_to_switches = {
            str(int(qid)): {str(int(sid)) for sid in switch_ids}
            for qid, switch_ids in queries.items()
        }
        all_qids = set(qid_to_switches)
        measurement_fields = {
            "q_drop_rate_100ms": "drop",
            "switch_latency": "lat",
            "tx_utilization": "util",
        }
        results = {
            qid: {
                sid: {"drop": 0.0, "lat": 0.0, "util": 0.0}
                for sid in switch_ids
            }
            for qid, switch_ids in qid_to_switches.items()
        }
        window = _new_window_audit(start_ns, stop_ns)

        with self._lock:
            for measurement, field in measurement_fields.items():
                for ts_ns, value, tags in self._iter_window_locked(
                    measurement,
                    start_ns,
                    stop_ns,
                ):
                    qid = tags.get("queue_id")
                    if qid not in all_qids:
                        continue
                    sid = tags.get("switch_id")
                    if sid in qid_to_switches[qid]:
                        results[qid][sid][field] = max(
                            results[qid][sid][field],
                            value,
                        )
                        _record_window_audit(window, measurement, ts_ns)

        return {"ok": True, "switch_metrics": results, "window": window}

    def _traffic_count(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_ns, stop_ns = _request_range_ns(request)
        qid = str(int(request["qid"]))
        count = 0
        window = _new_window_audit(start_ns, stop_ns)
        with self._lock:
            for ts_ns, _value, tags in self._iter_window_locked(
                "flow_telemetry_seen",
                start_ns,
                stop_ns,
            ):
                if tags.get("queue_id") == qid:
                    count += 1
                    _record_window_audit(window, "flow_telemetry_seen", ts_ns)
        return {"ok": True, "count": count, "window": window}

    def _traffic_count_multi(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_ns, stop_ns = _request_range_ns(request)
        qids = {str(int(qid)) for qid in request.get("qids", [0, 1, 7])}
        counts = {qid: 0 for qid in qids}
        window = _new_window_audit(start_ns, stop_ns)
        with self._lock:
            for ts_ns, _value, tags in self._iter_window_locked(
                "flow_telemetry_seen",
                start_ns,
                stop_ns,
            ):
                qid = tags.get("queue_id")
                if qid in qids:
                    counts[qid] += 1
                    _record_window_audit(window, "flow_telemetry_seen", ts_ns)
        return {"ok": True, "counts": counts, "window": window}

    def _flow_coverage(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_ns, stop_ns = _request_range_ns(request)
        qids = {str(int(qid)) for qid in request.get("qids", [0, 1, 7])}
        observed = {qid: set() for qid in qids}
        window = _new_window_audit(start_ns, stop_ns)

        with self._lock:
            for ts_ns, _value, tags in self._iter_window_locked(
                "flow_telemetry_seen",
                start_ns,
                stop_ns,
            ):
                qid = tags.get("queue_id")
                flow_id = tags.get("flow_id")
                if qid in qids and flow_id is not None:
                    observed[qid].add(str(flow_id))
                    _record_window_audit(window, "flow_telemetry_seen", ts_ns)

        return {
            "ok": True,
            "coverage_measurement": "flow_telemetry_seen",
            "observed": {
                qid: sorted(flow_ids, key=lambda item: int(item))
                for qid, flow_ids in observed.items()
            },
            "window": window,
        }

    def _egress_observations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_ns, stop_ns = _request_range_ns(request)
        qids = {str(int(qid)) for qid in request.get("qids", [0, 1, 7])}
        top_n = max(1, int(request.get("top_n", 10)))
        window = _new_window_audit(start_ns, stop_ns)
        report_edges: Dict[Tuple[int, str, str], set] = defaultdict(set)
        flow_signatures: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        egress_values: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        egress_flows: Dict[Tuple[str, str, str], set] = defaultdict(set)

        with self._lock:
            for ts_ns, value, tags in self._iter_window_locked(
                "tx_utilization",
                start_ns,
                stop_ns,
            ):
                qid = tags.get("queue_id")
                flow_id = tags.get("flow_id")
                switch_id = tags.get("switch_id")
                egress_port = tags.get("egress_port")
                if (
                    qid not in qids
                    or flow_id is None
                    or switch_id is None
                    or egress_port is None
                ):
                    continue
                edge = f"{switch_id}:{egress_port}"
                report_edges[(ts_ns, str(flow_id), qid)].add(edge)
                egress_key = (str(switch_id), str(egress_port), qid)
                egress_values[egress_key].append(float(value))
                egress_flows[egress_key].add(str(flow_id))
                _record_window_audit(window, "tx_utilization", ts_ns)

        for (_ts_ns, flow_id, qid), edges in report_edges.items():
            if not edges:
                continue
            signature = "|".join(sorted(edges))
            flow_signatures[flow_id][qid][signature] += 1

        flows = {}
        for flow_id, by_qid in flow_signatures.items():
            flows[flow_id] = {}
            for qid, counter in by_qid.items():
                total = sum(counter.values())
                signatures = [
                    {
                        "signature": signature,
                        "edges": signature.split("|") if signature else [],
                        "count": count,
                        "fraction": count / max(1, total),
                    }
                    for signature, count in counter.most_common()
                ]
                flows[flow_id][qid] = {
                    "report_count": total,
                    "unique_signatures": len(counter),
                    "signatures": signatures,
                }

        top_egresses = []
        for (switch_id, egress_port, qid), values in egress_values.items():
            top_egresses.append(
                {
                    "switch_id": switch_id,
                    "egress_port": egress_port,
                    "queue_id": qid,
                    "count": len(values),
                    "flow_count": len(egress_flows[(switch_id, egress_port, qid)]),
                    "mean_util": sum(values) / max(1, len(values)),
                    "p95_util": _quantile(values, 0.95) or 0.0,
                    "max_util": max(values) if values else 0.0,
                }
            )
        top_egresses.sort(
            key=lambda item: (
                item["p95_util"],
                item["mean_util"],
                item["count"],
            ),
            reverse=True,
        )

        return {
            "ok": True,
            "flows": flows,
            "top_egresses": top_egresses[:top_n],
            "window": window,
        }

    def _health(self) -> Dict[str, Any]:
        return {"ok": True, **self.stats()}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            sizes = {name: len(points) for name, points in self._points.items()}
            latest = {}
            for name, points in self._points.items():
                latest[name] = max((point[0] for point in points), default=None)
            return {
                "sizes": sizes,
                "latest_ns": latest,
                "epoch_id": self._epoch_id,
                "epoch_started_ns": self._epoch_started_ns,
                "records_seen": self._records_seen,
                "records_cached": self._records_cached,
                "retention_seconds": self.retention_ns / 1_000_000_000,
            }

    def _prune_locked(self, now_ns: int) -> None:
        cutoff = now_ns - self.retention_ns
        for points in self._points.values():
            while points and points[0][0] < cutoff:
                points.popleft()
            overflow = len(points) - self.max_points_per_measurement
            for _ in range(max(0, overflow)):
                points.popleft()

    def _iter_window_locked(self, measurement: str, start_ns: int, stop_ns: int):
        """Yield recent points in [start, stop), newest first.

        Collector timestamps are generated from system time immediately before
        caching/writing, so append order tracks timestamp order closely. Walking
        backward lets 1-second RL queries avoid scanning the full retention ring.
        """
        scan_cutoff_ns = start_ns - self.scan_slack_ns
        for ts_ns, value, tags, ingest_ns in reversed(self._points[measurement]):
            if ingest_ns < scan_cutoff_ns and ts_ns < start_ns:
                break
            if ts_ns >= stop_ns or ts_ns < start_ns:
                continue
            yield ts_ns, value, tags


class _TelemetryRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(1_000_000)
        if not raw:
            return
        try:
            request = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            response = {"ok": False, "error": f"invalid json: {exc}"}
        else:
            response = self.server.cache.handle_request(request)
        payload = json.dumps(response, separators=(",", ":")).encode("utf-8")
        try:
            self.wfile.write(payload + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            log.debug("local telemetry client disconnected before response write")


class _TelemetryUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: str, cache: LocalTelemetryCache):
        self.cache = cache
        super().__init__(socket_path, _TelemetryRequestHandler)


class LocalTelemetryCacheServer:
    """Background Unix-socket server for a ``LocalTelemetryCache``."""

    def __init__(self, cache: LocalTelemetryCache, socket_path: str):
        self.cache = cache
        self.socket_path = socket_path
        self._server: Optional[_TelemetryUnixServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return
        _unlink_stale_socket(self.socket_path)
        self._server = _TelemetryUnixServer(self.socket_path, self.cache)
        try:
            os.chmod(self.socket_path, 0o666)
        except OSError:
            pass
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="local-telemetry-cache",
            daemon=True,
        )
        self._thread.start()
        log.info("Local telemetry cache listening on %s", self.socket_path)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        _unlink_stale_socket(self.socket_path)


class LocalTelemetryClient:
    """Small blocking client used by the RL agent."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH, timeout: float = 1.0):
        self.socket_path = socket_path
        self.timeout = float(timeout)

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)
            sock.sendall(data)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
        if not chunks:
            raise RuntimeError("local telemetry cache returned no data")
        raw = b"".join(chunks).split(b"\n", 1)[0]
        response = json.loads(raw.decode("utf-8"))
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "local telemetry cache error"))
        return response


def _safe_path_component(value: Optional[str], fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in text
    )
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


@dataclass
class _SpoolTarget:
    run_id: str
    run_dir: Path
    collector_dir: Path
    spool_dir: Path
    path: Path
    manifest_path: Path
    segment_index: int
    segment_start_step: Optional[int]
    segment_end_step: Optional[int]
    current_step: Optional[int]
    state: Dict[str, Any]


class LineProtocolSpoolWriter:
    """Background writer for durable local INT telemetry artifacts.

    The spool stores the exact Influx line-protocol records produced by the
    collector. If the disk writer falls behind, ``write_lines`` applies
    backpressure instead of dropping records; that is the only way to make the
    artifact complete when Influx is disabled. When attached to a training
    state file, records are split into reusable run-scoped line-protocol files
    by training step range.
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str = "int_metrics",
        flush_interval_seconds: float = 1.0,
        max_queue_batches: int = 8192,
        run_state_path: Optional[str] = None,
        external_artifact_root: Optional[str] = None,
        split_every_steps: int = 10000,
    ):
        started_at = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.prefix = prefix
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        normalize_artifact_permissions(self.output_dir, dir_mode=0o775)
        self.external_artifact_root = (
            Path(external_artifact_root) if external_artifact_root else None
        )
        self.run_state_path = Path(run_state_path) if run_state_path else None
        self._dynamic_targets = bool(self.run_state_path or self.external_artifact_root)
        self.split_every_steps = max(0, int(split_every_steps or 0))
        self.path = self.output_dir / f"{started_at}_{prefix}.lp"
        self.manifest_path = self.output_dir / f"{started_at}_{prefix}.manifest.json"
        self.flush_interval_seconds = max(0.1, float(flush_interval_seconds))
        self._queue: "queue.Queue[Optional[List[str]]]" = queue.Queue(
            maxsize=max(1, int(max_queue_batches))
        )
        self._thread = threading.Thread(
            target=self._run,
            name="line-protocol-spool",
            daemon=True,
        )
        self._stop_requested = False
        self._error: Optional[str] = None
        self.records_written = 0
        self.batches_written = 0
        self.started_at = started_at
        self._state_cache: Dict[str, Any] = {}
        self._state_mtime_ns: Optional[int] = None
        self._active_target: Optional[_SpoolTarget] = None
        self._active_records_written = 0
        self._active_batches_written = 0
        self._last_index_event: Optional[Tuple[str, str]] = None
        self._thread.start()
        self._write_manifest("running")
        log.info("Local telemetry spool writing %s", self.path)

    def write_lines(self, lines: Iterable[str]) -> None:
        if self._error:
            raise RuntimeError(self._error)
        batch = lines if isinstance(lines, list) else list(lines)
        if not batch:
            return
        self._queue.put(batch)

    def flush(self) -> None:
        self._queue.join()

    def stats(self) -> Dict[str, Any]:
        target = self._active_target
        return {
            "path": str(self.path),
            "run_id": target.run_id if target else None,
            "segment_index": target.segment_index if target else None,
            "segment_start_step": (
                target.segment_start_step if target else None
            ),
            "segment_end_step": target.segment_end_step if target else None,
            "split_every_steps": self.split_every_steps,
            "records_written": self.records_written,
            "batches_written": self.batches_written,
            "queued_batches": self._queue.qsize(),
            "error": self._error,
        }

    def close(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        self._queue.put(None)
        self._thread.join(timeout=10.0)
        self._write_manifest("closed")

    def _run(self) -> None:
        last_flush = time.monotonic()
        handle = None
        try:
            while True:
                batch = self._queue.get()
                try:
                    if batch is None:
                        if handle is not None:
                            handle.flush()
                            os.fsync(handle.fileno())
                            handle.close()
                            self._write_manifest("closed")
                        return

                    target = self._resolve_target()
                    if (
                        self._active_target is None
                        or target.path != self._active_target.path
                    ):
                        if handle is not None:
                            handle.flush()
                            os.fsync(handle.fileno())
                            handle.close()
                            self._write_manifest("rotated")
                        self._activate_target(target)
                        handle = self.path.open("a", buffering=1024 * 1024)
                        normalize_artifact_permissions(self.path, file_mode=0o664)
                        self._write_manifest("running")
                    else:
                        self._active_target = target

                    handle.write("\n".join(batch))
                    handle.write("\n")
                    self.records_written += len(batch)
                    self.batches_written += 1
                    self._active_records_written += len(batch)
                    self._active_batches_written += 1
                    now = time.monotonic()
                    if now - last_flush >= self.flush_interval_seconds:
                        handle.flush()
                        last_flush = now
                        self._write_manifest("running")
                finally:
                    self._queue.task_done()
        except Exception as exc:
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
            self._error = f"local telemetry spool failed: {exc}"
            log.error(self._error)

    def _load_run_state(self) -> Dict[str, Any]:
        if self.run_state_path is None:
            return {}
        try:
            stat = self.run_state_path.stat()
            if stat.st_mtime_ns == self._state_mtime_ns:
                return self._state_cache
            payload = json.loads(self.run_state_path.read_text())
            if not isinstance(payload, dict):
                payload = {}
            self._state_cache = payload
            self._state_mtime_ns = stat.st_mtime_ns
            return payload
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            log.debug("failed to read training state file", exc_info=True)
            return self._state_cache

    def _resolve_target(self) -> _SpoolTarget:
        if not self._dynamic_targets:
            return _SpoolTarget(
                run_id="standalone",
                run_dir=self.output_dir,
                collector_dir=self.output_dir,
                spool_dir=self.output_dir,
                path=self.path,
                manifest_path=self.manifest_path,
                segment_index=0,
                segment_start_step=None,
                segment_end_step=None,
                current_step=None,
                state={},
            )

        state = self._load_run_state()
        run_id = _safe_path_component(
            state.get("run_id"),
            f"unassigned-{self.started_at}",
        )
        current_step = self._state_int(state.get("total_step"))
        if self.split_every_steps > 0 and current_step and current_step > 0:
            segment_index = (current_step - 1) // self.split_every_steps
            segment_start_step = segment_index * self.split_every_steps + 1
            segment_end_step = (segment_index + 1) * self.split_every_steps
        else:
            segment_index = 0
            segment_start_step = None
            segment_end_step = None

        run_dir = self._state_path(state.get("run_dir"))
        if run_dir is None:
            if self.external_artifact_root is not None:
                run_dir = self.external_artifact_root / run_id
            else:
                run_dir = self.output_dir

        collector_dir = self._state_path(state.get("collector_dir"))
        if collector_dir is None:
            collector_dir = run_dir / "collector"
        spool_dir = collector_dir / "spool"

        if segment_start_step is not None and segment_end_step is not None:
            step_range = f"steps_{segment_start_step:09d}-{segment_end_step:09d}"
        else:
            step_range = "steps_unassigned"
        filename = (
            f"{run_id}_{self.started_at}_{self.prefix}_"
            f"{step_range}_part{segment_index:04d}.lp"
        )
        path = spool_dir / filename
        manifest_path = path.with_suffix(".manifest.json")
        return _SpoolTarget(
            run_id=run_id,
            run_dir=run_dir,
            collector_dir=collector_dir,
            spool_dir=spool_dir,
            path=path,
            manifest_path=manifest_path,
            segment_index=segment_index,
            segment_start_step=segment_start_step,
            segment_end_step=segment_end_step,
            current_step=current_step,
            state=state,
        )

    @staticmethod
    def _state_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _state_path(value: Any) -> Optional[Path]:
        if not value:
            return None
        try:
            return Path(str(value))
        except TypeError:
            return None

    def _activate_target(self, target: _SpoolTarget) -> None:
        target.spool_dir.mkdir(parents=True, exist_ok=True)
        normalize_artifact_permissions(target.run_dir, dir_mode=0o775)
        normalize_artifact_permissions(target.collector_dir, dir_mode=0o775)
        normalize_artifact_permissions(target.spool_dir, dir_mode=0o775)
        self._active_target = target
        self._active_records_written = 0
        self._active_batches_written = 0
        self.path = target.path
        self.manifest_path = target.manifest_path
        log.info("Local telemetry spool segment writing %s", self.path)

    def _write_manifest(self, status: str) -> None:
        target = self._active_target
        record_path = self.path
        manifest_path = self.manifest_path
        payload = {
            "schema_version": 2,
            "status": status,
            "started_at": self.started_at,
            "updated_at_utc": datetime.utcnow().isoformat() + "Z",
            "format": "influx_line_protocol",
            "record_path": str(record_path),
            "records_written": (
                self._active_records_written if target else self.records_written
            ),
            "batches_written": (
                self._active_batches_written if target else self.batches_written
            ),
            "session_records_written": self.records_written,
            "session_batches_written": self.batches_written,
            "flush_interval_seconds": self.flush_interval_seconds,
            "split_every_steps": self.split_every_steps,
            "error": self._error,
            "description": (
                "Exact collector line-protocol telemetry records. Each line can "
                "be replayed into InfluxDB or parsed locally for paper analysis."
            ),
            "reuse": {
                "line_protocol": True,
                "append_order": "collector write order within this segment",
                "suggested_analysis_key": "training_run_id + step range + measurement tags",
            },
        }
        if target is not None:
            payload.update({
                "training_run_id": target.run_id,
                "run_dir": str(target.run_dir),
                "collector_dir": str(target.collector_dir),
                "spool_dir": str(target.spool_dir),
                "segment_index": target.segment_index,
                "segment_start_step": target.segment_start_step,
                "segment_end_step": target.segment_end_step,
                "latest_training_step": target.current_step,
                "training_state_path": (
                    str(self.run_state_path) if self.run_state_path else None
                ),
                "training_state": target.state,
            })
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            normalize_artifact_permissions(manifest_path, file_mode=0o664)
            self._append_spool_index(status, payload)
        except OSError:
            log.debug("failed to write local telemetry spool manifest", exc_info=True)

    def _append_spool_index(self, status: str, payload: Dict[str, Any]) -> None:
        target = self._active_target
        if target is None or status == "running":
            return
        event_key = (str(target.path), status)
        if event_key == self._last_index_event:
            return
        self._last_index_event = event_key
        index_path = target.collector_dir / "spool_index.jsonl"
        row = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "status": status,
            "training_run_id": target.run_id,
            "segment_index": target.segment_index,
            "segment_start_step": target.segment_start_step,
            "segment_end_step": target.segment_end_step,
            "record_path": payload["record_path"],
            "manifest_path": str(target.manifest_path),
            "records_written": payload["records_written"],
            "session_started_at": self.started_at,
            "format": payload["format"],
        }
        try:
            with index_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            normalize_artifact_permissions(index_path, file_mode=0o664)
        except OSError:
            log.debug("failed to append local telemetry spool index", exc_info=True)


def iso_to_ns(value: str) -> int:
    """Parse the UTC ISO strings produced by rl_agent_4._time_window()."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


def _request_range_ns(request: Dict[str, Any]) -> Tuple[int, int]:
    if "start_ns" in request and "stop_ns" in request:
        return int(request["start_ns"]), int(request["stop_ns"])
    return iso_to_ns(str(request["start"])), iso_to_ns(str(request["stop"]))


def _new_window_audit(start_ns: int, stop_ns: int) -> Dict[str, Any]:
    return {
        "start_ns": int(start_ns),
        "stop_ns": int(stop_ns),
        "count": 0,
        "min_ns": None,
        "max_ns": None,
        "within_window": True,
        "by_measurement": {},
    }


def _record_window_audit(window: Dict[str, Any], measurement: str, ts_ns: int) -> None:
    start_ns = int(window["start_ns"])
    stop_ns = int(window["stop_ns"])
    _record_window_stat(window, ts_ns, start_ns, stop_ns)
    by_measurement = window["by_measurement"]
    stat = by_measurement.get(measurement)
    if stat is None:
        stat = {
            "count": 0,
            "min_ns": None,
            "max_ns": None,
            "within_window": True,
        }
        by_measurement[measurement] = stat
    _record_window_stat(stat, ts_ns, start_ns, stop_ns)


def _record_window_stat(stat: Dict[str, Any], ts_ns: int, start_ns: int, stop_ns: int) -> None:
    ts_ns = int(ts_ns)
    stat["count"] += 1
    stat["min_ns"] = ts_ns if stat["min_ns"] is None else min(stat["min_ns"], ts_ns)
    stat["max_ns"] = ts_ns if stat["max_ns"] is None else max(stat["max_ns"], ts_ns)
    if ts_ns < start_ns or ts_ns >= stop_ns:
        stat["within_window"] = False


def _parse_line_protocol(line: str) -> Optional[Tuple[str, Dict[str, str], float, int]]:
    try:
        head, fields, ts_text = line.rsplit(" ", 2)
    except ValueError:
        return None

    if "," in head:
        measurement, tag_text = head.split(",", 1)
    else:
        measurement, tag_text = head, ""
    if measurement not in RELEVANT_MEASUREMENTS:
        return None

    value = None
    for field in fields.split(","):
        if not field.startswith("value="):
            continue
        value_text = field.split("=", 1)[1]
        if value_text.endswith(("i", "u")):
            value_text = value_text[:-1]
        value = float(value_text)
        break
    if value is None or not math.isfinite(value):
        return None

    tags: Dict[str, str] = {}
    if tag_text:
        for item in tag_text.split(","):
            if "=" not in item:
                continue
            key, tag_value = item.split("=", 1)
            tags[key] = tag_value
    return measurement, tags, value, int(ts_text)


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    fraction = rank - low
    return float(ordered[low] * (1.0 - fraction) + ordered[high] * fraction)


def _unlink_stale_socket(socket_path: str) -> None:
    try:
        if os.path.exists(socket_path) and not os.path.isdir(socket_path):
            os.unlink(socket_path)
    except OSError:
        pass
