import time
import unittest

from report_collector.collector import Collector, FlowInfo
from report_collector.local_telemetry_cache import LocalTelemetryCache


def _traffic_dst_port(flow_id, qid, base=6000):
    return base + flow_id * 10 + qid


class CapturingSpool:
    def __init__(self):
        self.lines = []

    def write_lines(self, lines):
        self.lines.extend(lines)

    def flush(self):
        return None

    def stats(self):
        return {"records_written": len(self.lines)}


def make_flow(latency_ms, flow_id=10, qid=7):
    flow = FlowInfo()
    flow.src_ip = "10.0.0.1"
    flow.dst_ip = "10.0.0.2"
    flow.src_port = 12345
    flow.dst_port = _traffic_dst_port(flow_id, qid)
    flow.ip_proto = 17
    flow.hop_cnt = 1
    flow.switch_ids = [13]
    flow.l1_ingress_ports = [1]
    flow.l1_egress_ports = [2]
    flow.hop_latencies = [1000]
    flow.queue_ids = [qid]
    flow.queue_occups = [0]
    flow.queue_drops = [0]
    flow.egress_tx_utils = [50.0]
    base_ns = 1_000_000_000
    flow.ingress_tstamps = [base_ns]
    flow.egress_tstamps = [base_ns + int(latency_ms * 1_000_000)]
    return flow


class CollectorLivenessTests(unittest.TestCase):
    def _collector(self):
        cache = LocalTelemetryCache(retention_seconds=10, max_points_per_measurement=1000)
        spool = CapturingSpool()
        collector = Collector(
            None,
            "org",
            "bucket",
            aggregate_enabled=False,
            telemetry_cache=cache,
            telemetry_spool=spool,
            influx_enabled=False,
        )
        return collector, cache, spool

    def test_high_positive_latency_is_cached_as_performance_not_rejected(self):
        collector, cache, spool = self._collector()

        collector.export_influxdb(make_flow(latency_ms=20000.0, flow_id=10, qid=7))

        self.assertTrue(any(line.startswith("flow_telemetry_seen,") for line in spool.lines))
        self.assertTrue(any(line.startswith("flow_latency,") and "value=20000.0 " in line for line in spool.lines))

        response = cache.handle_request({
            "kind": "queue_metrics",
            "start_ns": 0,
            "stop_ns": time.time_ns() + 1_000_000_000,
            "qids": [7],
        })
        self.assertTrue(response["ok"])
        self.assertEqual(response["metrics"]["7"]["lat_p95"], 20000.0)

        coverage = cache.handle_request({
            "kind": "flow_coverage",
            "start_ns": 0,
            "stop_ns": time.time_ns() + 1_000_000_000,
            "qids": [7],
        })
        self.assertTrue(coverage["ok"])
        self.assertEqual(coverage["coverage_measurement"], "flow_telemetry_seen")
        self.assertEqual(coverage["observed"]["7"], ["10"])

    def test_negative_latency_drops_performance_metric_but_keeps_liveness(self):
        collector, cache, spool = self._collector()

        collector.export_influxdb(make_flow(latency_ms=-5.0, flow_id=10, qid=7))

        self.assertTrue(any(line.startswith("flow_telemetry_seen,") for line in spool.lines))
        self.assertFalse(any(line.startswith("flow_latency,") for line in spool.lines))

        coverage = cache.handle_request({
            "kind": "flow_coverage",
            "start_ns": 0,
            "stop_ns": time.time_ns() + 1_000_000_000,
            "qids": [7],
        })
        self.assertTrue(coverage["ok"])
        self.assertEqual(coverage["observed"]["7"], ["10"])

        metrics = cache.handle_request({
            "kind": "queue_metrics",
            "start_ns": 0,
            "stop_ns": time.time_ns() + 1_000_000_000,
            "qids": [7],
        })
        self.assertTrue(metrics["ok"])
        self.assertFalse(metrics["metrics_received"]["7"]["lat"])


if __name__ == "__main__":
    unittest.main()
