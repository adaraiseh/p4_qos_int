import time
import tempfile
import unittest

from report_collector.local_telemetry_cache import (
    LineProtocolSpoolWriter,
    LocalTelemetryCache,
)


class LocalTelemetryCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = LocalTelemetryCache(retention_seconds=10, max_points_per_measurement=1000)
        self.base_ns = time.time_ns()

    def _line(self, measurement, tags, value, offset_ms=0):
        tag_text = ",".join(f"{key}={val}" for key, val in tags.items())
        return f"{measurement},{tag_text} value={value} {self.base_ns + offset_ms * 1_000_000}"

    def test_queue_metrics_match_rl_query_shape(self):
        tags = {"dst_ip": "10.0.0.2", "flow_id": "10", "queue_id": "0", "src_ip": "10.0.0.1"}
        points = [
            self._line("flow_latency", tags, 10.0, 1),
            self._line("flow_latency", tags, 20.0, 2),
            self._line("flow_latency", tags, 30.0, 3),
            self._line("q_drop_rate_100ms", {**tags, "switch_id": "13", "egress_port": "1"}, 2.0, 3),
            self._line("tx_utilization", {**tags, "switch_id": "13", "egress_port": "1"}, 50.0, 3),
        ]
        self.cache.add_line_points(points)

        response = self.cache.handle_request({
            "kind": "queue_metrics",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "qids": [0],
        })

        self.assertTrue(response["ok"])
        self.assertAlmostEqual(response["metrics"]["0"]["lat_p95"], 29.0)
        self.assertEqual(response["metrics"]["0"]["drop_p95"], 2.0)
        self.assertEqual(response["metrics"]["0"]["util_p95"], 50.0)
        self.assertEqual(response["metrics_received"]["0"], {"lat": True, "drop": True, "util": True})
        self.assertEqual(response["window"]["count"], 5)
        self.assertEqual(response["window"]["min_ns"], self.base_ns + 1_000_000)
        self.assertEqual(response["window"]["max_ns"], self.base_ns + 3_000_000)
        self.assertTrue(response["window"]["within_window"])
        self.assertEqual(response["window"]["by_measurement"]["flow_latency"]["count"], 3)

    def test_window_audit_excludes_records_outside_requested_range(self):
        tags = {"dst_ip": "10.0.0.2", "flow_id": "10", "queue_id": "0", "src_ip": "10.0.0.1"}
        self.cache.add_line_points([
            self._line("flow_latency", tags, 5.0, -1),
            self._line("flow_latency", tags, 10.0, 1),
            self._line("flow_latency", tags, 20.0, 2),
            self._line("flow_latency", tags, 30.0, 10),
        ])

        response = self.cache.handle_request({
            "kind": "queue_metrics",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "qids": [0],
        })

        self.assertTrue(response["ok"])
        self.assertEqual(response["counts"]["0"]["lat"], 2)
        self.assertEqual(response["window"]["count"], 2)
        self.assertEqual(response["window"]["min_ns"], self.base_ns + 1_000_000)
        self.assertEqual(response["window"]["max_ns"], self.base_ns + 2_000_000)
        self.assertTrue(response["window"]["within_window"])

    def test_window_scan_tolerates_late_older_record(self):
        tags = {"dst_ip": "10.0.0.2", "flow_id": "10", "queue_id": "0", "src_ip": "10.0.0.1"}
        self.cache.add_line_points([
            self._line("flow_latency", tags, 10.0, 1),
            self._line("flow_latency", tags, 5.0, -1),
        ])

        response = self.cache.handle_request({
            "kind": "queue_metrics",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "qids": [0],
        })

        self.assertTrue(response["ok"])
        self.assertEqual(response["counts"]["0"]["lat"], 1)
        self.assertEqual(response["window"]["min_ns"], self.base_ns + 1_000_000)
        self.assertEqual(response["window"]["max_ns"], self.base_ns + 1_000_000)
        self.assertTrue(response["window"]["within_window"])

    def test_hot_demands_uses_highest_mean_flow_latency_per_queue(self):
        common = {"flow_id": "10", "queue_id": "1"}
        self.cache.add_line_points([
            self._line("flow_latency", {**common, "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2"}, 10.0, 1),
            self._line("flow_latency", {**common, "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2"}, 20.0, 2),
            self._line("flow_latency", {**common, "src_ip": "10.0.0.3", "dst_ip": "10.0.0.4"}, 40.0, 2),
        ])

        response = self.cache.handle_request({
            "kind": "hot_demands",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "qids": [1],
        })

        self.assertTrue(response["ok"])
        self.assertEqual(response["demands"]["1"]["src_ip"], "10.0.0.3")
        self.assertEqual(response["demands"]["1"]["dst_ip"], "10.0.0.4")

    def test_queue_summary_combines_exact_window_metrics_and_demands(self):
        q0 = {"flow_id": "10", "queue_id": "0"}
        self.cache.add_line_points([
            self._line("flow_latency", {**q0, "src_ip": "10.0.0.9", "dst_ip": "10.0.0.10"}, 500.0, -5),
            self._line("flow_latency", {**q0, "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2"}, 10.0, 1),
            self._line("flow_latency", {**q0, "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2"}, 20.0, 2),
            self._line("flow_latency", {**q0, "src_ip": "10.0.0.3", "dst_ip": "10.0.0.4"}, 40.0, 3),
            self._line("q_drop_rate_100ms", {**q0, "switch_id": "13", "egress_port": "1"}, 0.5, 3),
            self._line("tx_utilization", {**q0, "switch_id": "13", "egress_port": "1"}, 72.0, 3),
            self._line("flow_latency", {**q0, "src_ip": "10.0.0.9", "dst_ip": "10.0.0.10"}, 600.0, 20),
        ])

        response = self.cache.handle_request({
            "kind": "queue_summary",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "qids": [0],
        })

        self.assertTrue(response["ok"])
        self.assertEqual(response["counts"]["0"], {"lat": 3, "drop": 1, "util": 1})
        self.assertAlmostEqual(response["metrics"]["0"]["lat_p95"], 38.0)
        self.assertEqual(response["metrics"]["0"]["drop_p95"], 0.5)
        self.assertEqual(response["metrics"]["0"]["util_p95"], 72.0)
        self.assertEqual(response["demands"]["0"]["src_ip"], "10.0.0.3")
        self.assertEqual(response["demands"]["0"]["dst_ip"], "10.0.0.4")
        self.assertEqual(response["window"]["count"], 5)
        self.assertEqual(response["window"]["min_ns"], self.base_ns + 1_000_000)
        self.assertEqual(response["window"]["max_ns"], self.base_ns + 3_000_000)
        self.assertTrue(response["window"]["within_window"])

    def test_switch_metrics_use_per_measurement_max(self):
        tags = {
            "dst_ip": "10.0.0.2",
            "egress_port": "1",
            "flow_id": "10",
            "queue_id": "7",
            "src_ip": "10.0.0.1",
            "switch_id": "21",
        }
        self.cache.add_line_points([
            self._line("switch_latency", tags, 5.0, 1),
            self._line("switch_latency", tags, 8.0, 2),
            self._line("q_drop_rate_100ms", tags, 0.5, 2),
            self._line("tx_utilization", tags, 72.0, 2),
        ])

        response = self.cache.handle_request({
            "kind": "switch_metrics",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "switch_ids": [21],
            "qid": 7,
        })

        self.assertTrue(response["ok"])
        self.assertEqual(response["switch_metrics"]["21"], {"drop": 0.5, "lat": 8.0, "util": 72.0})

    def test_switch_metrics_multi_batches_qids_with_exact_window(self):
        q0s21 = {
            "dst_ip": "10.0.0.2",
            "egress_port": "1",
            "flow_id": "10",
            "queue_id": "0",
            "src_ip": "10.0.0.1",
            "switch_id": "21",
        }
        q1s22 = {**q0s21, "queue_id": "1", "switch_id": "22"}
        self.cache.add_line_points([
            self._line("switch_latency", q0s21, 999.0, -5),
            self._line("switch_latency", q0s21, 5.0, 1),
            self._line("switch_latency", q0s21, 8.0, 2),
            self._line("q_drop_rate_100ms", q0s21, 0.5, 2),
            self._line("tx_utilization", q0s21, 72.0, 2),
            self._line("switch_latency", q1s22, 12.0, 3),
            self._line("q_drop_rate_100ms", q1s22, 1.5, 3),
            self._line("tx_utilization", q1s22, 82.0, 3),
            self._line("switch_latency", q0s21, 1000.0, 20),
        ])

        response = self.cache.handle_request({
            "kind": "switch_metrics_multi",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "queries": {"0": [21], "1": [22]},
        })

        self.assertTrue(response["ok"])
        self.assertEqual(response["switch_metrics"]["0"]["21"], {"drop": 0.5, "lat": 8.0, "util": 72.0})
        self.assertEqual(response["switch_metrics"]["1"]["22"], {"drop": 1.5, "lat": 12.0, "util": 82.0})
        self.assertEqual(response["window"]["count"], 7)
        self.assertEqual(response["window"]["min_ns"], self.base_ns + 1_000_000)
        self.assertEqual(response["window"]["max_ns"], self.base_ns + 3_000_000)
        self.assertTrue(response["window"]["within_window"])

    def test_traffic_count_counts_flow_latency_by_queue(self):
        self.cache.add_line_points([
            self._line("flow_latency", {"queue_id": "0", "src_ip": "a", "dst_ip": "b", "flow_id": "1"}, 1.0, 1),
            self._line("flow_latency", {"queue_id": "0", "src_ip": "a", "dst_ip": "b", "flow_id": "1"}, 1.0, 2),
            self._line("flow_latency", {"queue_id": "1", "src_ip": "a", "dst_ip": "b", "flow_id": "1"}, 1.0, 2),
        ])

        response = self.cache.handle_request({
            "kind": "traffic_count",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "qid": 0,
        })

        self.assertTrue(response["ok"])
        self.assertEqual(response["count"], 2)

    def test_traffic_count_multi_counts_queues_in_one_exact_window(self):
        self.cache.add_line_points([
            self._line("flow_latency", {"queue_id": "0", "src_ip": "a", "dst_ip": "b", "flow_id": "1"}, 1.0, -5),
            self._line("flow_latency", {"queue_id": "0", "src_ip": "a", "dst_ip": "b", "flow_id": "1"}, 1.0, 1),
            self._line("flow_latency", {"queue_id": "0", "src_ip": "a", "dst_ip": "b", "flow_id": "1"}, 1.0, 2),
            self._line("flow_latency", {"queue_id": "1", "src_ip": "a", "dst_ip": "b", "flow_id": "1"}, 1.0, 2),
            self._line("flow_latency", {"queue_id": "7", "src_ip": "a", "dst_ip": "b", "flow_id": "1"}, 1.0, 20),
        ])

        response = self.cache.handle_request({
            "kind": "traffic_count_multi",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "qids": [0, 1, 7],
        })

        self.assertTrue(response["ok"])
        self.assertEqual(response["counts"], {"0": 2, "1": 1, "7": 0})
        self.assertEqual(response["window"]["count"], 3)
        self.assertEqual(response["window"]["min_ns"], self.base_ns + 1_000_000)
        self.assertEqual(response["window"]["max_ns"], self.base_ns + 2_000_000)
        self.assertTrue(response["window"]["within_window"])

    def test_egress_observations_reconstruct_path_and_bottleneck(self):
        common = {
            "dst_ip": "10.0.0.2",
            "flow_id": "44",
            "queue_id": "0",
            "src_ip": "10.0.0.1",
        }
        self.cache.add_line_points([
            self._line(
                "tx_utilization",
                {**common, "switch_id": "21", "egress_port": "3"},
                60.0,
                1,
            ),
            self._line(
                "tx_utilization",
                {**common, "switch_id": "13", "egress_port": "1"},
                90.0,
                1,
            ),
            self._line(
                "tx_utilization",
                {**common, "switch_id": "21", "egress_port": "3"},
                62.0,
                2,
            ),
            self._line(
                "tx_utilization",
                {**common, "switch_id": "13", "egress_port": "1"},
                95.0,
                2,
            ),
        ])

        response = self.cache.handle_request({
            "kind": "egress_observations",
            "start_ns": self.base_ns,
            "stop_ns": self.base_ns + 10_000_000,
            "qids": [0],
            "top_n": 1,
        })

        self.assertTrue(response["ok"])
        flow = response["flows"]["44"]["0"]
        self.assertEqual(flow["report_count"], 2)
        self.assertEqual(flow["unique_signatures"], 1)
        self.assertEqual(
            flow["signatures"][0]["edges"],
            ["13:1", "21:3"],
        )
        self.assertEqual(flow["signatures"][0]["count"], 2)
        self.assertEqual(response["top_egresses"][0]["switch_id"], "13")
        self.assertEqual(response["top_egresses"][0]["egress_port"], "1")
        self.assertAlmostEqual(response["top_egresses"][0]["p95_util"], 94.75)
        self.assertEqual(response["window"]["by_measurement"]["tx_utilization"]["count"], 4)

    def test_line_protocol_spool_persists_exact_records(self):
        lines = [
            self._line("flow_latency", {"queue_id": "0", "src_ip": "a", "dst_ip": "b", "flow_id": "1"}, 1.0, 1),
            self._line("tx_utilization", {"queue_id": "0", "src_ip": "a", "dst_ip": "b", "flow_id": "1", "switch_id": "13", "egress_port": "1"}, 20.0, 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            writer = LineProtocolSpoolWriter(directory, flush_interval_seconds=0.1)
            writer.write_lines(lines)
            writer.flush()
            writer.close()

            self.assertEqual(writer.path.read_text().splitlines(), lines)
            manifest = writer.manifest_path.read_text()
            self.assertIn('"status": "closed"', manifest)
            self.assertIn('"records_written": 2', manifest)


if __name__ == "__main__":
    unittest.main()
