import csv
import copy
import json
import random
import re
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rl_agent_4
from benchmark import (
    BenchmarkOrchestrator,
    parse_args,
    summarize_run_csv,
    validate_runner_summary,
)
from ecmp_baseline import ECMPGroup, ECMPProgrammer
from rl_agent_4 import QIDS, SLA_THRESHOLDS, QoSRoutingEnv
from traffic_generator import TrafficManager


class FakeThrift:
    def __init__(self, reject_table=None, ignore_clear=False):
        self.reject_table = reject_table
        self.ignore_clear = ignore_clear
        self.counts = {
            ECMPProgrammer.GROUP_TABLE: 0,
            ECMPProgrammer.NHOP_TABLE: 0,
        }
        self.next_handle = 1

    def table_clear(self, table):
        if not self.ignore_clear:
            self.counts[table] = 0

    def table_num_entries(self, table):
        return self.counts[table]

    def table_add(self, table, _action, _keys, _params):
        if table == self.reject_table:
            return None
        self.counts[table] += 1
        handle = self.next_handle
        self.next_handle += 1
        return handle


class FakeTopo:
    @staticmethod
    def node_to_node_port_num(_switch, _next_hop):
        return 1

    @staticmethod
    def get_hosts():
        return {"h1": {}}

    @staticmethod
    def node_to_node_interface_ip(_next_hop, _switch):
        return "10.0.0.2/24"


class FakeController:
    def __init__(self, thrift):
        self.controllers = {"s1": thrift}
        self.topo = FakeTopo()

    @staticmethod
    def _call(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    @staticmethod
    def ensure_switching_and_mac(_switch, _next_hop):
        return None


def one_group():
    return [
        ECMPGroup(
            switch="s1",
            destination="h1",
            destination_ip="10.0.0.1",
            group_id=1,
            next_hops=("n1", "n2"),
        )
    ]


class ECMPProgrammingIntegrityTests(unittest.TestCase):
    def test_rejected_table_add_fails_the_run(self):
        thrift = FakeThrift(reject_table=ECMPProgrammer.NHOP_TABLE)
        programmer = ECMPProgrammer(FakeController(thrift), one_group())

        with self.assertRaisesRegex(RuntimeError, "write was rejected"):
            programmer.install()

    def test_successful_install_is_read_back(self):
        thrift = FakeThrift()
        programmer = ECMPProgrammer(FakeController(thrift), one_group())

        result = programmer.install()

        self.assertTrue(result["verified"])
        self.assertEqual(result["verification"]["observed_groups"], 1)
        self.assertEqual(result["verification"]["observed_members"], 2)

    def test_clear_must_reach_zero_entries(self):
        thrift = FakeThrift(ignore_clear=True)
        thrift.counts[ECMPProgrammer.GROUP_TABLE] = 1
        programmer = ECMPProgrammer(FakeController(thrift), one_group())

        with self.assertRaisesRegex(RuntimeError, "expected 0 entries"):
            programmer.clear(required=True)


class RLRoutingLogicIntegrityTests(unittest.TestCase):
    PROFILES = (
        "light_1",
        "medium_2",
        "high_1",
        "bursty_vo_1",
        "bursty_vi_2",
        "bursty_be_3",
    )

    @staticmethod
    def _env(profile):
        env = object.__new__(QoSRoutingEnv)
        env.current_traffic_profile = profile
        env.last_action_time = 0.0
        env.global_step = 0
        env._demand_unit_locks = {}
        return env

    @staticmethod
    def _reward_snapshot():
        return {
            0: {"lat_p95": 50.0, "drop_p95": 0.0, "util_p95": 10.0},
            1: {"lat_p95": 100.0, "drop_p95": 0.0, "util_p95": 20.0},
            7: {"lat_p95": 250.0, "drop_p95": 1.0, "util_p95": 30.0},
        }

    @staticmethod
    def _action_snapshot():
        return {
            qid: {
                "lat_p95": SLA_THRESHOLDS[qid] * 1.25,
                "bottleneck_sid": qid + 10,
                "alternatives": [{"name": "alt0"}, {"name": "alt1"}],
                "candidate_units": [
                    {
                        "qid": qid,
                        "src_ip": f"10.0.{qid}.{idx + 1}",
                        "dst_ip": f"10.1.{qid}.{idx + 1}",
                        "bottleneck_sid": qid + 10,
                        "mean_latency": 100.0 - idx,
                        "pressure_norm": 0.80 - (idx * 0.10),
                        "alternatives": [
                            {"name": f"q{qid}-alt0"},
                            {"name": f"q{qid}-alt1"},
                        ],
                    }
                    for idx in range(2)
                ],
            }
            for qid in QIDS
        }

    @staticmethod
    def _valid_cache_queue_summary():
        return {
            "ok": True,
            "metrics": {
                str(qid): {
                    "lat_p95": SLA_THRESHOLDS[qid] * 0.5,
                    "drop_p95": 0.0,
                    "util_p95": 20.0,
                }
                for qid in QIDS
            },
            "metrics_received": {
                str(qid): {"lat": True, "drop": True, "util": True}
                for qid in QIDS
            },
            "counts": {
                str(qid): {"lat": 1, "drop": 1, "util": 1}
                for qid in QIDS
            },
            "demands": {},
            "top_demands": {},
        }

    class CapturingCache:
        def __init__(self, responses=None):
            self.requests = []
            self.responses = responses or {}

        def request(self, payload):
            self.requests.append(copy.deepcopy(payload))
            return copy.deepcopy(self.responses.get(payload["kind"], {"ok": True}))

    class EmptyController:
        @staticmethod
        def get_all_switch_ids():
            return []

    class QueryRecord:
        def __init__(self, qid, value=None, measurement=None, src_ip=None, dst_ip=None, flow_id=None):
            self.values = {"queue_id": str(qid)}
            if src_ip is not None:
                self.values["src_ip"] = src_ip
            if dst_ip is not None:
                self.values["dst_ip"] = dst_ip
            if flow_id is not None:
                self.values["flow_id"] = str(flow_id)
            self._value = value
            self._measurement = measurement

        def get_value(self):
            return self._value

        def get_measurement(self):
            return self._measurement

    class QueryTable:
        def __init__(self, records):
            self.records = records

    class CapturingQueryApi:
        def __init__(self, records):
            self.records = records
            self.queries = []

        def query(self, org, query):
            self.queries.append(query)
            return [RLRoutingLogicIntegrityTests.QueryTable(self.records)]

    @staticmethod
    def _influx_env(profile, query_api):
        env = RLRoutingLogicIntegrityTests._env(profile)
        env.telemetry_cache = None
        env.telemetry_backend = "influx"
        env.bucket = "telemetry"
        env.org = "test-org"
        env.query_api = query_api
        return env

    def test_required_qids_are_profile_invariant(self):
        for profile in self.PROFILES:
            env = self._env(profile)

            self.assertEqual(env._required_qids_for_current_profile(), QIDS)
            self.assertEqual(env._min_required_valid_count(), 2)

    def test_reward_accounts_for_all_queue_slas_for_every_profile(self):
        reference = None
        for profile in self.PROFILES:
            env = self._env(profile)
            reward, info = env._compute_reward(self._reward_snapshot())

            self.assertEqual(tuple(info["reward_qids"]), QIDS)
            self.assertEqual(info["sla_total"], len(QIDS))
            self.assertEqual(set(info["per_queue"]), set(QIDS))

            observed = (
                round(float(reward), 6),
                tuple(info["sla_met"]),
                tuple(info["sla_violated"]),
            )
            if reference is None:
                reference = observed
            else:
                self.assertEqual(observed, reference)

    def test_action_mask_is_profile_invariant(self):
        expected = [True] * 14
        for profile in self.PROFILES:
            env = self._env(profile)

            self.assertEqual(
                env._get_valid_actions(self._action_snapshot()).tolist(),
                expected,
            )

    def test_bursty_profiles_do_not_neutralize_nonfocused_queues(self):
        env = self._env("bursty_be_1")
        snapshot = {
            qid: {
                "lat_p95": SLA_THRESHOLDS[qid] * 9.0,
                "drop_p95": 3.0,
                "util_p95": 77.0,
                "data_valid": False,
                "bottleneck_sid": qid + 20,
                "alternatives": [{"name": "keep"}],
            }
            for qid in QIDS
        }
        before = copy.deepcopy(snapshot)

        env._normalize_optional_queue_telemetry(snapshot)

        for qid in QIDS:
            self.assertEqual(snapshot[qid]["lat_p95"], before[qid]["lat_p95"])
            self.assertEqual(snapshot[qid]["drop_p95"], before[qid]["drop_p95"])
            self.assertEqual(snapshot[qid]["util_p95"], before[qid]["util_p95"])
            self.assertEqual(
                snapshot[qid]["alternatives"],
                before[qid]["alternatives"],
            )
            self.assertTrue(snapshot[qid]["profile_required"])
            self.assertFalse(snapshot[qid]["telemetry_optional"])

    def test_bursty_local_cache_snapshot_requests_all_queues(self):
        cache = self.CapturingCache({
            "queue_summary": self._valid_cache_queue_summary(),
        })
        env = self._env("bursty_vi_1")
        env.telemetry_cache = cache
        env.telemetry_backend = "cache"
        env.global_step = 1
        env._query_executor = SimpleNamespace(
            _work_queue=SimpleNamespace(qsize=lambda: 0)
        )
        env.controller = self.EmptyController()

        snapshot = env._collect_snapshot()

        self.assertEqual(cache.requests[0]["kind"], "queue_summary")
        self.assertEqual(cache.requests[0]["qids"], list(QIDS))
        self.assertEqual(cache.requests[0]["top_n"], rl_agent_4.TOP_N_HOT_DEMANDS)
        self.assertTrue(all(snapshot[qid]["data_valid"] for qid in QIDS))

    def test_bursty_local_cache_freshness_requests_all_queues(self):
        cache = self.CapturingCache()
        env = self._env("bursty_be_2")
        env.telemetry_cache = cache
        env.telemetry_backend = "cache"

        env._cache_freshness(
            start="2026-01-01T00:00:00Z",
            stop="2026-01-01T00:00:01Z",
        )

        self.assertEqual(cache.requests[0]["kind"], "freshness")
        self.assertEqual(cache.requests[0]["qids"], list(QIDS))

    def test_required_telemetry_freshness_uses_training_gate(self):
        response = {
            "ok": True,
            "epoch_id": 7,
            "queues": {
                str(qid): {
                    "lat": {"window_count": 1},
                    "drop": {"window_count": 1},
                    "util": {"window_count": 1},
                }
                for qid in QIDS
            },
            "missing": {},
            "complete": True,
        }
        cache = self.CapturingCache({"freshness": response})
        env = self._env("high_1")
        env.telemetry_cache = cache
        env.telemetry_backend = "cache"
        env.telemetry_liveness_enabled = True
        env.telemetry_liveness_window_seconds = 1.0
        env.telemetry_liveness_retries = 1
        env.telemetry_liveness_interval_seconds = 0.0

        report = env.verify_required_telemetry_freshness(
            raise_on_error=False
        )

        self.assertTrue(report["verified"])
        self.assertEqual(
            report["telemetry_policy"],
            "training_required_queue_metrics",
        )
        self.assertEqual(cache.requests[0]["kind"], "freshness")
        self.assertEqual(cache.requests[0]["qids"], list(QIDS))
        self.assertEqual(cache.requests[0]["required_labels"], ["lat", "drop", "util"])

    def test_required_telemetry_freshness_reports_missing_metrics(self):
        response = {
            "ok": True,
            "epoch_id": 7,
            "queues": {
                str(qid): {
                    "lat": {"window_count": 1},
                    "drop": {"window_count": 1},
                    "util": {"window_count": 1},
                }
                for qid in QIDS
            },
            "missing": {"7": ["util"]},
            "complete": False,
        }
        cache = self.CapturingCache({"freshness": response})
        env = self._env("high_1")
        env.telemetry_cache = cache
        env.telemetry_backend = "cache"
        env.telemetry_liveness_enabled = True
        env.telemetry_liveness_window_seconds = 1.0
        env.telemetry_liveness_retries = 1
        env.telemetry_liveness_interval_seconds = 0.0

        report = env.verify_required_telemetry_freshness(
            raise_on_error=False
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["missing"], {7: ["util"]})
        self.assertIn("Q7 missing util", report["errors"])

    def test_influx_aggregated_metric_filter_tracks_qids(self):
        extended_qids = (0, 1, 7, 9)
        records = [
            self.QueryRecord(qid, 1.0, measurement)
            for qid in extended_qids
            for measurement in ("lat_p95", "drop_p95", "util_p95")
        ]
        query_api = self.CapturingQueryApi(records)
        env = self._influx_env("bursty_be_1", query_api)

        with patch.object(rl_agent_4, "QIDS", extended_qids):
            result = env._query_aggregated_metrics(
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                step=1,
            )

        self.assertIn('r.queue_id == "9"', query_api.queries[0])
        self.assertTrue(result["metrics_received"][9]["lat"])
        self.assertTrue(result["metrics_received"][9]["drop"])
        self.assertTrue(result["metrics_received"][9]["util"])

    def test_influx_retry_metric_filter_uses_requested_queues(self):
        query_api = self.CapturingQueryApi([
            self.QueryRecord(9, 12.5),
        ])
        env = self._influx_env("bursty_be_1", query_api)

        result = env._retry_metric_query(
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:01Z",
            [9],
            "flow_latency",
            "flow_latency",
            step=1,
        )

        self.assertEqual(result, {9: 12.5})
        self.assertIn('r.queue_id == "9"', query_api.queries[0])
        self.assertNotIn('r.queue_id == "0"', query_api.queries[0])

    def test_bursty_influx_hot_demands_request_all_queues(self):
        query_api = self.CapturingQueryApi([
            self.QueryRecord(qid, src_ip=f"10.0.{qid}.1", dst_ip=f"10.0.{qid}.2")
            for qid in QIDS
        ])
        env = self._influx_env("bursty_vo_1", query_api)

        result = env._get_all_hottest_demands(
            step=1,
            target_qids=list(env._required_qids_for_current_profile()),
        )

        self.assertEqual(set(result), set(QIDS))
        self.assertTrue(all(isinstance(result[qid], list) for qid in QIDS))
        for qid in QIDS:
            self.assertIn(f'r.queue_id == "{qid}"', query_api.queries[0])
        self.assertIn(f"limit(n:{rl_agent_4.TOP_N_HOT_DEMANDS})", query_api.queries[0])

    def test_influx_flow_coverage_uses_int_liveness_measurement(self):
        query_api = self.CapturingQueryApi([
            self.QueryRecord(qid, flow_id=flow_id)
            for qid in QIDS
            for flow_id in (10, 11)
        ])
        env = self._influx_env("high_1", query_api)

        report = env.verify_telemetry_flow_coverage(
            [10, 11],
            window_seconds=0.1,
            retries=1,
            raise_on_error=False,
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["coverage_measurement"], "flow_telemetry_seen")
        self.assertIn('r._measurement == "flow_telemetry_seen"', query_api.queries[0])
        self.assertNotIn('r._measurement == "flow_latency"', query_api.queries[0])


class BenchmarkReportingIntegrityTests(unittest.TestCase):
    def test_runner_summary_requires_verified_routing_state(self):
        item = {
            "method": "ecmp",
            "profile": "medium_2",
            "seed": 43,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "method": "ecmp",
                        "traffic_profile": "medium_2",
                        "traffic_seed": 43,
                        "routing_state_verified": False,
                    }
                )
            )

            _, errors = validate_runner_summary(item, path)

        self.assertIn("routing_state_verified is not true", errors)
        self.assertIn(
            "routing_state evidence is missing or unverified",
            errors,
        )

    def test_verified_ecmp_runner_summary_is_accepted(self):
        item = {
            "method": "ecmp",
            "profile": "medium_2",
            "seed": 43,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "method": "ecmp",
                        "traffic_profile": "medium_2",
                        "traffic_seed": 43,
                        "routing_state_verified": True,
                        "traffic_state_verified": True,
                        "telemetry_state_verified": True,
                        "routing_state": {
                            "verified": True,
                            "plan_sha256": "abc123",
                            "traffic_processes": {
                                "verified": True,
                                "post_measurement": {
                                    "verified": True,
                                },
                            },
                        },
                    }
                )
            )

            _, errors = validate_runner_summary(item, path)

        self.assertEqual(errors, [])

    def test_runner_summary_requires_post_measurement_traffic_evidence(self):
        item = {
            "method": "ospf",
            "profile": "medium_2",
            "seed": 43,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "method": "ospf",
                        "traffic_profile": "medium_2",
                        "traffic_seed": 43,
                        "routing_state_verified": True,
                        "traffic_state_verified": True,
                        "telemetry_state_verified": True,
                        "routing_state": {
                            "verified": True,
                            "traffic_processes": {
                                "verified": True,
                            },
                        },
                    }
                )
            )

            _, errors = validate_runner_summary(item, path)

        self.assertIn("post-measurement traffic verification is missing", errors)

    def test_aggregate_summary_does_not_confuse_two_ecmp_runs(self):
        fieldnames = [
            "step",
            "reward",
            "sla_met_count",
            "data_valid",
            "load_q0_mbps",
            "load_q1_mbps",
            "load_q7_mbps",
            "q0_latency_ms",
            "q0_drop",
            "q0_util_pct",
            "q1_latency_ms",
            "q1_drop",
            "q1_util_pct",
            "q7_latency_ms",
            "q7_drop",
            "q7_util_pct",
        ]
        rows = [
            {
                "step": 1,
                "reward": 1.0,
                "sla_met_count": 3,
                "data_valid": 1,
                "load_q0_mbps": 1,
                "load_q1_mbps": 1,
                "load_q7_mbps": 1,
                "q0_latency_ms": 1,
                "q0_drop": 0,
                "q0_util_pct": 10,
                "q1_latency_ms": 1,
                "q1_drop": 0,
                "q1_util_pct": 10,
                "q7_latency_ms": 1,
                "q7_drop": 0,
                "q7_util_pct": 10,
            },
            {
                "step": 2,
                "reward": -1.0,
                "sla_met_count": 0,
                "data_valid": 1,
                "load_q0_mbps": 1,
                "load_q1_mbps": 1,
                "load_q7_mbps": 1,
                "q0_latency_ms": 10,
                "q0_drop": 1,
                "q0_util_pct": 99,
                "q1_latency_ms": 10,
                "q1_drop": 1,
                "q1_util_pct": 99,
                "q7_latency_ms": 10,
                "q7_drop": 1,
                "q7_util_pct": 99,
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize_run_csv(path)

        self.assertEqual(summary["sla_compliance_valid_pct"], 50.0)
        self.assertEqual(summary["sla_met_pct"], 50.0)
        self.assertEqual(summary["reward_mean_valid"], 0.0)


class BenchmarkInterruptTests(unittest.TestCase):
    def test_active_child_is_terminated_by_first_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = object.__new__(BenchmarkOrchestrator)
            orchestrator.args = SimpleNamespace(run_timeout_seconds=0.0)
            orchestrator.current_process = None
            orchestrator.interrupted = False
            orchestrator.interrupt_signal = None
            orchestrator._signal_count = 0
            log_path = Path(directory) / "attempt.log"
            timer = threading.Timer(
                0.2,
                orchestrator._handle_signal,
                args=(signal.SIGINT, None),
            )
            started = time.monotonic()
            timer.start()
            try:
                return_code, _ = orchestrator._execute_attempt(
                    [
                        "python3",
                        "-c",
                        "import time; time.sleep(30)",
                    ],
                    log_path,
                )
            finally:
                timer.cancel()

        self.assertEqual(return_code, 130)
        self.assertTrue(orchestrator.interrupted)
        self.assertIsNone(orchestrator.current_process)
        self.assertLess(time.monotonic() - started, 5.0)

    def test_interrupted_attempt_is_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = object.__new__(BenchmarkOrchestrator)
            orchestrator.args = SimpleNamespace(
                resume=False,
                cooldown_seconds=0.0,
                max_retries=3,
                repetitions=1,
            )
            orchestrator.output_dir = Path(directory)
            orchestrator.schedule = [1, 2, 3]
            orchestrator.current_process = None
            orchestrator.interrupted = False
            orchestrator.interrupt_signal = None
            orchestrator._signal_count = 0
            orchestrator._clean_traffic = Mock()
            orchestrator._command_for = Mock(return_value=["fake"])

            def interrupt_attempt(_command, _log_path):
                orchestrator.interrupted = True
                orchestrator.interrupt_signal = signal.SIGINT
                return 130, 0.1

            orchestrator._execute_attempt = Mock(side_effect=interrupt_attempt)
            item = {
                "sequence": 1,
                "profile": "light_1",
                "repetition": 1,
                "seed": 42,
                "order_index": 1,
                "method": "ecmp",
                "block_order": ["ecmp", "rl", "ospf"],
            }

            result = orchestrator.run_one(item)

        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(orchestrator._execute_attempt.call_count, 1)


class TrafficIntegrityTests(unittest.TestCase):
    @staticmethod
    def _make_variables():
        variables = {}
        for line in Path("Makefile").read_text().splitlines():
            match = re.match(r"^([A-Z0-9_]+)\s*\?=\s*(.*)$", line)
            if match:
                variables[match.group(1)] = match.group(2).strip()
        return variables

    @staticmethod
    def _expand_make_value(value, variables):
        previous = None
        while value != previous:
            previous = value
            for name in re.findall(r"\$\(([A-Z0-9_]+)\)", value):
                value = value.replace(f"$({name})", variables.get(name, ""))
        return value

    def test_makefile_profile_lists_cover_runtime_registry(self):
        variables = self._make_variables()
        expected = list(TrafficManager.TRAFFIC_PROFILES)

        all_profiles = self._expand_make_value(
            variables["ALL_TRAFFIC_PROFILES"],
            variables,
        ).split()
        all_profiles_csv = [
            item
            for item in self._expand_make_value(
                variables["ALL_TRAFFIC_PROFILES_CSV"],
                variables,
            ).split(",")
            if item
        ]
        bench_profiles = [
            item
            for item in self._expand_make_value(
                variables["BENCH_PROFILES"],
                variables,
            ).split(",")
            if item
        ]

        self.assertEqual(all_profiles, expected)
        self.assertEqual(all_profiles_csv, expected)
        self.assertEqual(bench_profiles, expected)
        self.assertEqual(
            self._expand_make_value(
                variables["PRODUCTION_PROFILES"],
                variables,
            ).split(),
            expected,
        )
        for name in (
            "TRAIN_FOUNDATION_PROFILE_WEIGHTS",
            "TRAIN_BURST_PROFILE_WEIGHTS",
            "TRAIN_POLISH_PROFILE_WEIGHTS",
        ):
            value = self._expand_make_value(variables[name], variables)
            weighted_profiles = [
                item.split(":", 1)[0]
                for item in value.split(",")
                if item
            ]
            self.assertEqual(weighted_profiles, expected)

    def test_benchmark_default_profiles_cover_runtime_registry(self):
        with patch.object(sys, "argv", ["benchmark.py"]):
            args = parse_args()
        self.assertEqual(args.profiles, list(TrafficManager.TRAFFIC_PROFILES))
        self.assertEqual(args.methods, ["rl", "ecmp", "ospf"])

    def test_benchmark_methods_are_configurable(self):
        with patch.object(
            sys,
            "argv",
            ["benchmark.py", "--methods", "rl,ecmp"],
        ):
            args = parse_args()
        self.assertEqual(args.methods, ["rl", "ecmp"])

    def test_benchmark_schedule_uses_selected_methods(self):
        with patch.object(
            sys,
            "argv",
            [
                "benchmark.py",
                "--profiles",
                "high_1",
                "--repetitions",
                "3",
                "--methods",
                "rl,ecmp",
            ],
        ):
            args = parse_args()
        orchestrator = BenchmarkOrchestrator(args)

        orchestrator.build_schedule()

        self.assertEqual(len(orchestrator.schedule), 6)
        self.assertEqual(
            {item["method"] for item in orchestrator.schedule},
            {"rl", "ecmp"},
        )
        self.assertNotIn(
            "ospf",
            {item["method"] for item in orchestrator.schedule},
        )

    def test_tc_root_class_and_rate_parser(self):
        parsed = TrafficManager._parse_root_htb_class(
            "class htb 5:1 root prio 0 rate 4740Kbit "
            "ceil 4740Kbit burst 15Kb cburst 1600b"
        )
        self.assertEqual(parsed["classid"], "5:1")
        self.assertAlmostEqual(
            TrafficManager._tc_rate_to_mbps(parsed["rate"]),
            4.740,
        )

    def test_short_horizon_shaped_profiles_are_fixed_and_ordered(self):
        expected_high_steps = {
            "light_1": 0,
            "light_2": 0,
            "medium_1": 3,
            "medium_2": 4,
            "high_1": 6,
            "high_2": 0,
        }
        average_totals = []
        expected_ranges = {
            "light_1": (0.60, 0.60),
            "light_2": (0.75, 0.75),
            "medium_1": (1.563, 1.803),
            "medium_2": (1.50, 1.81),
            "high_1": (2.70, 3.15),
            "high_2": (2.98, 2.98),
        }

        for profile in (
            "light_1",
            "light_2",
            "medium_1",
            "medium_2",
            "high_1",
            "high_2",
        ):
            stages = TrafficManager.PROFILE_STAGE_LOADS[profile]
            ranges = TrafficManager.TRAFFIC_PROFILES[profile]

            self.assertEqual(
                ranges,
                {
                    qid: (
                        min(stages["low"][qid], stages["high"][qid]),
                        max(stages["low"][qid], stages["high"][qid]),
                    )
                    for qid in stages["low"]
                },
            )
            pattern = TrafficManager.SHAPED_PROFILE_PATTERNS[profile]
            self.assertEqual(sum(pattern), expected_high_steps[profile])
            self.assertAlmostEqual(
                sum(stages["low"].values()),
                expected_ranges[profile][0],
                places=3,
            )
            self.assertAlmostEqual(
                sum(stages["high"].values()),
                expected_ranges[profile][1],
                places=3,
            )
            average_totals.append(
                (
                    sum(pattern) * expected_ranges[profile][1]
                    + (len(pattern) - sum(pattern)) * expected_ranges[profile][0]
                )
                / len(pattern)
            )

        self.assertLess(average_totals[0], average_totals[1])
        self.assertLess(
            sum(TrafficManager.PROFILE_STAGE_LOADS["light_2"]["high"].values()),
            sum(TrafficManager.PROFILE_STAGE_LOADS["medium_1"]["high"].values()),
        )
        self.assertLess(
            sum(TrafficManager.PROFILE_STAGE_LOADS["medium_1"]["high"].values()),
            sum(TrafficManager.PROFILE_STAGE_LOADS["medium_2"]["high"].values()),
        )
        self.assertLess(average_totals[3], average_totals[4])
        self.assertLess(average_totals[4], average_totals[5])

    def test_one_registry_covers_every_profile_and_derives_categories(self):
        expected = {
            profile
            for profiles in TrafficManager.PROFILE_GROUPS.values()
            for profile in profiles
        }
        self.assertEqual(set(TrafficManager.TRAFFIC_PROFILES), expected)
        self.assertEqual(
            TrafficManager.profile_category("light_1"),
            "light",
        )
        self.assertEqual(
            TrafficManager.profile_category("bursty_be_2"),
            "bursty",
        )

    def test_profile_selection_rejects_unknown_names_and_categories(self):
        manager = object.__new__(TrafficManager)
        manager._rng = random.Random(42)

        self.assertEqual(
            manager._choose_profile_name("light_1", {"high": 1.0}),
            "light_1",
        )
        with self.assertRaises(ValueError):
            manager._choose_profile_name("not_a_profile", None)
        with self.assertRaises(ValueError):
            manager._choose_profile_name(None, {"not_a_category": 1.0})
        with self.assertRaises(ValueError):
            manager._choose_profile_name(None, {"light": 0.0})

    def test_bursty_profiles_have_queue_biased_bursts(self):
        expected_high_totals = {
            "vo": {1: 3.40, 2: 3.155, 3: 3.04},
            "vi": {1: 3.25, 2: 3.20, 3: 3.03},
            "be": {1: 3.35, 2: 3.13, 3: 3.148},
        }
        for queue_name, qid in (("vo", 0), ("vi", 1), ("be", 7)):
            for suffix, high_steps in ((1, 3), (2, 6), (3, 8)):
                profile = f"bursty_{queue_name}_{suffix}"
                stages = TrafficManager.PROFILE_STAGE_LOADS[profile]
                self.assertIn(profile, TrafficManager.TRAFFIC_PROFILES)
                self.assertEqual(
                    TrafficManager.BURSTY_PROFILE_HIGH_STEPS[profile],
                    high_steps,
                )
                self.assertEqual(
                    max(stages["high"], key=stages["high"].get),
                    qid,
                )
                self.assertAlmostEqual(sum(stages["low"].values()), 0.70)
                self.assertAlmostEqual(
                    sum(stages["high"].values()),
                    expected_high_totals[queue_name][suffix],
                )

    def test_flow_load_plan_is_heterogeneous_bounded_and_preserves_average(self):
        manager = object.__new__(TrafficManager)
        manager.traffic_pairs = [
            (f"h{index % 6 + 1}", f"h{index % 6 + 7}", 10 + index)
            for index in range(18)
        ]
        manager._max_flow_total_mbps = 5.0
        manager._flow_load_weights = manager._build_flow_load_weights()

        average_loads = TrafficManager.PROFILE_STAGE_LOADS["high_2"]["high"]
        plan = manager._flow_load_plan(average_loads)
        totals = [sum(loads.values()) for loads in plan.values()]

        self.assertEqual(len(plan), 18)
        self.assertLess(min(totals), max(totals))
        self.assertLessEqual(max(totals), 5.0)
        self.assertAlmostEqual(
            sum(totals) / len(totals),
            sum(average_loads.values()),
            places=6,
        )

    def test_profiles_are_calibrated_for_single_path_ospf_hot_edge(self):
        link_mbps = 10.0
        hot_edge_demands = 10.0

        def hot_edge_ratio(profile, stage):
            total = sum(TrafficManager.PROFILE_STAGE_LOADS[profile][stage].values())
            return total * hot_edge_demands / link_mbps

        self.assertAlmostEqual(hot_edge_ratio("light_1", "high"), 0.60)
        self.assertAlmostEqual(hot_edge_ratio("light_2", "high"), 0.75)
        self.assertAlmostEqual(hot_edge_ratio("medium_1", "high"), 1.803)
        self.assertAlmostEqual(hot_edge_ratio("medium_2", "high"), 1.81)
        self.assertAlmostEqual(hot_edge_ratio("high_1", "high"), 3.15)
        self.assertAlmostEqual(hot_edge_ratio("high_2", "high"), 2.98)

        expected_focused = {
            "bursty_vo_1": (0, 3.40 * 0.72),
            "bursty_vo_2": (0, 3.155 * 0.72),
            "bursty_vo_3": (0, 3.04 * 0.72),
            "bursty_vi_1": (1, 3.25 * 0.70),
            "bursty_vi_2": (1, 3.20 * 0.70),
            "bursty_vi_3": (1, 3.03 * 0.70),
            "bursty_be_1": (7, 3.35 * 0.75),
            "bursty_be_2": (7, 3.13 * 0.75),
            "bursty_be_3": (7, 3.148 * 0.75),
        }
        for profile, (qid, expected_total) in expected_focused.items():
            focused_load = TrafficManager.PROFILE_STAGE_LOADS[profile]["high"][qid]
            ratio = focused_load * hot_edge_demands / link_mbps
            self.assertAlmostEqual(ratio, expected_total)

    def test_bursty_cycles_are_seeded_random_with_fixed_high_step_count(self):
        first = object.__new__(TrafficManager)
        first.current_profile_name = "bursty_vo_1"
        first._seed_material = "42"
        first._bursty_cycle_patterns = {}

        second = object.__new__(TrafficManager)
        second.current_profile_name = "bursty_vo_1"
        second._seed_material = "42"
        second._bursty_cycle_patterns = {}

        pattern_0 = first._bursty_pattern_for_cycle(0)
        pattern_1 = first._bursty_pattern_for_cycle(1)

        self.assertEqual(len(pattern_0), 10)
        self.assertEqual(sum(pattern_0), 3)
        self.assertEqual(len(pattern_1), 10)
        self.assertEqual(sum(pattern_1), 3)
        self.assertEqual(pattern_0, second._bursty_pattern_for_cycle(0))
        self.assertEqual(pattern_1, second._bursty_pattern_for_cycle(1))
        self.assertNotEqual(pattern_0, pattern_1)

    def test_bursty_profiles_use_longer_telemetry_coverage_window(self):
        manager = object.__new__(TrafficManager)
        manager.current_profile_category = "bursty"
        self.assertEqual(manager.telemetry_coverage_window_seconds(5.0), 30.0)
        self.assertEqual(manager.telemetry_coverage_window_seconds(45.0), 45.0)

        manager.current_profile_category = "medium"
        self.assertEqual(manager.telemetry_coverage_window_seconds(5.0), 5.0)

    def test_step_profile_follows_shaped_pattern(self):
        manager = object.__new__(TrafficManager)
        manager.current_profile_name = "bursty_be_1"
        manager.current_load = TrafficManager.PROFILE_STAGE_LOADS[
            "bursty_be_1"
        ]["low"].copy()
        manager._shaped_stage_high = False
        manager._bursty_cycle_patterns = {
            0: (0, 0, 1, 1, 1, 0, 0, 0, 0, 0),
        }
        manager._apply_sender_caps = Mock(
            return_value={"verified": True}
        )

        self.assertIsNone(manager.apply_step_profile(1))
        report = manager.apply_step_profile(3)
        self.assertEqual(report["step"], 3)
        self.assertEqual(
            manager.current_load,
            TrafficManager.PROFILE_STAGE_LOADS["bursty_be_1"]["high"],
        )
        manager._apply_sender_caps.assert_called_once_with(
            "bursty_be_1",
            verify=False,
            verify_reason="step_3",
        )

    def test_bursty_step_profile_keeps_schedule_with_periodic_verification(self):
        manager = object.__new__(TrafficManager)
        manager.current_profile_name = "bursty_be_1"
        manager.current_load = TrafficManager.PROFILE_STAGE_LOADS[
            "bursty_be_1"
        ]["low"].copy()
        manager._shaped_stage_high = False
        manager._bursty_cycle_patterns = {
            0: (0, 0, 1, 1, 1, 0, 0, 0, 0, 0),
            1: (0, 0, 1, 0, 0, 0, 0, 0, 1, 1),
        }

        def fake_apply(profile_name, verify=True, verify_reason="startup"):
            return {
                "profile": profile_name,
                "verified": verify,
                "verify_reason": verify_reason,
            }

        manager._apply_sender_caps = Mock(side_effect=fake_apply)

        reports = {}
        for step in range(1, 21):
            report = manager.apply_step_profile(step)
            if report is not None:
                reports[step] = report

        self.assertEqual(sorted(reports), [3, 6, 13, 14, 19, 20])
        self.assertEqual(
            manager.current_load,
            TrafficManager.PROFILE_STAGE_LOADS["bursty_be_1"]["high"],
        )
        self.assertEqual(reports[19]["profile_cycle"], 2)
        self.assertEqual(reports[19]["cycle_pattern"], "0010000011")
        self.assertFalse(reports[19]["verified"])
        self.assertTrue(reports[20]["verified"])
        self.assertFalse(reports[20]["stage_changed"])
        self.assertEqual(
            [call.kwargs["verify"] for call in manager._apply_sender_caps.call_args_list],
            [False, False, False, False, False, True],
        )

    def test_step_profile_verifies_every_twenty_steps(self):
        manager = object.__new__(TrafficManager)
        manager.current_profile_name = "light_1"
        manager.current_load = TrafficManager.PROFILE_STAGE_LOADS[
            "light_1"
        ]["low"].copy()
        manager._shaped_stage_high = False
        manager._apply_sender_caps = Mock(
            return_value={"verified": True}
        )

        report = manager.apply_step_profile(20)

        self.assertEqual(report["step"], 20)
        self.assertEqual(report["profile_step"], 20)
        self.assertFalse(report["stage_changed"])
        manager._apply_sender_caps.assert_called_once_with(
            "light_1",
            verify=True,
            verify_reason="step_20",
        )

    def test_sender_cap_skip_verification_uses_cached_tc_metadata(self):
        manager = object.__new__(TrafficManager)
        manager._tc_original_classes = {
            "h1": {
                "pid": "1234",
                "interface": "h1-eth0",
                "classid": "1:1",
                "rate": "10Mbit",
                "ceil": "10Mbit",
                "burst": "15Kb",
                "cburst": "15Kb",
            }
        }

        with patch("traffic_generator.subprocess.run") as run:
            run.return_value = SimpleNamespace(stdout="")
            report = manager._set_sender_rate_cap("h1", 3.2, verify=False)

        self.assertEqual(run.call_count, 1)
        self.assertIn("change", run.call_args.args[0])
        self.assertTrue(report["verification_skipped"])

    def test_begin_measurement_applies_target_without_restarting_flows(self):
        manager = object.__new__(TrafficManager)
        manager.current_profile_name = "medium_1"
        manager.current_load = {0: 0.1, 1: 0.2, 7: 0.2}
        manager._shaped_stage_high = None
        manager._apply_sender_caps = Mock(
            return_value={"verified": True}
        )

        with patch("traffic_generator.time.sleep") as sleep:
            report = manager.begin_measurement(settle_seconds=1.0)

        self.assertEqual(report["phase"], "measurement_start")
        self.assertEqual(
            manager.current_load,
            TrafficManager.PROFILE_STAGE_LOADS["medium_1"]["low"],
        )
        manager._apply_sender_caps.assert_called_once_with("medium_1")
        sleep.assert_called_once_with(1.0)

    def test_begin_measurement_is_idempotent_when_target_stage_is_active(self):
        manager = object.__new__(TrafficManager)
        manager.current_profile_name = "medium_1"
        manager.current_load = TrafficManager.PROFILE_STAGE_LOADS[
            "medium_1"
        ]["low"].copy()
        manager._shaped_stage_high = False
        manager._tc_shape_report = {
            "verified": True,
            "profile": "medium_1",
            "stage": "low",
        }
        manager._apply_sender_caps = Mock()

        with patch("traffic_generator.time.sleep") as sleep:
            report = manager.begin_measurement(settle_seconds=1.0)

        self.assertTrue(report["already_active"])
        self.assertEqual(report["settle_seconds"], 0.0)
        manager._apply_sender_caps.assert_not_called()
        sleep.assert_not_called()

    def test_exact_process_verification_counts_real_iperf_commands(self):
        manager = object.__new__(TrafficManager)
        manager.traffic_pairs = [("h1", "h2", 10)]
        output = "\n".join(
            [
                "101 iperf3 -s -p 6100",
                "102 iperf3 -c 10.0.0.2 -p 6100",
                "103 iperf3 -s -p 6101",
                "104 iperf3 -c 10.0.0.2 -p 6101",
                "105 iperf3 -s -p 6107",
                "106 iperf3 -c 10.0.0.2 -p 6107",
            ]
        )
        completed = SimpleNamespace(returncode=0, stdout=output)

        with patch("traffic_generator.subprocess.run", return_value=completed):
            report = manager.verify_exact_processes(
                timeout=0.01,
                raise_on_error=False,
            )

        self.assertTrue(report["verified"])
        self.assertEqual(report["observed_total"], 6)
        self.assertEqual(report["observed_per_queue"], {0: 2, 1: 2, 7: 2})

    def test_exact_verification_rejects_duplicate_replacing_missing_port(self):
        manager = object.__new__(TrafficManager)
        manager.traffic_pairs = [("h1", "h2", 10)]
        # Still six processes and two per queue, but Q0 has a duplicated client
        # and no server. Aggregate-only validation would incorrectly pass.
        output = "\n".join(
            [
                "101 iperf3 -c 10.0.0.2 -p 6100",
                "102 iperf3 -c 10.0.0.2 -p 6100",
                "103 iperf3 -s -p 6101",
                "104 iperf3 -c 10.0.0.2 -p 6101",
                "105 iperf3 -s -p 6107",
                "106 iperf3 -c 10.0.0.2 -p 6107",
            ]
        )
        completed = SimpleNamespace(returncode=0, stdout=output)

        with patch("traffic_generator.subprocess.run", return_value=completed):
            report = manager.verify_exact_processes(
                timeout=0.01,
                raise_on_error=False,
            )

        self.assertFalse(report["verified"])
        self.assertTrue(report["missing_endpoints"])
        self.assertTrue(report["duplicate_endpoints"])

    def test_start_retries_then_requires_exact_success(self):
        manager = object.__new__(TrafficManager)
        manager._startup_recovery_count = 0
        manager._last_start_report = None
        manager._traffic_failed = False
        manager._traffic_active = True
        manager._stop_all_iperf = Mock()
        manager._start_servers = Mock()
        manager._start_clients = Mock()
        manager._restart_all_taskservers = Mock(return_value=True)
        failed = {
            "verified": False,
            "observed_total": 5,
            "expected_total": 6,
            "observed_per_queue": {0: 1, 1: 2, 7: 2},
            "errors": ["missing endpoints"],
        }
        passed = {
            "verified": True,
            "observed_total": 6,
            "expected_total": 6,
            "observed_per_queue": {0: 2, 1: 2, 7: 2},
            "errors": [],
        }
        manager.verify_exact_processes = Mock(
            side_effect=[failed, passed]
        )

        with patch("traffic_generator.time.sleep"):
            report = manager._ensure_complete_traffic(
                1250,
                max_attempts=3,
                context="test",
            )

        self.assertTrue(report["verified"])
        self.assertEqual(report["attempts_used"], 2)
        manager._restart_all_taskservers.assert_called_once()

    def test_start_fails_closed_after_exhausting_retries(self):
        manager = object.__new__(TrafficManager)
        manager._startup_recovery_count = 0
        manager._last_start_report = None
        manager._traffic_failed = False
        manager._traffic_active = True
        manager._stop_all_iperf = Mock()
        manager._start_servers = Mock()
        manager._start_clients = Mock()
        manager._restart_all_taskservers = Mock(return_value=True)
        manager.verify_exact_processes = Mock(
            return_value={
                "verified": False,
                "observed_total": 5,
                "expected_total": 6,
                "observed_per_queue": {0: 1, 1: 2, 7: 2},
                "errors": ["missing endpoints"],
            }
        )

        with patch("traffic_generator.time.sleep"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Unable to establish complete traffic",
            ):
                manager._ensure_complete_traffic(
                    1250,
                    max_attempts=2,
                    context="test",
                )

        self.assertFalse(manager._traffic_active)
        self.assertTrue(manager._traffic_failed)

    def test_health_monitor_restart_invalidates_measured_run(self):
        manager = object.__new__(TrafficManager)
        manager.traffic_pairs = [("h1", "h2", 10)]
        manager._restart_count = 1
        output = "\n".join(
            [
                "101 iperf3 -s -p 6100",
                "102 iperf3 -c 10.0.0.2 -p 6100",
                "103 iperf3 -s -p 6101",
                "104 iperf3 -c 10.0.0.2 -p 6101",
                "105 iperf3 -s -p 6107",
                "106 iperf3 -c 10.0.0.2 -p 6107",
            ]
        )
        completed = SimpleNamespace(returncode=0, stdout=output)

        with patch("traffic_generator.subprocess.run", return_value=completed):
            report = manager.verify_exact_processes(
                timeout=0.01,
                require_no_restarts=True,
                raise_on_error=False,
            )

        self.assertFalse(report["verified"])
        self.assertIn("restart(s)", report["errors"][0])

    def test_health_monitor_requires_sustained_endpoint_mismatch(self):
        manager = object.__new__(TrafficManager)
        manager.verify_exact_processes = Mock(
            return_value={"verified": True, "errors": []}
        )
        manager._ensure_complete_traffic = Mock()

        manager._check_and_restart_traffic()

        manager.verify_exact_processes.assert_called_once_with(
            timeout=3.0,
            raise_on_error=False,
        )
        manager._ensure_complete_traffic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
