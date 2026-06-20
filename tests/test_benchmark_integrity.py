import csv
import json
import random
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark import (
    BenchmarkOrchestrator,
    summarize_run_csv,
    validate_runner_summary,
)
from ecmp_baseline import ECMPGroup, ECMPProgrammer
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
                        },
                    }
                )
            )

            _, errors = validate_runner_summary(item, path)

        self.assertEqual(errors, [])

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
            "light_1": (1.50, 1.50),
            "light_2": (1.62, 1.62),
            "medium_1": (1.50, 1.80),
            "medium_2": (1.50, 1.92),
            "high_1": (1.50, 1.85),
            "high_2": (1.85, 1.85),
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

    def test_bursty_profiles_have_deterministic_queue_biased_bursts(self):
        for queue_name, qid in (("vo", 0), ("vi", 1), ("be", 7)):
            for suffix, high_steps in ((1, 3), (2, 6)):
                profile = f"bursty_{queue_name}_{suffix}"
                stages = TrafficManager.PROFILE_STAGE_LOADS[profile]
                pattern = TrafficManager.SHAPED_PROFILE_PATTERNS[profile]
                self.assertIn(profile, TrafficManager.TRAFFIC_PROFILES)
                self.assertEqual(sum(pattern), high_steps)
                self.assertEqual(len(pattern), 10)
                self.assertEqual(
                    max(stages["high"], key=stages["high"].get),
                    qid,
                )
                self.assertAlmostEqual(sum(stages["low"].values()), 0.5)
                self.assertAlmostEqual(sum(stages["high"].values()), 3.2)

    def test_step_profile_follows_shaped_pattern(self):
        manager = object.__new__(TrafficManager)
        manager.current_profile_name = "bursty_be_1"
        manager.current_load = TrafficManager.PROFILE_STAGE_LOADS[
            "bursty_be_1"
        ]["low"].copy()
        manager._shaped_stage_high = False
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
            "bursty_be_1"
        )

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


if __name__ == "__main__":
    unittest.main()
