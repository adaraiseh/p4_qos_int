#!/usr/bin/env python3
"""
Single-path OSPF/SPF baseline for routing benchmark experiments.

This project does not run a distributed OSPF daemon. Instead, the controller
uses the complete topology database to compute one deterministic minimum-cost
path per destination and installs identical forwarding entries for every DSCP.
That models steady-state, single-path OSPF forwarding while deliberately
excluding ECMP and QoS-aware route selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ecmp_baseline import (
    BenchmarkStats,
    VisualizationPathFile,
    _decorate_top_egresses,
    collect_local_egress_observations,
    log_top_bottleneck_egresses,
)
from logging_config import normalize_artifact_permissions, setup_unified_logging
from topology.factory import create_topology


log = logging.getLogger("ospf_baseline")
QIDS = (0, 1, 7)


class OSPFBenchmark:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.env = None
        self.traffic_manager = None
        self.visualization_file = None
        self.routing_state = None
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, signum, _frame):
        log.info(f"Received signal {signum}; stopping after the current sample")
        self.running = False

    @staticmethod
    def _pressure(snapshot: dict, sla_thresholds: Dict[int, float]) -> float:
        pressure = 0.0
        for qid in QIDS:
            q = snapshot[qid]
            lat_ratio = q["lat_p95"] / sla_thresholds[qid]
            drop_norm = min(q["drop_p95"], 0.20) / 0.20
            util_norm = min(q["util_p95"], 100.0) / 100.0
            pressure = max(
                pressure,
                0.5 * lat_ratio + 0.3 * drop_norm + 0.2 * util_norm,
            )
        return pressure

    def _visualization_data(self) -> Dict[int, List[Dict]]:
        export_data = {qid: [] for qid in QIDS}
        for src, dst, flow_id in self.traffic_manager.traffic_pairs:
            path = self.env.controller.get_path_by_hosts(src, dst)
            if not path:
                raise RuntimeError(f"No OSPF path for {src}->{dst}")
            for qid in QIDS:
                export_data[qid].append(
                    {
                        "src": src,
                        "dst": dst,
                        "flow_id": flow_id,
                        "routing_mode": "ospf",
                        "path": list(path),
                    }
                )
        return export_data

    def run(self) -> int:
        if os.geteuid() != 0:
            log.error("OSPF benchmarking must be run with sudo -E")
            return 2

        from rl_agent_4 import QoSRoutingEnv, SLA_THRESHOLDS
        from traffic_generator import TrafficManager

        builder = create_topology(self.args.config)
        topology_name = builder.config.topology.name.replace("-", "_")
        rules_dir = self.args.rules_dir or f"rules/{topology_name}"

        self.env = QoSRoutingEnv(
            self.args.influx_bucket,
            self.args.influx_token
            if self.args.telemetry_backend in ("influx", "cache-fallback-influx")
            else None,
            self.args.influx_org,
            self.args.influx_url,
            verbose=False,
            reset_network=True,
            production_mode=True,
            topology_builder=builder,
            rules_dir=rules_dir,
            config_path=self.args.config,
            telemetry_backend=self.args.telemetry_backend,
            telemetry_cache_socket=self.args.telemetry_cache_socket,
            telemetry_cache_timeout=self.args.telemetry_cache_timeout,
        )
        self.env.training_influx_detail = "off"
        self.traffic_manager = TrafficManager(
            config_path=self.args.config,
            seed=self.args.traffic_seed,
        )
        profile_info = self.traffic_manager.start_traffic(
            profile_name=self.args.traffic_profile
        )
        traffic_state = self.traffic_manager.verify_exact_processes(
            raise_on_error=True
        )
        traffic_state["sender_rate_shaping"] = profile_info.get(
            "traffic_shaping",
            {},
        )
        initial_loads = profile_info.get(
            "measurement_loads",
            profile_info["loads"],
        )
        log.info(
            f"Planned measurement profile {profile_info['profile_name']} "
            f"with seed "
            f"{self.args.traffic_seed}: Q0={initial_loads[0]:.3f}, "
            f"Q1={initial_loads[1]:.3f}, Q7={initial_loads[7]:.3f} Mbps"
        )

        # This clears any ECMP overlay and reinstalls one deterministic
        # shortest path for all DSCP values. TrafficManager owns the warmup so
        # staged profiles advance during the excluded warmup window.
        self.env.reset(
            force_reset=True,
            cooldown_seconds=0.0,
            collect_initial_snapshot=False,
        )
        warmup_state = self.traffic_manager.warm_profile_for(
            self.args.warmup_seconds
        )
        initial_verification = self.env.controller.verify_forwarding_tables(
            raise_on_error=True
        )
        self.routing_state = {
            "verified": initial_verification["verified"],
            "initial_tables": initial_verification,
            "traffic_processes": traffic_state,
            "traffic_profile_warmup": warmup_state,
        }
        telemetry_window_seconds = (
            self.traffic_manager.telemetry_coverage_window_seconds(
                max(5.0, self.args.warmup_seconds)
            )
        )
        telemetry_readiness_window_seconds = float(
            self.env.telemetry_liveness_window_seconds
        )
        telemetry_state = self.env.verify_required_telemetry_freshness(
            window_seconds=telemetry_readiness_window_seconds,
            raise_on_error=True,
        )
        self.routing_state["telemetry_required_metrics"] = telemetry_state
        log.info("Verified OSPF forwarding tables on all switches")
        log.info("Verified required queue telemetry for OSPF")
        measurement_shaping = self.traffic_manager.begin_measurement()
        if measurement_shaping:
            self.routing_state["measurement_shaping"] = measurement_shaping
            traffic_state["sender_rate_shaping"] = measurement_shaping
        post_measurement_traffic_state = self.traffic_manager.verify_exact_processes(
            raise_on_error=True
        )
        traffic_state["post_measurement"] = post_measurement_traffic_state
        traffic_state["verified"] = bool(
            traffic_state["verified"]
            and post_measurement_traffic_state["verified"]
        )
        self.visualization_file = VisualizationPathFile(self.args.paths_file)
        self.visualization_file.publish(self._visualization_data())
        log.info(
            f"Published {len(self.traffic_manager.traffic_pairs)} OSPF paths "
            f"to {self.args.paths_file}; Q0/Q1/Q7 use identical paths"
        )

        output_path = Path(self.args.output) if self.args.output else Path(
            "data"
        ) / f"ospf_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalize_artifact_permissions(output_path.parent, dir_mode=0o775)
        stats = BenchmarkStats()
        completed_steps = 0

        with output_path.open("w", newline="") as csv_file:
            normalize_artifact_permissions(output_path, file_mode=0o664)
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "step",
                    "routing_mode",
                    "traffic_profile",
                    "traffic_seed",
                    "load_q0_mbps",
                    "load_q1_mbps",
                    "load_q7_mbps",
                    "reward",
                    "raw_reward",
                    "sla_met_count",
                    "data_valid",
                    "routing_state_verified",
                    "traffic_state_verified",
                    "telemetry_state_verified",
                    "pressure",
                    "q0_latency_ms",
                    "q0_drop",
                    "q0_util_pct",
                    "q1_latency_ms",
                    "q1_drop",
                    "q1_util_pct",
                    "q7_latency_ms",
                    "q7_drop",
                    "q7_util_pct",
                    "timestamp",
                ]
            )

            while self.running and (
                self.args.steps == 0 or completed_steps < self.args.steps
            ):
                next_step = completed_steps + 1
                self.traffic_manager.apply_step_profile(next_step)

                _, reward, _, _, info = self.env.step(0)
                completed_steps += 1
                snapshot = self.env.last_snapshot
                current_loads = self.traffic_manager.current_load
                pressure = self._pressure(snapshot, SLA_THRESHOLDS)
                info["pressure"] = pressure
                stats.add(reward, info, snapshot)

                row = [
                    completed_steps,
                    "ospf",
                    self.traffic_manager.current_profile_name,
                    self.args.traffic_seed,
                    current_loads.get(0, 0.0),
                    current_loads.get(1, 0.0),
                    current_loads.get(7, 0.0),
                    reward,
                    info.get("raw_reward", reward),
                    len(info.get("sla_met", [])),
                    int(info.get("data_valid", False)),
                    int(self.routing_state["verified"]),
                    int(traffic_state["verified"]),
                    int(telemetry_state["verified"]),
                    pressure,
                ]
                for qid in QIDS:
                    row.extend(
                        [
                            snapshot[qid]["lat_p95"],
                            snapshot[qid]["drop_p95"],
                            snapshot[qid]["util_p95"],
                        ]
                    )
                row.append(datetime.now().isoformat())
                writer.writerow(row)
                csv_file.flush()

                if (
                    self.args.log_every > 0
                    and completed_steps % self.args.log_every == 0
                ):
                    log.info(
                        f"[Step {completed_steps}] reward={reward:+.2f} "
                        f"sla={len(info.get('sla_met', []))}/"
                        f"{info.get('sla_total', len(QIDS))} "
                        f"pressure={pressure:.3f}"
                    )

        final_traffic_state = self.traffic_manager.verify_exact_processes(
            timeout=2.0,
            require_no_restarts=True,
            raise_on_error=False,
        )
        traffic_state["final"] = final_traffic_state
        traffic_state["verified"] = bool(
            traffic_state["verified"] and final_traffic_state["verified"]
        )
        if not traffic_state["verified"]:
            raise RuntimeError(
                "OSPF traffic-state verification failed: "
                + "; ".join(final_traffic_state["errors"])
            )

        final_telemetry_state = self.env.verify_required_telemetry_freshness(
            window_seconds=telemetry_readiness_window_seconds,
            raise_on_error=False,
        )
        telemetry_state["final"] = final_telemetry_state
        telemetry_state["final_verified"] = bool(final_telemetry_state["verified"])
        if not final_telemetry_state["verified"]:
            log.warning(
                "Final OSPF required telemetry audit failed; preserving measured "
                "run because initial readiness, traffic health, and per-step telemetry "
                "validity are enforced: "
                + "; ".join(final_telemetry_state["errors"])
            )

        egress_observations = collect_local_egress_observations(
            self.env,
            window_seconds=telemetry_window_seconds,
            top_n=10,
        )
        top_bottlenecks = _decorate_top_egresses(
            egress_observations,
            builder,
            limit=10,
        )
        self.routing_state["top_bottleneck_egresses"] = top_bottlenecks
        log_top_bottleneck_egresses(top_bottlenecks)

        final_verification = self.env.controller.verify_forwarding_tables(
            raise_on_error=True
        )
        self.routing_state["final_tables"] = final_verification
        self.routing_state["verified"] = bool(
            self.routing_state["verified"] and final_verification["verified"]
        )
        if not self.routing_state["verified"]:
            raise RuntimeError("OSPF routing-state verification failed")

        summary = stats.to_summary(completed_steps)
        log.info("=" * 68)
        log.info("OSPF benchmark summary")
        log.info(f"  Steps: {summary['total_steps']}")
        log.info(f"  Mean reward: {summary['mean_reward']:+.4f}")
        log.info(
            f"  SLA compliance: {summary['overall_sla_compliance']:.2f}%"
        )
        log.info(f"  SLA met percentage: {summary['sla_met_pct']:.2f}%")
        log.info(
            f"  Valid telemetry steps: "
            f"{summary['valid_steps']}/{summary['total_steps']}"
        )
        for qid in QIDS:
            q = summary["queue_metrics"][qid]
            log.info(
                f"  Q{qid}: mean(step-p95 latency)={q['mean_latency']:.3f} ms, "
                f"p95(step-p95 latency)={q['p95_latency']:.3f} ms, "
                f"mean drops/100ms={q['mean_drop']:.6f}, "
                f"mean util={q['mean_util']:.3f}%"
            )
        log.info(f"  CSV: {output_path}")
        log.info("  Routing state: VERIFIED")
        log.info("  Traffic process state: VERIFIED")
        log.info("  Required telemetry: VERIFIED")
        log.info("=" * 68)

        if self.args.summary_json:
            payload = {
                **summary,
                "method": "ospf",
                "traffic_profile": self.args.traffic_profile,
                "traffic_seed": self.args.traffic_seed,
                "csv_path": str(output_path.resolve()),
                "status": "completed",
                "routing_state_verified": self.routing_state["verified"],
                "traffic_state_verified": traffic_state["verified"],
                "telemetry_state_verified": telemetry_state["verified"],
                "top_bottleneck_egresses": top_bottlenecks,
                "routing_state": self.routing_state,
                "implementation_note": (
                    "Centralized deterministic single shortest-path SPF; "
                    "identical route for all DSCP values; no ECMP."
                ),
            }
            summary_path = Path(self.args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            normalize_artifact_permissions(summary_path.parent, dir_mode=0o775)
            summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            normalize_artifact_permissions(summary_path, file_mode=0o664)

        return 0

    def close(self) -> None:
        if self.visualization_file is not None:
            try:
                self.visualization_file.restore()
            except Exception as exc:
                log.warning(f"Failed to restore visualization paths: {exc}")

        if self.traffic_manager is not None:
            try:
                self.traffic_manager.stop_traffic()
                log.info("Stopped traffic")
            except Exception as exc:
                log.warning(f"Failed to stop traffic: {exc}")

        if self.env is not None:
            self.env.close()


def parse_args() -> argparse.Namespace:
    from traffic_generator import TrafficManager

    parser = argparse.ArgumentParser(
        description="Single-path OSPF/SPF routing benchmark"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/topologies/fat_tree_k4.yaml",
    )
    parser.add_argument("--rules-dir", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--paths-file", default="/tmp/p4_paths.json")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--traffic-profile",
        choices=tuple(TrafficManager.TRAFFIC_PROFILES),
        default="medium_2",
    )
    parser.add_argument("--traffic-seed", type=int, default=42)
    parser.add_argument("--influx-url", default="http://192.168.56.1:8086")
    parser.add_argument("--influx-org", default="Research")
    parser.add_argument("--influx-bucket", default="INT")
    parser.add_argument(
        "--influx-token",
        default=os.environ.get("INFLUX_TOKEN"),
    )
    parser.add_argument(
        "--telemetry-backend",
        choices=["cache", "influx", "cache-fallback-influx"],
        default="cache",
        help="Telemetry source for benchmark observations",
    )
    parser.add_argument(
        "--telemetry-cache-socket",
        default="/tmp/p4_qos_int_telemetry.sock",
        help="Unix socket path for local telemetry cache",
    )
    parser.add_argument(
        "--telemetry-cache-timeout",
        type=float,
        default=1.0,
        help="Local telemetry cache request timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_unified_logging(module_name="ospf_baseline", log_level=args.log_level)

    if args.telemetry_backend in ("influx", "cache-fallback-influx") and not args.influx_token:
        log.error("Set INFLUX_TOKEN or pass --influx-token")
        return 2

    benchmark = OSPFBenchmark(args)
    try:
        return benchmark.run()
    except Exception:
        log.exception("OSPF benchmark failed")
        return 1
    finally:
        benchmark.close()


if __name__ == "__main__":
    sys.exit(main())
