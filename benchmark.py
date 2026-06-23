#!/usr/bin/env python3
"""
Reproducible RL vs ECMP vs OSPF routing benchmark orchestrator.

The independent experimental unit is one complete run. Traffic seeds are
paired across methods inside each (profile, repetition) block, while method
order is balanced over all permutations to reduce order and carry-over bias.
Confidence intervals and hypothesis tests are computed across independent
runs, never across highly autocorrelated per-second samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from logging_config import normalize_artifact_permissions
from traffic_generator import TrafficManager


METHODS = ("rl", "ecmp", "ospf")
QIDS = (0, 1, 7)

# Primary metrics selected before the experiment. Positive direction means
# larger is better; negative means smaller is better.
PRIMARY_METRICS = {
    "reward_mean_valid": 1,
    "sla_compliance_valid_pct": 1,
    "macro_latency_mean_ms": -1,
    "macro_drop_mean": -1,
    "valid_fraction": 1,
}

AGGREGATE_METRICS = [
    "reward_mean_valid",
    "sla_compliance_valid_pct",
    "valid_fraction",
    "macro_latency_mean_ms",
    "worst_queue_p95_latency_ms",
    "macro_drop_mean",
    "macro_util_mean_pct",
    "action_applied_rate",
]
for _qid in QIDS:
    AGGREGATE_METRICS.extend(
        [
            f"q{_qid}_latency_mean_ms",
            f"q{_qid}_latency_p95_ms",
            f"q{_qid}_drop_mean",
            f"q{_qid}_util_mean_pct",
        ]
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else math.nan


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float,
    resamples: int,
    seed: int,
) -> Tuple[float, float]:
    clean = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if clean.size == 0:
        return math.nan, math.nan
    if clean.size == 1:
        return float(clean[0]), float(clean[0])

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clean.size, size=(resamples, clean.size))
    bootstrap_means = clean[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(bootstrap_means, alpha)),
        float(np.quantile(bootstrap_means, 1.0 - alpha)),
    )


def paired_randomization_pvalue(
    differences: Sequence[float],
    seed: int,
    monte_carlo_samples: int = 50000,
) -> float:
    diffs = np.asarray(
        [value for value in differences if math.isfinite(value)],
        dtype=float,
    )
    if diffs.size == 0:
        return math.nan
    if np.allclose(diffs, 0.0):
        return 1.0

    observed = abs(float(diffs.mean()))
    n = int(diffs.size)
    tolerance = 1e-12

    if n <= 18:
        extreme = 0
        total = 1 << n
        for mask in range(total):
            signs = np.fromiter(
                (1.0 if mask & (1 << i) else -1.0 for i in range(n)),
                dtype=float,
                count=n,
            )
            if abs(float(np.mean(diffs * signs))) + tolerance >= observed:
                extreme += 1
        return extreme / total

    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = monte_carlo_samples
    batch_size = 2000
    while remaining > 0:
        batch = min(batch_size, remaining)
        signs = rng.choice((-1.0, 1.0), size=(batch, n))
        means = np.abs((signs * diffs).mean(axis=1))
        extreme += int(np.sum(means + tolerance >= observed))
        remaining -= batch
    return (extreme + 1) / (monte_carlo_samples + 1)


def holm_adjust(rows: List[Dict], p_key: str = "p_value") -> None:
    indexed = [
        (index, row[p_key])
        for index, row in enumerate(rows)
        if isinstance(row.get(p_key), (int, float))
        and math.isfinite(float(row[p_key]))
    ]
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running_max = 0.0
    for rank, (index, p_value) in enumerate(indexed):
        adjusted = min(1.0, (m - rank) * float(p_value))
        running_max = max(running_max, adjusted)
        rows[index]["p_value_holm"] = running_max


def clean_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def summarize_run_csv(path: Path) -> Dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No measurement rows in {path}")

    valid_rows = [
        row for row in rows if int(clean_float(row.get("data_valid", "0"))) == 1
    ]
    analysis_rows = valid_rows

    def values(column: str, source: Sequence[Dict] = analysis_rows) -> List[float]:
        return [
            clean_float(row.get(column))
            for row in source
            if math.isfinite(clean_float(row.get(column)))
        ]

    sla_denominator = float(len(QIDS))
    summary = {
        "steps": len(rows),
        "valid_steps": len(valid_rows),
        "valid_fraction": len(valid_rows) / len(rows),
        "reward_mean_all": mean(values("reward", rows)),
        "reward_mean_valid": mean(values("reward")),
        "sla_compliance_all_pct": (
            100.0 * sum(values("sla_met_count", rows)) / (sla_denominator * len(rows))
        ),
        "sla_compliance_valid_pct": (
            100.0 * sum(values("sla_met_count")) / (sla_denominator * len(valid_rows))
            if valid_rows
            else math.nan
        ),
        "offered_load_mean_mbps": mean(
            [
                clean_float(row.get("load_q0_mbps"))
                + clean_float(row.get("load_q1_mbps"))
                + clean_float(row.get("load_q7_mbps"))
                for row in rows
            ]
        ),
        "action_applied_rate": mean(values("action_applied", rows))
        if "action_applied" in rows[0]
        else 0.0,
    }

    queue_latency_means = []
    queue_latency_p95 = []
    queue_drop_means = []
    queue_util_means = []
    for qid in QIDS:
        latency = values(f"q{qid}_latency_ms")
        drops = values(f"q{qid}_drop")
        utilization = values(f"q{qid}_util_pct")
        q_latency_mean = mean(latency)
        q_latency_p95 = percentile(latency, 95)
        q_drop_mean = mean(drops)
        q_util_mean = mean(utilization)
        summary.update(
            {
                f"q{qid}_latency_mean_ms": q_latency_mean,
                f"q{qid}_latency_p95_ms": q_latency_p95,
                f"q{qid}_drop_mean": q_drop_mean,
                f"q{qid}_util_mean_pct": q_util_mean,
            }
        )
        queue_latency_means.append(q_latency_mean)
        queue_latency_p95.append(q_latency_p95)
        queue_drop_means.append(q_drop_mean)
        queue_util_means.append(q_util_mean)

    summary["macro_latency_mean_ms"] = mean(
        [value for value in queue_latency_means if math.isfinite(value)]
    )
    summary["worst_queue_p95_latency_ms"] = max(
        (value for value in queue_latency_p95 if math.isfinite(value)),
        default=math.nan,
    )
    summary["macro_drop_mean"] = mean(
        [value for value in queue_drop_means if math.isfinite(value)]
    )
    summary["macro_util_mean_pct"] = mean(
        [value for value in queue_util_means if math.isfinite(value)]
    )
    return summary


def validate_runner_summary(item: Dict, path: Path) -> Tuple[Dict, List[str]]:
    """Validate that a runner completed the requested routing treatment."""
    if not path.exists():
        return {}, [f"runner summary missing: {path}"]

    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {}, [f"runner summary unreadable: {exc}"]

    errors = []
    if payload.get("status") != "completed":
        errors.append(f"runner status={payload.get('status')!r}")
    if payload.get("method") != item["method"]:
        errors.append(
            f"runner method={payload.get('method')!r}, "
            f"expected {item['method']!r}"
        )
    if payload.get("traffic_profile") != item["profile"]:
        errors.append(
            f"runner profile={payload.get('traffic_profile')!r}, "
            f"expected {item['profile']!r}"
        )
    try:
        summary_seed = int(payload.get("traffic_seed"))
    except (TypeError, ValueError):
        summary_seed = None
    if summary_seed != int(item["seed"]):
        errors.append(
            f"runner seed={payload.get('traffic_seed')!r}, "
            f"expected {item['seed']}"
        )
    if payload.get("routing_state_verified") is not True:
        errors.append("routing_state_verified is not true")
    if payload.get("traffic_state_verified") is not True:
        errors.append("traffic_state_verified is not true")
    if payload.get("telemetry_state_verified") is not True:
        errors.append("telemetry_state_verified is not true")

    routing_state = payload.get("routing_state")
    if not isinstance(routing_state, dict) or routing_state.get("verified") is not True:
        errors.append("routing_state evidence is missing or unverified")
    if item["method"] == "ecmp" and not (
        isinstance(routing_state, dict) and routing_state.get("plan_sha256")
    ):
        errors.append("ECMP route-plan digest is missing")
    if item["method"] == "ecmp":
        audit_required = payload.get("ecmp_path_audit_required")
        audit_verified = payload.get("ecmp_path_audit_verified")
        if audit_required is True and audit_verified is not True:
            errors.append(
                "ECMP INT-observed path audit is required but not verified"
            )
        elif audit_verified is False:
            errors.append("ECMP INT-observed path audit failed")

    return payload, errors


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_artifact_permissions(path.parent, dir_mode=0o775)
    with path.open("w", newline="") as handle:
        normalize_artifact_permissions(path, file_mode=0o664)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class BenchmarkOrchestrator:
    def __init__(self, args):
        self.args = args
        self.output_dir = self._resolve_output_dir()
        self.current_process: Optional[subprocess.Popen] = None
        self.interrupted = False
        self.interrupt_signal: Optional[int] = None
        self._signal_count = 0
        self.run_rows: List[Dict] = []
        self.schedule: List[Dict] = []
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _resolve_output_dir(self) -> Path:
        if self.args.output_dir:
            return Path(self.args.output_dir).resolve()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("benchmark_results") / f"benchmark_{timestamp}"

    def _handle_signal(self, signum, _frame):
        self.interrupted = True
        self.interrupt_signal = signum
        self._signal_count += 1
        if self._signal_count == 1:
            print(
                f"\nReceived signal {signum}; terminating the entire benchmark...",
                flush=True,
            )
            child_signal = signal.SIGTERM
        else:
            child_signal = signal.SIGKILL
        if self.current_process and self.current_process.poll() is None:
            try:
                os.killpg(
                    os.getpgid(self.current_process.pid),
                    child_signal,
                )
            except ProcessLookupError:
                pass

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Sleep while remaining responsive to a benchmark-wide interrupt."""
        deadline = time.monotonic() + max(0.0, float(seconds))
        while not self.interrupted:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.2, remaining))
        return False

    @staticmethod
    def _kill_process_group(
        process: subprocess.Popen,
        sig: signal.Signals,
    ) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except ProcessLookupError:
            pass

    def _stop_active_process(
        self,
        process: subprocess.Popen,
        grace_seconds: float = 2.0,
    ) -> int:
        """Terminate one runner, escalating to SIGKILL after a short grace."""
        self._kill_process_group(process, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            self._kill_process_group(process, signal.SIGKILL)
        try:
            return process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._kill_process_group(process, signal.SIGKILL)
            return process.wait()

    def preflight(self) -> None:
        if os.geteuid() != 0:
            raise RuntimeError("Run the benchmark through `make benchmark` (sudo -E)")
        needs_influx = (
            self.args.telemetry_backend in ("influx", "cache-fallback-influx")
            or self.args.production_influx_write == "on"
        )
        if needs_influx and not os.environ.get("INFLUX_TOKEN"):
            raise RuntimeError("INFLUX_TOKEN is not set")

        config = Path(self.args.config)
        if not config.exists():
            raise RuntimeError(f"Topology config not found: {config}")
        if not Path("/tmp/topology.json").exists() and not Path("topology.json").exists():
            raise RuntimeError("Running Mininet topology was not found")
        active_topology = Path(".active_topology")
        if active_topology.exists():
            active = Path(active_topology.read_text().strip()).resolve()
            requested = Path(self.args.config).resolve()
            if active != requested:
                raise RuntimeError(
                    f"Running topology is {active}, but benchmark requested {requested}"
                )

        available_profiles = set(TrafficManager.TRAFFIC_PROFILES)
        unknown = sorted(set(self.args.profiles) - available_profiles)
        if unknown:
            raise RuntimeError(f"Unknown traffic profiles: {unknown}")
        if "high_2" in self.args.profiles and not self.args.allow_high_2:
            raise RuntimeError(
                "high_2 is disabled by project guidance; use --allow-high-2 "
                "only if you intentionally accept CPU-overload risk"
            )
        if self.args.repetitions < 2:
            print(
                "WARNING: fewer than 2 repetitions cannot estimate variance.",
                flush=True,
            )
        elif self.args.repetitions < 6:
            print(
                "WARNING: use at least 6 repetitions for a complete balanced "
                "method-order cycle; 12+ is recommended for paper results.",
                flush=True,
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "runs").mkdir(exist_ok=True)
        normalize_artifact_permissions(self.output_dir, dir_mode=0o775)
        normalize_artifact_permissions(self.output_dir / "runs", dir_mode=0o775)

    def build_schedule(self) -> None:
        permutations = list(itertools.permutations(METHODS))
        rng = random.Random(self.args.schedule_seed)
        rng.shuffle(permutations)

        schedule = []
        sequence = 0
        for profile_index, profile in enumerate(self.args.profiles):
            profile_offset = (profile_index * self.args.repetitions) % len(permutations)
            for repetition in range(1, self.args.repetitions + 1):
                # Keep methods paired within a block while making repetitions
                # from different traffic profiles use distinct random streams.
                seed = (
                    self.args.base_seed
                    + profile_index * 1000
                    + repetition
                    - 1
                )
                order = permutations[
                    (profile_offset + repetition - 1) % len(permutations)
                ]
                for order_index, method in enumerate(order, 1):
                    sequence += 1
                    schedule.append(
                        {
                            "sequence": sequence,
                            "profile": profile,
                            "repetition": repetition,
                            "seed": seed,
                            "order_index": order_index,
                            "method": method,
                            "block_order": list(order),
                        }
                    )
        self.schedule = schedule

    def _checkpoint_path(self) -> Optional[Path]:
        save_dir = Path(self.args.save_dir)
        matches = sorted(save_dir.glob(f"*-dqn_v4_{self.args.weights_tag}.pth"))
        if matches:
            return matches[-1]
        legacy = save_dir / f"dqn_v4_{self.args.weights_tag}.pth"
        return legacy if legacy.exists() else None

    def write_manifest(self) -> None:
        checkpoint = self._checkpoint_path()
        if checkpoint is None:
            raise RuntimeError(
                f"No RL checkpoint for tag {self.args.weights_tag!r} "
                f"in {self.args.save_dir}"
            )

        manifest = {
            "created_at_utc": utc_now(),
            "status": "planned",
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_status_short": command_output(["git", "status", "--short"]),
            "config": str(Path(self.args.config).resolve()),
            "config_sha256": file_sha256(Path(self.args.config)),
            "p4_source_sha256": file_sha256(Path("p4src/int_md.p4")),
            "runtime_topology_sha256": file_sha256(Path("/tmp/topology.json")),
            "rl_checkpoint": str(checkpoint.resolve()),
            "rl_checkpoint_sha256": file_sha256(checkpoint),
            "source_artifact_hashes": {
                path: file_sha256(Path(path))
                for path in (
                    "benchmark.py",
                    "ospf_baseline.py",
                    "ecmp_baseline.py",
                    "rl_production.py",
                    "rl_agent_4.py",
                    "controller.py",
                    "traffic_generator.py",
                    "p4src/include/forward.p4",
                )
            },
            "tool_versions": {
                "p4c": command_output(["p4c-bm2-ss", "--version"]),
                "simple_switch": command_output(["simple_switch", "--version"]),
                "numpy": np.__version__,
            },
            "arguments": vars(self.args),
            "methods": {
                "rl": "Greedy trained DQN policy with queue-specific rerouting.",
                "ecmp": (
                    "Queue-independent CRC16 hash over source/destination IP "
                    "and per-switch group ID across equal-cost next hops."
                ),
                "ospf": (
                    "Centralized steady-state single shortest-path SPF model; "
                    "identical forwarding path for all DSCP values; no ECMP."
                ),
            },
            "experimental_design": {
                "unit": "one complete run",
                "routing_cost_model": (
                    "dimensionless inverse bandwidth, normalized so the "
                    "fastest configured link has cost 100"
                ),
                "pairing": "same profile and traffic seed across all methods",
                "order_control": "balanced deterministic permutations",
                "warmup_excluded_seconds": self.args.warmup_seconds,
                "measurement_settle_excluded_seconds": (
                    TrafficManager.MEASUREMENT_SETTLE_SECONDS
                ),
                "inter_run_cooldown_seconds": self.args.cooldown_seconds,
                "confidence_interval": (
                    f"{self.args.confidence * 100:.1f}% percentile bootstrap "
                    f"over independent runs ({self.args.bootstrap_resamples} resamples)"
                ),
                "paired_test": (
                    "two-sided paired sign-flip randomization test; "
                    "Holm family-wise correction"
                ),
                "invalid_data_policy": (
                    "raw rows and failed attempts retained; primary metrics "
                    "use rows marked data_valid; runs also require verified "
                    "routing state and exact traffic process counts"
                ),
                "treatment_integrity": (
                    "P4 tables are cleared and read back before measurement; "
                    "ECMP entries are read back at start and end; traffic "
                    "process counts and all-flow telemetry coverage are "
                    "verified at start and end; traffic restarts invalidate runs"
                ),
            },
            "schedule": self.schedule,
            "references": [
                "RFC 2328 OSPF Version 2",
                "RFC 2992 Analysis of an Equal-Cost Multi-Path Algorithm",
                "ACM Artifact Review and Badging Version 1.1",
            ],
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        normalize_artifact_permissions(manifest_path, file_mode=0o664)
        environment_lines = [
            f"created_at_utc={manifest['created_at_utc']}",
            f"hostname={manifest['hostname']}",
            f"platform={manifest['platform']}",
            f"python={manifest['python']}",
            f"git_commit={manifest['git_commit']}",
            f"p4c={manifest['tool_versions']['p4c']}",
            f"simple_switch={manifest['tool_versions']['simple_switch']}",
            f"numpy={manifest['tool_versions']['numpy']}",
            "",
            "pip_freeze:",
            command_output([sys.executable, "-m", "pip", "freeze"]) or "unavailable",
        ]
        environment_path = self.output_dir / "environment.txt"
        environment_path.write_text("\n".join(environment_lines) + "\n")
        normalize_artifact_permissions(environment_path, file_mode=0o664)

    def _clean_traffic(self) -> None:
        commands = [
            ["pkill", "-9", "iperf3"],
            ["pkill", "-9", "-f", TrafficManager.TRAFFIC_TAG],
        ]
        for command in commands:
            subprocess.run(command, capture_output=True, check=False)

    def _run_paths(self, item: Dict) -> Dict[str, Path]:
        stem = (
            f"{item['profile']}__rep{item['repetition']:02d}"
            f"__seed{item['seed']}__order{item['order_index']}"
            f"__{item['method']}"
        )
        run_dir = self.output_dir / "runs" / item["profile"] / (
            f"rep_{item['repetition']:02d}_seed_{item['seed']}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        normalize_artifact_permissions(run_dir, dir_mode=0o775)
        return {
            "dir": run_dir,
            "csv": run_dir / f"{stem}.csv",
            "summary": run_dir / f"{stem}.summary.json",
            "result": run_dir / f"{stem}.result.json",
            "log_base": run_dir / stem,
        }

    def _command_for(self, item: Dict, paths: Dict[str, Path]) -> List[str]:
        common = [
            sys.executable,
            {
                "rl": "rl_production.py",
                "ecmp": "ecmp_baseline.py",
                "ospf": "ospf_baseline.py",
            }[item["method"]],
            "--config",
            self.args.config,
            "--steps",
            str(self.args.steps),
            "--traffic-profile",
            item["profile"],
            "--traffic-seed",
            str(item["seed"]),
            "--warmup-seconds",
            str(self.args.warmup_seconds),
            "--log-every",
            "0",
            "--output",
            str(paths["csv"]),
            "--summary-json",
            str(paths["summary"]),
            "--log-level",
            self.args.log_level,
            "--telemetry-backend",
            self.args.telemetry_backend,
            "--telemetry-cache-socket",
            self.args.telemetry_cache_socket,
            "--telemetry-cache-timeout",
            str(self.args.telemetry_cache_timeout),
        ]
        if item["method"] == "rl":
            common.extend(
                [
                    "--generate-traffic",
                    "--save-dir",
                    self.args.save_dir,
                    "--weights-tag",
                    self.args.weights_tag,
                    "--production-influx-write",
                    self.args.production_influx_write,
                ]
            )
        return common

    def _execute_attempt(
        self,
        command: Sequence[str],
        log_path: Path,
    ) -> Tuple[int, float]:
        start = time.monotonic()
        timeout_deadline = (
            start + self.args.run_timeout_seconds
            if self.args.run_timeout_seconds
            else None
        )
        with log_path.open("w") as log_file:
            normalize_artifact_permissions(log_path, file_mode=0o664)
            log_file.write("COMMAND: " + " ".join(command) + "\n\n")
            log_file.flush()
            self.current_process = subprocess.Popen(
                list(command),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
                text=True,
            )
            try:
                while True:
                    if self.interrupted:
                        self._stop_active_process(self.current_process)
                        return_code = 130
                        break

                    if (
                        timeout_deadline is not None
                        and time.monotonic() >= timeout_deadline
                    ):
                        return_code = self._stop_active_process(
                            self.current_process,
                            grace_seconds=20.0,
                        )
                        break

                    try:
                        child_return_code = self.current_process.wait(timeout=0.25)
                        return_code = (
                            130 if self.interrupted else child_return_code
                        )
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                self.current_process = None
        return return_code, time.monotonic() - start

    def run_one(self, item: Dict) -> Dict:
        paths = self._run_paths(item)
        if self.args.resume and paths["result"].exists():
            previous = json.loads(paths["result"].read_text())
            previous_summary_path = previous.get("runner_summary_path")
            previous_summary = (
                Path(previous_summary_path)
                if previous_summary_path
                else paths["summary"]
            )
            _, resume_errors = validate_runner_summary(
                item,
                previous_summary,
            )
            if previous.get("status") == "success" and not resume_errors:
                print(
                    f"[{item['sequence']:03d}/{len(self.schedule)}] "
                    f"resume {item['profile']} rep={item['repetition']} "
                    f"{item['method']}",
                    flush=True,
                )
                return previous
            print(
                f"[{item['sequence']:03d}/{len(self.schedule)}] "
                f"rerun unverified result {item['profile']} "
                f"rep={item['repetition']} {item['method']}: "
                + "; ".join(resume_errors or ["result status is not success"]),
                flush=True,
            )

        self._clean_traffic()
        if self.args.cooldown_seconds > 0 and not self._interruptible_sleep(
            self.args.cooldown_seconds
        ):
            result = {
                **item,
                "status": "interrupted",
                "result_path": str(paths["result"].resolve()),
                "attempts": [],
                "completed_at_utc": utc_now(),
                "interrupt_signal": self.interrupt_signal,
            }
            paths["result"].write_text(
                json.dumps(result, indent=2, sort_keys=True)
            )
            normalize_artifact_permissions(paths["result"], file_mode=0o664)
            self._clean_traffic()
            return result

        command = self._command_for(item, paths)
        attempts = []
        final_status = "failed"
        final_summary = {}
        final_csv_path = None
        final_runner_summary_path = None

        for attempt in range(1, self.args.max_retries + 2):
            if self.interrupted:
                final_status = "interrupted"
                break
            print(
                f"[{item['sequence']:03d}/{len(self.schedule)}] "
                f"profile={item['profile']} rep={item['repetition']}/"
                f"{self.args.repetitions} seed={item['seed']} "
                f"method={item['method']} order={item['order_index']} "
                f"attempt={attempt}",
                flush=True,
            )
            log_path = Path(f"{paths['log_base']}.attempt{attempt}.log")
            for transient_path in (paths["csv"], paths["summary"]):
                try:
                    transient_path.unlink()
                except FileNotFoundError:
                    pass
            return_code, runtime = self._execute_attempt(command, log_path)
            attempt_csv_path = Path(f"{paths['log_base']}.attempt{attempt}.csv")
            attempt_summary_path = Path(
                f"{paths['log_base']}.attempt{attempt}.summary.json"
            )
            if paths["csv"].exists():
                shutil.copy2(paths["csv"], attempt_csv_path)
            if paths["summary"].exists():
                shutil.copy2(paths["summary"], attempt_summary_path)
            attempt_row = {
                "attempt": attempt,
                "return_code": return_code,
                "runtime_seconds": runtime,
                "log_path": str(log_path.resolve()),
                "csv_path": str(attempt_csv_path.resolve())
                if attempt_csv_path.exists()
                else None,
                "runner_summary_path": str(attempt_summary_path.resolve())
                if attempt_summary_path.exists()
                else None,
                "finished_at_utc": utc_now(),
            }
            attempts.append(attempt_row)

            if self.interrupted or return_code == 130:
                attempt_row["interrupted"] = True
                attempt_row["interrupt_signal"] = self.interrupt_signal
                final_status = "interrupted"
                break

            try:
                if return_code == 0 and attempt_csv_path.exists():
                    final_summary = summarize_run_csv(attempt_csv_path)
                    runner_payload, runner_errors = validate_runner_summary(
                        item,
                        attempt_summary_path,
                    )
                    final_summary["routing_state_verified"] = bool(
                        runner_payload.get("routing_state_verified", False)
                    )
                    final_summary["traffic_state_verified"] = bool(
                        runner_payload.get("traffic_state_verified", False)
                    )
                    final_summary["telemetry_state_verified"] = bool(
                        runner_payload.get("telemetry_state_verified", False)
                    )
                    routing_state = runner_payload.get("routing_state", {})
                    final_summary["route_plan_sha256"] = (
                        routing_state.get("plan_sha256")
                        if isinstance(routing_state, dict)
                        else None
                    )
                    top_bottlenecks = runner_payload.get(
                        "top_bottleneck_egresses"
                    )
                    if not top_bottlenecks and isinstance(routing_state, dict):
                        top_bottlenecks = routing_state.get(
                            "top_bottleneck_egresses"
                        )
                    if top_bottlenecks:
                        top = top_bottlenecks[0]
                        final_summary["top_bottleneck_egress"] = top.get(
                            "label"
                        )
                        final_summary["top_bottleneck_p95_util"] = top.get(
                            "p95_util"
                        )
                        final_summary["top_bottleneck_mean_util"] = top.get(
                            "mean_util"
                        )
                        final_summary["top_bottleneck_max_util"] = top.get(
                            "max_util"
                        )
                        final_summary["top_bottleneck_samples"] = top.get(
                            "count"
                        )
                        final_summary["top_bottleneck_flow_count"] = top.get(
                            "flow_count"
                        )
                    final_summary["top_bottleneck_count"] = len(
                        top_bottlenecks or []
                    )
                    if item["method"] == "ecmp":
                        audit = (
                            routing_state.get("ecmp_observed_path_audit", {})
                            if isinstance(routing_state, dict)
                            else {}
                        )
                        final_summary["ecmp_path_audit_verified"] = (
                            runner_payload.get("ecmp_path_audit_verified")
                        )
                        final_summary["ecmp_path_audit_required"] = (
                            runner_payload.get("ecmp_path_audit_required")
                        )
                        final_summary["ecmp_path_mismatch_count"] = (
                            runner_payload.get(
                                "ecmp_path_mismatch_count",
                                audit.get("mismatch_count"),
                            )
                        )
                        final_summary[
                            "ecmp_queue_independence_violations"
                        ] = runner_payload.get(
                            "ecmp_queue_independence_violations",
                            audit.get("queue_independence_violations"),
                        )

                    quality_errors = list(runner_errors)
                    if (
                        final_summary["valid_fraction"]
                        < self.args.min_valid_fraction
                    ):
                        quality_errors.append(
                            f"valid_fraction={final_summary['valid_fraction']:.3f} "
                            f"< {self.args.min_valid_fraction:.3f}"
                        )

                    if not quality_errors:
                        final_status = "success"
                        final_csv_path = attempt_csv_path
                        final_runner_summary_path = (
                            attempt_summary_path
                            if attempt_summary_path.exists()
                            else None
                        )
                        break
                    attempt_row["quality_error"] = "; ".join(quality_errors)
            except Exception as exc:
                attempt_row["parse_error"] = str(exc)

            self._clean_traffic()
            if attempt <= self.args.max_retries:
                if not self._interruptible_sleep(self.args.cooldown_seconds):
                    final_status = "interrupted"
                    break

        fallback_csv_path = (
            Path(f"{paths['log_base']}.attempt{len(attempts)}.csv")
            if attempts
            else None
        )
        result_csv_path = final_csv_path or fallback_csv_path
        result = {
            **item,
            **final_summary,
            "status": final_status,
            "csv_path": (
                str(result_csv_path.resolve())
                if result_csv_path is not None
                else None
            ),
            "runner_summary_path": str(final_runner_summary_path.resolve())
            if final_runner_summary_path
            else None,
            "result_path": str(paths["result"].resolve()),
            "attempts": attempts,
            "completed_at_utc": utc_now(),
        }
        if final_status == "interrupted":
            result["interrupt_signal"] = self.interrupt_signal
        paths["result"].write_text(json.dumps(result, indent=2, sort_keys=True))
        normalize_artifact_permissions(paths["result"], file_mode=0o664)
        self._clean_traffic()
        return result

    def aggregate(self, successful: Sequence[Dict]) -> List[Dict]:
        groups = defaultdict(list)
        for row in successful:
            groups[(row["profile"], row["method"])].append(row)

        output = []
        for (profile, method), rows in sorted(groups.items()):
            for metric in AGGREGATE_METRICS:
                values = [
                    float(row[metric])
                    for row in rows
                    if metric in row and math.isfinite(float(row[metric]))
                ]
                low, high = bootstrap_mean_ci(
                    values,
                    self.args.confidence,
                    self.args.bootstrap_resamples,
                    self.args.analysis_seed
                    + sum(ord(char) for char in f"{profile}:{method}:{metric}"),
                )
                output.append(
                    {
                        "profile": profile,
                        "method": method,
                        "metric": metric,
                        "n_runs": len(values),
                        "mean": mean(values),
                        "sample_std": sample_std(values),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
        return output

    def paired_comparisons(self, successful: Sequence[Dict]) -> List[Dict]:
        by_key = {
            (
                row["profile"],
                int(row["repetition"]),
                row["method"],
            ): row
            for row in successful
        }
        comparisons = []

        for profile in self.args.profiles:
            for baseline in ("ospf", "ecmp"):
                for metric, direction in PRIMARY_METRICS.items():
                    pairs = []
                    for repetition in range(1, self.args.repetitions + 1):
                        rl = by_key.get((profile, repetition, "rl"))
                        base = by_key.get((profile, repetition, baseline))
                        if not rl or not base:
                            continue
                        if int(rl["seed"]) != int(base["seed"]):
                            raise AssertionError(
                                f"Unpaired seeds for {profile} repetition "
                                f"{repetition}: RL={rl['seed']} "
                                f"{baseline}={base['seed']}"
                            )
                        load_difference = abs(
                            float(rl["offered_load_mean_mbps"])
                            - float(base["offered_load_mean_mbps"])
                        )
                        if load_difference > 1e-6:
                            raise AssertionError(
                                f"Paired offered-load mismatch for {profile} "
                                f"repetition {repetition}: {load_difference} Mbps"
                            )
                        rl_value = float(rl.get(metric, math.nan))
                        baseline_value = float(base.get(metric, math.nan))
                        if math.isfinite(rl_value) and math.isfinite(baseline_value):
                            pairs.append((rl_value, baseline_value))

                    differences = [rl - base for rl, base in pairs]
                    directed = [direction * diff for diff in differences]
                    baselines = [base for _, base in pairs]
                    diff_low, diff_high = bootstrap_mean_ci(
                        differences,
                        self.args.confidence,
                        self.args.bootstrap_resamples,
                        self.args.analysis_seed
                        + sum(ord(char) for char in f"{profile}:{baseline}:{metric}"),
                    )
                    average_difference = mean(differences)
                    baseline_mean = mean(baselines)
                    relative_improvement = (
                        direction * average_difference / abs(baseline_mean) * 100.0
                        if math.isfinite(baseline_mean) and baseline_mean != 0
                        else math.nan
                    )
                    standard_deviation = sample_std(differences)
                    effect_size = (
                        average_difference / standard_deviation
                        if standard_deviation > 0
                        else (math.inf if average_difference != 0 else 0.0)
                    )
                    comparisons.append(
                        {
                            "profile": profile,
                            "comparison": f"rl_vs_{baseline}",
                            "baseline": baseline,
                            "metric": metric,
                            "direction": "higher_better"
                            if direction > 0
                            else "lower_better",
                            "n_pairs": len(pairs),
                            "rl_mean": mean([rl for rl, _ in pairs]),
                            "baseline_mean": baseline_mean,
                            "mean_difference_rl_minus_baseline": average_difference,
                            "difference_ci_low": diff_low,
                            "difference_ci_high": diff_high,
                            "relative_improvement_pct": relative_improvement,
                            "paired_cohens_dz": effect_size,
                            "win_rate": (
                                sum(value > 0 for value in directed) / len(directed)
                                if directed
                                else math.nan
                            ),
                            "p_value": paired_randomization_pvalue(
                                differences,
                                self.args.analysis_seed
                                + sum(
                                    ord(char)
                                    for char in f"p:{profile}:{baseline}:{metric}"
                                ),
                            ),
                            "p_value_holm": math.nan,
                        }
                    )

        holm_adjust(comparisons)
        return comparisons

    @staticmethod
    def _aggregate_lookup(rows: Sequence[Dict]) -> Dict[Tuple[str, str, str], Dict]:
        return {
            (row["profile"], row["method"], row["metric"]): row for row in rows
        }

    def render_summary(
        self,
        aggregate_rows: Sequence[Dict],
        comparison_rows: Sequence[Dict],
        all_results: Sequence[Dict],
    ) -> str:
        def _audit_text(value) -> str:
            if value is True:
                return "yes"
            if value is False:
                return "NO"
            return "n/a"

        def _count_text(value) -> str:
            try:
                if value is None or (
                    isinstance(value, float) and not math.isfinite(value)
                ):
                    return "n/a"
                return str(int(value))
            except (TypeError, ValueError):
                return "n/a"

        def _percent_text(value) -> str:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return "n/a"
            if not math.isfinite(value):
                return "n/a"
            return f"{value:.2f}%"

        aggregate = self._aggregate_lookup(aggregate_rows)
        successful = sum(row.get("status") == "success" for row in all_results)
        interrupted = sum(
            row.get("status") == "interrupted" for row in all_results
        )
        failed = len(all_results) - successful - interrupted
        lines = [
            "=" * 108,
            "ROUTING BENCHMARK SUMMARY",
            "=" * 108,
            f"Output directory: {self.output_dir.resolve()}",
            (
                f"Completed runs: {successful}/{len(all_results)}; "
                f"interrupted: {interrupted}; failed/invalid: {failed}"
            ),
            (
                f"Design: {self.args.repetitions} paired repetitions/profile, "
                f"{self.args.steps} measured steps/run, "
                f"{self.args.warmup_seconds:.1f}s warm-up, "
                f"{TrafficManager.MEASUREMENT_SETTLE_SECONDS:.1f}s "
                "measurement settle, "
                f"{self.args.cooldown_seconds:.1f}s inter-run cooldown"
            ),
            "",
        ]

        for profile in self.args.profiles:
            lines.append(f"Profile: {profile}")
            lines.append("  Aggregate across independent runs:")
            confidence_label = f"{self.args.confidence * 100:.1f}% CI"
            lines.append(
                f"  Method  n    SLA valid % [{confidence_label}]       "
                f"Macro latency ms [{confidence_label}]    "
                f"Mean drops/100ms [{confidence_label}] "
                f"Mean reward [{confidence_label}]"
            )
            for method in METHODS:
                def item(metric: str) -> Dict:
                    return aggregate.get(
                        (profile, method, metric),
                        {
                            "n_runs": 0,
                            "mean": math.nan,
                            "ci_low": math.nan,
                            "ci_high": math.nan,
                        },
                    )

                sla = item("sla_compliance_valid_pct")
                latency = item("macro_latency_mean_ms")
                drop = item("macro_drop_mean")
                reward = item("reward_mean_valid")
                lines.append(
                    f"  {method.upper():<6} {sla['n_runs']:>2}   "
                    f"{sla['mean']:>7.2f} [{sla['ci_low']:>7.2f}, {sla['ci_high']:>7.2f}]   "
                    f"{latency['mean']:>8.3f} [{latency['ci_low']:>8.3f}, {latency['ci_high']:>8.3f}]   "
                    f"{drop['mean']:>9.6f} [{drop['ci_low']:>9.6f}, {drop['ci_high']:>9.6f}]   "
                    f"{reward['mean']:>8.4f} [{reward['ci_low']:>8.4f}, {reward['ci_high']:>8.4f}]"
                )

            lines.append("  ECMP independent-run audit:")
            ecmp_runs = sorted(
                (
                    row
                    for row in all_results
                    if row.get("profile") == profile
                    and row.get("method") == "ecmp"
                ),
                key=lambda row: int(row.get("repetition", 0)),
            )
            if not ecmp_runs:
                lines.append("    none")
            for row in ecmp_runs:
                verified = (
                    "yes"
                    if row.get("routing_state_verified") is True
                    else "NO"
                )
                traffic_verified = (
                    "yes"
                    if row.get("traffic_state_verified") is True
                    else "NO"
                )
                telemetry_verified = (
                    "yes"
                    if row.get("telemetry_state_verified") is True
                    else "NO"
                )
                ecmp_path_audit = _audit_text(
                    row.get("ecmp_path_audit_verified")
                )
                lines.append(
                    f"    rep={int(row.get('repetition', 0)):>2} "
                    f"seed={int(row.get('seed', 0)):>5} "
                    f"order={int(row.get('order_index', 0))} "
                    f"status={row.get('status', 'unknown'):<7} "
                    f"routing_verified={verified:<3} "
                    f"traffic_verified={traffic_verified:<3} "
                    f"telemetry_verified={telemetry_verified:<3} "
                    f"SLA={float(row.get('sla_compliance_valid_pct', math.nan)):>7.2f}% "
                    f"reward={float(row.get('reward_mean_valid', math.nan)):>8.4f} "
                    f"ecmp_path={ecmp_path_audit:<3} "
                    f"mismatches={_count_text(row.get('ecmp_path_mismatch_count'))} "
                    f"qsplit={_count_text(row.get('ecmp_queue_independence_violations'))}"
                )
                if row.get("top_bottleneck_egress"):
                    lines.append(
                        f"      top_bottleneck={row.get('top_bottleneck_egress')} "
                        f"p95_util={_percent_text(row.get('top_bottleneck_p95_util'))} "
                        f"flows={_count_text(row.get('top_bottleneck_flow_count'))}"
                    )
            lines.append("  Paired RL comparisons (positive improvement favors RL):")
            for baseline in ("ospf", "ecmp"):
                selected = [
                    row
                    for row in comparison_rows
                    if row["profile"] == profile
                    and row["baseline"] == baseline
                    and row["metric"]
                    in (
                        "sla_compliance_valid_pct",
                        "macro_latency_mean_ms",
                        "macro_drop_mean",
                    )
                ]
                for row in selected:
                    lines.append(
                        f"    RL vs {baseline.upper():<4} {row['metric']:<30} "
                        f"n={row['n_pairs']:>2} "
                        f"improvement={row['relative_improvement_pct']:>8.2f}% "
                        f"delta={row['mean_difference_rl_minus_baseline']:>10.4f} "
                        f"CI=[{row['difference_ci_low']:>9.4f}, {row['difference_ci_high']:>9.4f}] "
                        f"p_holm={row['p_value_holm']:.4f} "
                        f"dz={row['paired_cohens_dz']:.3f}"
                    )
            lines.append("")

        lines.extend(
            [
                "Artifacts:",
                "  run_results.csv          one row per independent run",
                "  aggregate_results.csv    means, sample SD, bootstrap confidence intervals",
                "  paired_comparisons.csv   paired effects, effect sizes, corrected p-values",
                "  manifest.json            code/config/checkpoint hashes and full schedule",
                "  environment.txt          tool versions and Python package snapshot",
                "  runs/                    raw per-step CSV, logs, and runner summaries",
                "=" * 108,
            ]
        )
        return "\n".join(lines)

    def analyze_and_write(self, results: Sequence[Dict]) -> None:
        result_fields = sorted({key for row in results for key in row if key != "attempts"})
        write_csv(self.output_dir / "run_results.csv", results, result_fields)

        successful = [row for row in results if row.get("status") == "success"]
        aggregate_rows = self.aggregate(successful)
        comparison_rows = self.paired_comparisons(successful)

        aggregate_fields = [
            "profile",
            "method",
            "metric",
            "n_runs",
            "mean",
            "sample_std",
            "ci_low",
            "ci_high",
        ]
        comparison_fields = [
            "profile",
            "comparison",
            "baseline",
            "metric",
            "direction",
            "n_pairs",
            "rl_mean",
            "baseline_mean",
            "mean_difference_rl_minus_baseline",
            "difference_ci_low",
            "difference_ci_high",
            "relative_improvement_pct",
            "paired_cohens_dz",
            "win_rate",
            "p_value",
            "p_value_holm",
        ]
        write_csv(
            self.output_dir / "aggregate_results.csv",
            aggregate_rows,
            aggregate_fields,
        )
        write_csv(
            self.output_dir / "paired_comparisons.csv",
            comparison_rows,
            comparison_fields,
        )

        summary_text = self.render_summary(
            aggregate_rows,
            comparison_rows,
            results,
        )
        summary_path = self.output_dir / "summary.txt"
        summary_path.write_text(summary_text + "\n")
        normalize_artifact_permissions(summary_path, file_mode=0o664)
        print("\n" + summary_text, flush=True)

    def run(self) -> int:
        self.preflight()
        self.build_schedule()
        self.write_manifest()

        expected_seconds = len(self.schedule) * (
            self.args.steps
            + self.args.warmup_seconds
            + TrafficManager.MEASUREMENT_SETTLE_SECONDS
            + self.args.cooldown_seconds
            + 4.0
        )
        print(
            f"Benchmark directory: {self.output_dir.resolve()}\n"
            f"Planned runs: {len(self.schedule)} "
            f"({len(self.args.profiles)} profiles × "
            f"{self.args.repetitions} repetitions × 3 methods)\n"
            f"Approximate minimum runtime: {expected_seconds / 3600:.2f} hours",
            flush=True,
        )

        results = []
        exit_code = 0
        try:
            for item in self.schedule:
                if self.interrupted:
                    exit_code = 130
                    break
                result = self.run_one(item)
                results.append(result)
                if self.interrupted or result.get("status") == "interrupted":
                    exit_code = 130
                    break
                if result.get("status") != "success" and not self.args.continue_on_error:
                    print(
                        "Aborting after failed/invalid run; artifacts retained. "
                        "Use --continue-on-error to finish remaining trials.",
                        flush=True,
                    )
                    exit_code = 1
                    break
        finally:
            self._clean_traffic()
            self.analyze_and_write(results)
            manifest_path = self.output_dir / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest["status"] = (
                    "interrupted"
                    if exit_code == 130
                    else ("failed" if exit_code else "completed")
                )
                manifest["finished_at_utc"] = utc_now()
                manifest["runs_recorded"] = len(results)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True)
                )
                normalize_artifact_permissions(manifest_path, file_mode=0o664)
        return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-oriented RL/ECMP/OSPF routing benchmark"
    )
    parser.add_argument(
        "--config",
        default="config/topologies/fat_tree_k4.yaml",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(TrafficManager.TRAFFIC_PROFILES),
        help="Comma-separated traffic profiles",
    )
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--schedule-seed", type=int, default=20260620)
    parser.add_argument("--analysis-seed", type=int, default=20260621)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.80)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--run-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-high-2", action="store_true")
    parser.add_argument("--save-dir", default="training_files")
    parser.add_argument("--weights-tag", default="best")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--telemetry-backend",
        choices=["cache", "influx", "cache-fallback-influx"],
        default="cache",
        help="Telemetry source passed to RL, ECMP, and OSPF runners",
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
    parser.add_argument(
        "--production-influx-write",
        choices=["on", "off"],
        default="off",
        help="Write RL production metrics to InfluxDB during benchmark",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    args.profiles = [
        item.strip() for item in args.profiles.split(",") if item.strip()
    ]
    return args


def main() -> int:
    args = parse_args()
    try:
        return BenchmarkOrchestrator(args).run()
    except Exception as exc:
        print(f"BENCHMARK ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
