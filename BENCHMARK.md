# Routing Benchmark Protocol

This benchmark compares:

- **RL:** the trained greedy DQN policy with queue-specific rerouting.
- **ECMP:** shortest-path equal-cost multipath with a stable CRC16 flow hash.
  DSCP and transport ports are excluded so all QoS queues of the same demand
  receive the same routing decision.
- **OSPF/SPF:** one deterministic minimum-cost path per destination, identical
  for all DSCP values, with no ECMP or QoS-aware route selection.

All methods use the same dimensionless inverse-bandwidth link cost normalized
so the fastest configured link has cost 100. This is a configured OSPF-style
cost model; it is not protocol overhead or measured dynamic link delay.

The OSPF condition models steady-state single-path SPF forwarding. It does not
run distributed OSPF daemons and does not measure protocol convergence or LSA
overhead. This distinction must be stated in publications.

## Default experiment

Start the network and collector before running the benchmark:

```bash
make run topo=fat_tree_k4
make collect
make benchmark
```

The default experiment uses:

- Profiles: `light_2`, `medium_2`, `high_1`
- Six independent repetitions per profile
- 300 measured one-second control intervals per run
- Five-second warm-up excluded from measurement
- Thirty-second inter-run cooldown
- One retry for infrastructure or telemetry-quality failure
- Minimum valid telemetry fraction of 0.80

This produces 54 runs: 3 profiles × 6 repetitions × 3 methods. For final
paper results, 12 or more repetitions are recommended:

```bash
make benchmark BENCH_REPETITIONS=12 STEPS=300
```

To include transient QoS stress, add the short burst profiles:

```bash
make benchmark \
  BENCH_PROFILES=light_2,medium_2,high_1,bursty_vo_1,bursty_vi_1,bursty_be_1 \
  BENCH_REPETITIONS=12 STEPS=300
```

`high_2` is rejected by default because project testing found that it can
overload the experiment host and contaminate measurements.

## Experimental design

The independent experimental unit is one complete run, not one telemetry
sample. Per-step samples within a run are temporally correlated and are not
treated as independent observations.

Each `(traffic profile, repetition)` forms a block:

1. One traffic seed is assigned to the block.
2. RL, ECMP, and OSPF use that same seed and profile.
3. Method order is selected from balanced permutations.
4. Traffic is stopped and the system is cooled down between runs.
5. Each routing method starts from a clean routing state.
6. P4 tables are read back before measurement; attempted writes alone are not
   accepted as evidence that the requested routing method is active.
7. The exact iperf3 process population is verified at the beginning and end of
   every run.
8. Every configured flow ID must produce INT latency records in Q0, Q1, and Q7
   over five-second validation windows before and after measurement. This
   prevents blackholed flows from disappearing from the SLA denominator.
9. A traffic health-monitor restart during measurement invalidates the run.

Traffic startup itself is fail-closed: every `start_traffic()` request verifies
one iperf3 server and one client for each configured `(flow_id, queue)` port.
It retries from a clean process state, restarts TaskServers between attempts,
and raises an error after three unsuccessful attempts. Aggregate process counts
alone are not accepted because duplicates could otherwise hide missing flows.

Shaped profiles apply their first scheduled offered-load stage before iperf
clients are launched. Exact endpoint and all-flow telemetry coverage are
therefore verified at the intended workload, not at a generic low-rate startup
load. This transition never inspects SLA results and never restarts iperf.

Using common random numbers makes RL-vs-baseline comparisons paired, reducing
variance caused by traffic randomization. Balanced order limits systematic
carry-over and thermal/resource effects.

### Deterministic shaped profiles

Light profiles use fixed offered-load tiers throughout the measured interval.
On `fat_tree_k4`, the measured ECMP congestion knee is sharp, so medium and
high profiles use deterministic recovery-low plus overload-pulse cycles. This
keeps short runs near their intended SLA band instead of starting every run
from an unrealistically empty queue and then slowly decaying into congestion.

Per-demand totals are:

| Profile | Low Mbps | High Mbps | High steps / 10 |
|---------|----------|-----------|-----------------|
| `light_1` | 1.50 | 1.50 | 0 |
| `light_2` | 1.62 | 1.62 | 0 |
| `medium_1` | 1.50 | 1.80 | 3 |
| `medium_2` | 1.50 | 1.92 | 4 |
| `high_1` | 1.50 | 1.85 | 6 |
| `high_2` | 1.85 | 1.85 | 0 |

The common queue split is Q0=22.327%, Q1=34.591%, and Q7=43.082% of the
active stage total. With three demands per sender, these correspond to source
HTB caps from 45.0% of a 10 Mbps sender link at a 1.50 Mbps stage to 57.6% at
a 1.92 Mbps pulse stage.

The former 70%, 50%, and 20% SLA figures motivated the selected stress points
but are calibration references only. They are not runtime acceptance criteria,
and no run is rejected because a particular congestion or SLA state was not
reached. This avoids conditioning results on the routing method's observed
outcome.

All steady and bursty names live in the single
`TrafficManager.TRAFFIC_PROFILES` registry. Bursty profiles use queue-biased
load vectors and deterministic bursts in every ten-step block: three high
steps for `_1` and six for `_2`. A short benchmark therefore always exercises
the named burst, and changing stages does not restart iperf.

A 50-step ECMP sanity sweep on `fat_tree_k4` with traffic seed 42 produced the
expected load staircase with every run verified: `light_1` 100.00% SLA,
`light_2` 100.00%, `medium_1` 53.33%, `medium_2` 18.67%, `high_1` 16.00%,
and `high_2` 10.67%. Treat these as calibration evidence, not acceptance
thresholds for paper experiments.

Iperf offers slightly above each profile's maximum target while the existing
Mininet root HTB class on every sender is changed and read back to enforce the
exact aggregate offered load. The original HTB settings are restored when
traffic stops. No traffic process is restarted during a burst transition.

This short calibration is a workload sanity check, not inferential evidence.
Paper comparisons must still use repeated complete runs and report the actual
SLA values with confidence intervals. The benchmark must use one common step
horizon for RL, ECMP, and OSPF.

## Metrics

Primary pre-specified metrics are:

- Mean reward on valid telemetry intervals — higher is better.
- SLA compliance on valid intervals — higher is better.
- Macro-average latency across queues — lower is better.
- Macro-average drops per 100 ms across queues — lower is better.
- Valid telemetry fraction — higher is better.

The artifacts also retain:

- Per-queue mean and p95 of the step-level p95 latency.
- Per-queue mean drop and utilization.
- Worst-queue p95 latency.
- Offered traffic load.
- RL action application rate.
- All invalid rows, failed attempts, logs, and return codes.

Utilization is descriptive; lower utilization is not automatically better.
The current system does not directly export receiver goodput, so utilization
must not be described as throughput in the paper.

## Statistical analysis

Statistics are computed across independent runs:

- Mean and sample standard deviation.
- 95% percentile-bootstrap confidence interval.
- Paired RL-minus-baseline mean difference.
- Relative improvement with direction normalized so positive favors RL.
- Paired Cohen's `dz` effect size.
- Two-sided paired sign-flip randomization test.
- Holm correction across the reported paired hypotheses.

Confidence intervals are not computed by treating individual one-second
samples as independent replications.

## Artifacts

Each invocation creates `benchmark_results/benchmark_<timestamp>/` containing:

- `manifest.json`: full schedule, parameters, code/config/model hashes,
  platform details, method definitions, and analysis policy.
- `environment.txt`: compiler, BMv2, Python, NumPy, and package versions.
- `run_results.csv`: one row per independent run.
- `aggregate_results.csv`: means, standard deviations, and confidence intervals.
- `paired_comparisons.csv`: RL-vs-OSPF and RL-vs-ECMP paired analyses.
- `summary.txt`: the terminal comparison table.
- `runs/`: raw per-step CSV files, complete logs, runner summaries, retries,
  and failure metadata.

Every successful run must include `routing_state_verified=true`,
`traffic_state_verified=true`, and `telemetry_state_verified=true` in its
runner summary. ECMP summaries also include a SHA-256 digest of the
deterministic route plan and per-switch expected/observed ECMP table counts.
Missing verification marks the run invalid even when the available telemetry
values themselves appear complete.

Use a fixed output directory to resume an interrupted experiment:

```bash
make benchmark BENCH_OUTPUT=benchmark_results/paper_fat_tree_k4
```

The raw artifacts should be archived with the paper. Do not publish only the
aggregate tables.

Protocol and reproducibility references:

- [RFC 2328: OSPF Version 2](https://www.rfc-editor.org/rfc/rfc2328.html)
- [RFC 2992: Analysis of an Equal-Cost Multi-Path Algorithm](https://www.rfc-editor.org/rfc/rfc2992.html)
- [ACM Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current)

## Interpretation boundaries

- Results apply to the configured topology, link rates, traffic matrix,
  queue scheduler, model checkpoint, and software/hardware environment.
- ECMP is flow-stable and queue-independent by design.
- The OSPF baseline is single-path SPF, despite OSPF itself permitting ECMP.
- Benchmarking a topology unseen during training tests generalization and
  should be reported separately from in-topology evaluation.
- Failed or low-validity runs must be reported rather than silently removed.
- The headline method table is an aggregate across independent repetitions.
  The terminal summary also prints every ECMP repetition separately so a
  100%/0% split cannot be mistaken for one run.
