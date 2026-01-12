# P4 QoS INT - Project Documentation

## Project Overview

This project implements a **DQN-based Reinforcement Learning agent** for dynamic QoS-aware routing optimization in P4-programmable networks. It uses **In-band Network Telemetry (INT)** to collect real-time network metrics and automatically reroute traffic to avoid congestion.

### Key Features
- P4 switch programming with INT support
- Real-time telemetry collection via INT reports
- DQN agent with 960-dimensional state space (stacked observations + actions)
- 8-action space for queue-specific and multi-queue rerouting
- InfluxDB for metrics storage and querying
- Support for multiple topology types (Fat-Tree, Leaf-Spine, Three-Tier)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NETWORK LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  network.py          │ Mininet setup with P4 switches (BMv2)        │
│  controller.py       │ P4 switch control, routing, path computation │
│  p4src/int_md.p4     │ P4 program with INT metadata insertion       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TELEMETRY LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  report_collector/   │ INT report parsing and InfluxDB export       │
│    collector.py      │ - Per-interface sniffers (Scapy AsyncSniffer)│
│    influxdb_export.py│ - Async batch writes to InfluxDB             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          RL LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│  rl_agent_4.py       │ DQN training with Dueling architecture       │
│  rl_production.py    │ Production inference (greedy policy)         │
│  traffic_generator.py│ Traffic generation using iperf3              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### Network Layer

#### `network.py`
- Creates Mininet topology with P4 switches
- Configures hosts, links, and QoS queues
- Starts INT collectors on designated interfaces
- Spawns iperf3 servers on hosts

#### `controller.py`
- Installs P4 table rules for routing
- Manages path computation (ECMP, per-queue paths)
- Handles rerouting requests from RL agent
- Thrift client connections to BMv2 switches
- Identify worst flow and bottleneck nodes

### Telemetry Layer

#### `report_collector/collector.py`
- Parses INT report packets (UDP port 1234)
- Extracts per-hop metrics: switch latency, TX utilization, queue drops
- Computes flow latency (end-to-end)
- Exports metrics to InfluxDB with batched async writes (50ms flush interval)
- **Note:** Aggregation is disabled in production (`aggregate_enabled=False`)

**Metrics Collected:**
| Measurement | Description | Tags |
|-------------|-------------|------|
| `switch_latency` | Per-hop latency (ms) | switch_id, queue_id, flow_id |
| `tx_utilization` | Egress port utilization (%) | switch_id, queue_id, egress_port |
| `q_drop_rate_100ms` | Instantaneous drop rate | switch_id, queue_id, flow_id |
| `flow_latency` | End-to-end latency (ms) | src_ip, dst_ip, queue_id |

### RL Layer

#### `rl_agent_4.py`
- **State**: 960 dimensions (52 metrics × 16 frame stack + 8 actions × 16 history)
- **Actions**: 8 total (0=no-op, 1-6=single queue reroute, 7=multi-queue)
- **Reward**: SLA-based with drop penalty and action cost
- **Architecture**: Dueling DQN with prioritized experience replay

#### `rl_production.py`
- Loads trained model for production inference
- Greedy action selection (no exploration)
- Logs Q-values and actions to InfluxDB
- Hot-reload support for model updates

---

## Configuration

### Topology Files
Located in `config/topologies/`:
- `fat_tree_k2.yaml`, `fat_tree_k4.yaml`
- `leaf_spine_*.yaml`
- `three_tier_*.yaml`

### Key Configuration Parameters

```yaml
qos_queues:
  - id: 0, name: "voice", sla_ms: 100
  - id: 1, name: "video", sla_ms: 150
  - id: 7, name: "best_effort", sla_ms: 200

link_bandwidths:
  host_leaf: 10    # Mbps
  leaf_spine: 10
  spine_core: 10
```

### Timing Constants (rl_agent_4.py:146-153)

```python
WINDOW_SECONDS = 1.0        # InfluxDB query window
SAFETY_LAG_MS = 0           # No safety lag (aggressive)
DELAY_AFTER_ACTION = 1.0    # Sleep after action
DELAY_NO_ACTION = 1.0       # Sleep when no action taken
MIN_POINTS_PER_METRIC = 1   # Minimum data points for freshness
```

---

## Operational Procedures

### Testing Guidelines

**IMPORTANT:** Before starting any new test:
1. **Kill all previous traffic processes** to ensure clean state:
   ```bash
   sudo pkill -9 iperf3
   sudo pkill -9 -f "__RL_TRAFFIC__"
   sudo pkill -9 -f "rl_production"
   ```
2. **Wait 30-60 seconds** for InfluxDB data to stabilize before querying
3. **Do NOT use high_2 profile** for testing - it causes CPU overload and unreliable data
4. Use `medium_1` or `medium_2` profiles for stable testing

**CRITICAL: After completing any test:**
Always stop traffic generation after completing tests to ensure clean state for the next test:
```bash
# Stop all traffic processes
sudo pkill -9 iperf3
sudo pkill -9 -f "__RL_TRAFFIC__"

# Or if using make:
make stop_traffic  # If available
```

Leaving traffic running after tests can cause:
- InfluxDB data pollution (old traffic metrics mixed with new test data)
- CPU/memory resource exhaustion
- Incorrect baseline measurements in subsequent tests
- Stale flow data affecting RL agent decisions

### Starting the System

```bash
# Terminal 1: Start network
make run topo=fat_tree_k4

# Terminal 2: Start collector
make collect

# Terminal 3: Start training
make train

# OR for production:
make production profile=medium_2
```

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make run topo=<name>` | Start network with topology |
| `make stop` | Stop mininet |
| `make collect` | Start INT collector |
| `make train` | Full training (50K steps) |
| `make production profile=<name>` | Run with best weights |
| `make test_traffic profile=<name>` | Test traffic profile |
| `make validate` | Validate topology config |
| `make help` | Show all available commands |

### Debug Logging

```bash
# Enable debug logging for any command
LOG_LEVEL=debug make train

# Log files location
ls log/*.log
```

---

## Troubleshooting

### Missing Metrics Issue

**Symptoms:**
```
[WARNING] [Telemetry] Invalid data for queues [0, 1, 7] (q0:stale_data, q1:missing_metrics, q7:missing_metrics) - only 0/3 valid
[INFO] [Snapshot] Queue 0: No hot demand found, skipping bottleneck detection
```

**Diagnostic Steps:**

1. **Check if collector is writing:**
   ```bash
   tail -f log/collector_*.log | grep "Exported"
   ```

2. **Check if traffic is running:**
   ```bash
   ps aux | grep iperf | grep -v grep | wc -l
   ```

3. **Check InfluxDB data availability:**
   ```bash
   python3 diagnose_queries.py --loops 5
   ```

4. **Check INT reports arriving:**
   ```bash
   sudo tcpdump -i t1-eth10 -c 20 'udp and dst port 1234' -q
   ```

**Common Causes:**
1. **Traffic stopped** - iperf3 processes crashed or were killed (now auto-recovered by health monitor)
2. **Collector crashed** - Check collector log for errors
3. **InfluxDB connection issues** - External DB may be unreachable
4. **Network degraded** - P4 switches may have crashed

**Automatic Recovery:**
The TrafficManager now includes a health monitoring thread that:
- Checks iperf process count every 10 seconds
- Auto-restarts traffic if <70% of expected processes are running
- Logs: `[Health Monitor] Only X/Y iperf processes running... Restarting traffic`

**Resolution (if auto-recovery doesn't work):**
```bash
# Restart everything
make stop
make run topo=fat_tree_k4
make collect  # In another terminal
make train    # In another terminal
```

### Query Performance Issues

If queries are slow (>500ms), you'll see:
```
[WARNING] [Query Slow] 750ms total (1 attempts)
```

Check InfluxDB connectivity:
```bash
ping 192.168.56.1
```

### Thread Pool Issues

If work queue is backing up:
```
[WARNING] [ThreadPool] Work queue backlog: 15 (expected ~0)
```

This indicates queries are taking too long. Check InfluxDB performance.

---

## Diagnostic Tools

### `diagnose_queries.py`
Real-time InfluxDB query diagnostics:
```bash
python3 diagnose_queries.py --loops 10 --interval 1.0
```

### `int_metrics_tester.py`
Verify INT metrics collection:
```bash
sudo python3 int_metrics_tester.py --config config/topologies/fat_tree_k4.yaml --verbose
```

---

## RL Agent Metrics Collection Logic

### Overview

The RL agent collects metrics from InfluxDB at each step to build a state snapshot. The collection follows a **retry-based recovery** strategy with **no stale cache fallback** - data must come from fresh InfluxDB queries or be declared missing.

### Metrics Collected Per Step

For each of the 3 QoS queues (Q0=voice, Q1=video, Q7=best_effort), the agent collects:

| Metric | Source Measurement | Description | Unit |
|--------|-------------------|-------------|------|
| `lat_p95` | `flow_latency` | P95 end-to-end latency | ms |
| `drop_p95` | `q_drop_rate_100ms` | P95 drop rate | ratio (0-1) |
| `util_p95` | `tx_utilization` | P95 egress utilization | % |
| `hot_demand` | `flow_latency` | Highest-latency (src_ip, dst_ip) pair | IP tuple |

### Query Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    _collect_snapshot()                              │
├─────────────────────────────────────────────────────────────────────┤
│  1. Parallel queries via ThreadPoolExecutor:                        │
│     ├─ _query_aggregated_metrics() → lat_p95, drop_p95, util_p95   │
│     └─ _get_all_hottest_demands() → hot_demand per queue           │
│                                                                     │
│  2. For each metric type missing from initial query:                │
│     └─ _retry_metric_query() with 5 retries, 1s delay, window exp  │
│                                                                     │
│  3. Validate data:                                                  │
│     ├─ metrics_present: all 3 metrics received?                    │
│     └─ values_sane: within reasonable bounds?                      │
│                                                                     │
│  4. Mark data_valid = metrics_present AND values_sane              │
└─────────────────────────────────────────────────────────────────────┘
```

### Retry Logic (All Metrics)

All metrics use the same retry strategy - **no stale cache fallback**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max retries | 5 | Maximum retry attempts |
| Retry delay | 1 second | Sleep between retries |
| Window expansion | +1s per retry | Query window extends on each retry |
| Max wait time | 5 seconds | Total max time before declaring missing |

**Retry behavior:**
1. Initial query fails to return data for some queues
2. Log warning: `[Query S<step>] <metric> missing for Q[...], triggering retry`
3. Sleep 1 second, expand query window by cumulative wait time
4. Query again, check if target queues now have data
5. Repeat until all queues recovered OR 5 retries exhausted
6. If still missing after 5 retries, declare metric missing and continue

**Note:** All retry logs include step number (`S<step>`) to identify which step triggered the retry, making it easier to debug when retries from previous steps overlap with new steps.

**Example log output:**
```
[Query S19] flow_latency missing for Q[0, 1, 7], triggering retry
[Query S19] flow_latency retry 1/5: no data (35ms, )
[Query S19] flow_latency retry 2/5 PARTIAL: got Q[1], still missing Q[0, 7] (91ms +1s window)
[Query S19] flow_latency retry 3/5 SUCCESS: got ALL Q[1, 0, 7] in 46ms +2s window
[Query S19] Recovered lat_p95=2.19ms for Q1
[Query S19] Recovered lat_p95=2.22ms for Q0
[Query] Recovered lat_p95=4.99ms for Q7
```

### Metrics-Specific Details

#### P95 Aggregated Metrics (`_query_aggregated_metrics`)

Single Flux query retrieves all 3 metrics for all 3 queues:

```flux
base = from(bucket:"int_telemetry")
    |> range(start:<start>, stop:<stop>)
    |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
    |> toFloat()

lat_p95 = base |> filter(fn: (r) => r._measurement == "flow_latency")
    |> group(columns:["queue_id"]) |> quantile(q:0.95)

drop_p95 = base |> filter(fn: (r) => r._measurement == "q_drop_rate_100ms")
    |> group(columns:["queue_id"]) |> quantile(q:0.95)

util_p95 = base |> filter(fn: (r) => r._measurement == "tx_utilization")
    |> group(columns:["queue_id"]) |> quantile(q:0.95)

union(tables:[lat_p95, drop_p95, util_p95])
```

After initial query, each missing metric type triggers its own retry loop:
- Missing `lat` → retry with `measurement="flow_latency"`
- Missing `drop` → retry with `measurement="q_drop_rate_100ms"`
- Missing `util` → retry with `measurement="tx_utilization"`

#### Hot Demands (`_get_all_hottest_demands`)

Finds the (src_ip, dst_ip) pair with highest mean latency per queue:

```flux
from(bucket:"int_telemetry")
    |> range(start:<start>, stop:<stop>)
    |> filter(fn: (r) => r._measurement == "flow_latency")
    |> filter(fn: (r) => r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7")
    |> group(columns:["queue_id", "src_ip", "dst_ip"])
    |> mean(column:"_value")
    |> group(columns:["queue_id"])
    |> sort(columns:["_value"], desc:true)
    |> limit(n:1)
```

Has its own built-in retry loop (same 5 retries, 1s delay, window expansion).

#### Freshness Check (`_check_all_queues_freshness`)

Counts data points per queue to ensure sufficient data exists:

```flux
from(bucket:"int_telemetry")
    |> range(start:<start>, stop:<stop>)
    |> filter(fn: (r) => r._measurement == "flow_latency" or
                         r._measurement == "q_drop_rate_100ms" or
                         r._measurement == "tx_utilization")
    |> group(columns:["queue_id"])
    |> count()
```

A queue is "fresh" if it has >= `MIN_POINTS_PER_METRIC` (default: 1) data points.

### Data Validation

After collection, each queue's data is validated:

1. **metrics_present**: All 3 metrics (lat, drop, util) must be received
2. **values_sane** (`_metric_sane()`): Values within reasonable bounds:
   - `lat_p95`: 0 < lat < 10000 ms
   - `drop_p95`: 0 <= drop <= 1.0
   - `util_p95`: 0 <= util <= 200%

Final: `data_valid = metrics_present AND values_sane`

**Note:** Freshness check was removed - if metrics are successfully retrieved from InfluxDB and pass sanity checks, they are considered valid. This simplifies validation and prevents false negatives when data exists but point counts are low.

### When Data is Missing

If data is invalid after all retries:
- Log warning with diagnostics (time window, step, metrics received, hot demands)
- Queue is marked `data_valid=False`
- Agent uses default values (2x SLA for latency, caps for drop/util)
- Bottleneck detection skipped for that queue

**Example missing data log:**
```
[Telemetry] Invalid data for queues [0] (q0:missing_metrics) - only 2/3 valid
[Telemetry Debug] Time window: 2026-01-12T16:01:14Z to 2026-01-12T16:01:15Z
[Telemetry Debug] Global step: 59
[Telemetry Debug] Metrics received: {0: {'lat': True, 'drop': False, 'util': True}, ...}
[Telemetry Debug] Hot demands: {1: ('10.14.3.2', '10.18.12.2'), ...}
```

---

## Key Code Locations

### Telemetry Collection
- `rl_agent_4.py:1099-1193` - `_query_aggregated_metrics()`: P95 metrics query with retry
- `rl_agent_4.py:1195-1285` - `_retry_metric_query()`: Generalized retry logic for any metric
- `rl_agent_4.py:1287-1500` - `_collect_snapshot()`: Main telemetry function
- `rl_agent_4.py:1567-1655` - `_get_all_hottest_demands()`: Hot flow detection with retry

### Data Validation
- `rl_agent_4.py:1360-1390` - Data validity checks with diagnostic logging
- `rl_agent_4.py:1657-1700` - `_metric_sane()`: Sanity checks for metric values

### Query Execution
- `rl_agent_4.py:1065-1097` - `_influx_query_with_retry()`: Base retry logic with timing

### Collector
- `collector.py:260-278` - `log_export_rate()`: Export statistics
- `collector.py:279-329` - `record_drop_rate_instant()`: Drop rate calculation

---

## Traffic Profiles

| Profile | Voice (Q0) | Video (Q1) | BE (Q7) | Category |
|---------|------------|------------|---------|----------|
| light_1 | 0.05-0.10 | 0.05-0.15 | 0.10-0.20 | Light |
| light_2 | 0.08-0.12 | 0.10-0.18 | 0.15-0.25 | Light |
| medium_1 | 0.12-0.20 | 0.15-0.25 | 0.25-0.40 | Medium |
| medium_2 | 0.15-0.25 | 0.20-0.30 | 0.30-0.50 | Medium |
| high_1 | 0.25-0.35 | 0.30-0.45 | 0.50-0.80 | High |
| high_2 | 0.30-0.45 | 0.35-0.55 | 0.70-1.00 | High |

Bursty profiles add periodic traffic spikes on top of medium_1 baseline.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFLUX_TOKEN` | InfluxDB authentication token | (required) |
| `LOG_LEVEL` | Logging level (debug/info) | info |
---

## INT Sampling and Collector Behavior

### INT Time-Based Sampling
The P4 program uses time-based sampling to limit INT report rate:
- **Sampling interval**: 100ms per (flow_id, queue_id) combination
- Each unique flow+queue can only generate 1 INT report every 100ms
- This prevents overwhelming the collector while still providing representative data

### Collector Export Statistics
The collector logs export statistics every ~30 seconds:
```
[Collector] Exported 8800 records (Q0:2900 Q1:3100 Q7:2800) | caches: drop=265 agg=0 | lat:500 skip:0
```
- `records`: Total per-hop metrics (switch_latency, tx_util, drop_rate)
- `Q0/Q1/Q7`: Per-queue breakdown
- `lat`: Number of flow_latency records written (one per INT packet)
- `skip`: Number of flow_latency records skipped (should be 0)
- `drop`: Size of drop rate cache
- `agg`: Size of aggregation cache (0 = aggregation disabled)

### Aggregation
Collector aggregation is **disabled** by default (`aggregate_enabled=False`). This provides:
- Lower latency: data written immediately
- Higher data resolution: no averaging
- Better for RL training: more data points per window

---

## Flow Latency Calculation

### How Flow Latency is Calculated

**Formula:** `flow_latency = (last_hop_egress_time - first_hop_ingress_time) / 1,000,000.0` (result in milliseconds)

**INT Metadata Structure:**
INT metadata is **prepended** by each switch, meaning:
- Index 0 = **last hop** (most recent metadata, added by the final switch)
- Index n-1 = **first hop** (oldest metadata, added by the ingress switch)

**Calculation in Code** ([collector.py:615-617](report_collector/collector.py#L615-L617)):
```python
flow_latency = (
    flow_info.egress_tstamps[0] - flow_info.ingress_tstamps[-1]
) / 1_000_000.0
```

Where:
- `egress_tstamps[0]` = Egress timestamp from **last hop** (when packet left the final switch)
- `ingress_tstamps[-1]` = Ingress timestamp from **first hop** (when packet entered the network)
- Division by 1,000,000 converts nanoseconds to milliseconds

**Note:** Flow latency is independent of `safe_hops` - it only requires at least 1 egress and 1 ingress timestamp.

### Timestamp Consistency with Per-Hop Metrics

**All metrics use the same `report_time`** for InfluxDB timestamp consistency:

1. **report_time selection** ([collector.py:465-476](report_collector/collector.py#L465-L476)):
   - `use_device_time=False` (default): Uses `time.time_ns()` (system time when packet was processed) - **REQUIRED** because P4 device timestamps are NOT Unix epoch timestamps
   - `use_device_time=True`: Uses device timestamp from INT metadata - **DO NOT USE** as P4 timestamps are switch uptime, not Unix epoch

2. **Per-hop metrics** (switch_latency, tx_utilization, queue_occupancy, drop_rate):
   - All use `report_time` as their InfluxDB timestamp
   - Each hop's metrics are written with the same timestamp

3. **flow_latency**:
   - Also uses `report_time` as its InfluxDB timestamp
   - The **value** is calculated from device timestamps, but the **record timestamp** in InfluxDB is `report_time`

### Queue ID Consistency

**All metrics use queue_id from INT metadata** for consistency:

- Per-hop metrics: Use `q_ids[i]` for each hop
- flow_latency: Uses `q_ids[-1]` (queue_id from the **first hop**)

This ensures that flow_latency records can be joined with per-hop metrics using the same queue_id tag.

### Sanity Checks

Flow latency values are validated before writing ([collector.py:622-626](report_collector/collector.py#L622-L626)):
- **Negative latency**: Rejected (indicates timestamp wraparound)
- **Latency > 10 seconds**: Rejected (indicates stale packet or clock sync issues)

---

## Traffic Logging

Traffic configurations are automatically logged to `log/traffic_log.csv` with the following events:

| Event | Description |
|-------|-------------|
| `start` | Traffic started with a new profile |
| `stop` | Traffic stopped |
| `restart` | Traffic restarted by health monitor |
| `burst_start` | Burst traffic started (for bursty profiles) |
| `burst_end` | Burst traffic ended, returning to baseline |

**CSV Columns:**
- `timestamp`: ISO format timestamp
- `event`: Event type
- `profile_name`: Traffic profile name (e.g., medium_2, bursty_be_1)
- `profile_category`: Category (light, medium, high, bursty)
- `load_q0`, `load_q1`, `load_q7`: Per-queue load in Mbps
- `is_bursty`: Whether this is a bursty profile episode
- `baseline_profile`: Baseline profile for bursty episodes
- `num_traffic_pairs`: Number of active traffic pairs
- `extra_info`: JSON with additional event-specific info

---

## Recent Changes

- **Removed freshness check**: Validation simplified to `data_valid = metrics_present AND values_sane`. If metrics are retrieved from InfluxDB, they're considered fresh.
- **Step numbers in retry logs**: All retry logs now include `[Query S<step>]` prefix to identify which step triggered the retry.
- **Removed stale cache fallback**: All metrics now use retry-only recovery (5 retries, 1s delay, expanding window). No cache fallback - data must come from fresh InfluxDB queries.
- **Unified retry logic**: All metrics (lat, drop, util, hot_demands) use the same `_retry_metric_query()` function with consistent behavior.
- **Diagnostic logging**: Added detailed telemetry debug output when metrics are missing
- **Query timing**: Added slow query warnings (>500ms threshold)
- **Collector health**: Added cache size and latency stats (lat/skip counters)
- **ThreadPool monitoring**: Added work queue backlog detection
- **Health monitoring**: TrafficManager auto-restarts crashed iperf processes
- **flow_latency fix**: Uses INT queue_id from metadata (consistent with per-hop metrics)
- **safe_hops fix**: Timestamps no longer required in safe_hops calculation

---

## Known Issues

1. **Intermittent missing flow_latency in short windows**: Due to INT time-based sampling (100ms), some 1-second query windows may have no flow_latency for a specific queue. The retry logic with expanding window handles this.

2. **Q7 lower data rate**: Best-effort queue typically has fewer flows than voice/video queues, which can cause intermittent "missing_metrics" for Q7.

3. **External InfluxDB latency**: High latency to external InfluxDB (192.168.56.1) can cause query timeouts.
