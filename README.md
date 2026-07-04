# P4-INT
Implementation In­Band Network Telemetry in P4

<!-- GETTING STARTED -->
# Getting Started
This is an example of how you may give instructions on setting up your project locally. To get a local copy up and running follow these simple example steps.

## Prerequisites
### environment
* win10
* vm-ubuntu20.04

## Installation
1. install Mininet
    ```
      git clone https://github.com/mininet/mininet
      cd mininet

      sudo PYTHON=python3 mininet/util/install.sh -n
    ```
2. install P4
  
    For Ubuntu 20.04 and Ubuntu 21.04 it can be installed as follows:
    ```
      . /etc/os-release
      echo "deb http://download.opensuse.org/repositories/home:/p4lang/xUbuntu_${VERSION_ID}/ /" | sudo tee /etc/apt/sources.list.d/home:p4lang.list
      curl -L "http://download.opensuse.org/repositories/home:/p4lang/xUbuntu_${VERSION_ID}/Release.key" | sudo apt-key add -
      sudo apt-get update
      sudo apt install p4lang-p4c
    ```
3. python3 Dependency package
    ```
   sudo pip3 install psutil networkx
    ```
4. Influxdb
    ```sh
    sudo apt-get install influxdb
    sudo service influxdb start
    sudo pip3 install influxdb
    ```


## Usage

1. Clone the repo
   ```sh
   git clone https://github.com/adaraiseh/p4_qos_int.git
   ```
2. run the network
   ```
   make
   ```
3. start the influxdb collector:
   ```
   sudo python3 report_collector/influxdb_export.py
   ```
4. start RL_Agent:
   ```
   python3 rl_agent_4.py
   ```

## Current RL Behavior

The active training and production agent is `rl_agent_4.py`.

Key behavior:

- State dimension: `1200` stacked features.
  - `61` raw observation features per frame.
  - `16` observation frames.
  - `14` one-hot action history entries per frame.
- Action dimension: `14`.
  - `0`: no-op.
  - `1-12`: queue-specific reroute actions for Q0, Q1, and Q7, with two
    alternate paths and `K={1,2}` demand units.
  - `13`: `multi-k1`, which reroutes at most one eligible demand unit per
    violating queue.
- `K=1` reroutes the worst eligible demand unit for the selected queue and
  alternate path.
- `K=2` reroutes the worst two eligible demand units for the selected queue
  and alternate path when at least two unlocked units are available.
- A successfully rerouted demand unit is locked for `5` control steps using
  the dataplane-compatible key `(qid, dst_ip, bottleneck_sid)`. Other demand
  units on the same queue may still be rerouted during that lock window.
- RL baseline resets start from a selectable base routing method:
  `--rl-base-routing ospf` (default) or `--rl-base-routing ecmp`. The learned
  `/32 + DSCP` reroutes are installed in `l3_forward.rl_overlay`, which is
  applied before either base method, so the agent can take over from the
  selected base without changing the action logic.
- When a traffic profile is active, RL publishes `/tmp/p4_paths.json` with the
  same configured-flow schema as standalone OSPF/ECMP (`flow_id`,
  `routing_mode`, and per-queue path). This makes the RL-base starting diagram
  directly comparable with the matching standalone baseline before overlays are
  applied.
- RL production, standalone ECMP, and standalone OSPF use the same traffic
  validation contract: exact startup endpoints, verified sender HTB shaping at
  startup, verified sender HTB shaping at measurement start, and exact
  post-measurement endpoints. Benchmark runs reject summaries that lack this
  evidence.
- Production and standalone benchmark runners also read back sender HTB caps on
  every warmup and measured profile step. RL training keeps its existing
  periodic step verification policy.
- `traffic_seed` controls seeded high/low positions for shaped traffic
  profiles while preserving each profile's high-step count and average load.
- Per-flow demand loads use a deterministic wide distribution from very low to
  very high senders while preserving each profile's aggregate target. Each
  individual demand is capped at 75% of the weakest topology link, which is
  7.5 Mbps on the 10 Mbps `fat_tree_k4` links.
- The observation includes batch-awareness per queue:
  `eligible_count_norm`, `top1_pressure_norm`, and `top2_pressure_norm`.
  These expose whether K2 is meaningful without exposing full demand IDs.
- The local telemetry cache can return `top_demands` with `top_n`; the agent
  requests the top `6` hot demands per queue so it can choose eligible K1/K2
  units and skip locked units.

Production and benchmark CSV logs include batch and lock diagnostics:
`requested_batch_size`, `batch_reroute_count`, `locked_units_count`, and
`rerouted_units`. Production, ECMP, and OSPF rows also include
`invalid_reason`; rows with `data_valid=0` are retained for audit but excluded
from benchmark aggregates and runner-side reward/SLA summaries.

# Hosts terminal tests:
1. in mininet terminal
   ```sh
   xterm h1 h2
   ```
2. in xterm h2
    ```sh
    python3 ./receive.py
    ```
3. in xterm h1 
    ```sh
   python3 ./send.py --ip 10.0.1.1 --l4 udp --port 8080 --m "hello world !" --c 1    
   ```

# influxdb operation
```sh
INSERT flow_latency,src_ip="10.0.1.1",dst_ip="10.0.3.2",src_port=1234,dst_port=1234,protocol=17 value=0.64
INSERT switch_latency,switch_id=1 value=0.64
INSERT queue_occupancy,switch_id=1,queue_id=1 value=0.1
INSERT link_latency,ingress_switch_id=2,ingress_port_id=1,egress_switch_id=1,egress_port_id=2 value=

SELECT * FROM flow_latency
drop measurement flow_latency
```
