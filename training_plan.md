# DQN Training Plan for QoS Routing

This document provides training configurations for the RL-based QoS routing
agent (`rl_agent_4.py`). The current agent uses a 14-action K1/K2 reroute
space, batch-aware observations, and demand-unit lockout to balance fast
recovery from bad SLA states with lower churn during bursty `_3` profiles.

---

## Table of Contents
1. [Single Topology Training](#single-topology-training)
2. [Multi-Topology Training](#multi-topology-training)
3. [Hyperparameter Reference](#hyperparameter-reference)
4. [Evaluation Commands](#evaluation-commands)

---

## Single Topology Training

For training on a single topology (e.g., `fat_tree_k4`), use the default parameters without multi-topology features.

### Recommended Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Steps | 80,000 total | Three-stage curriculum sized for the new K1/K2 action space |
| Learning Rate | `1e-4` (default) | Optimal for stable training |
| Epsilon | Stage-local decay | Re-open exploration when resuming into burst and polish stages |
| Batch Size | 64 | Stable gradient estimates |
| Replay Buffer | 80,000 | Holds the full recommended run while keeping checkpoints manageable |
| Reset Probability | High early, moderate late | Trains fast recovery from OSPF baseline instead of only warm-state maintenance |

### Training Command

```bash
# Recommended fat-tree k=4 training curriculum
make train_paper
```

The Makefile default is:

| Stage | Steps | Resume epsilon | Epsilon decay | Reset probability | Traffic emphasis |
|-------|-------|----------------|---------------|-------------------|------------------|
| 1. Foundation | 35,000 | fresh `1.00` | 45,000 | `0.95 → 0.70` | high baseline recovery plus `_3` burst exposure |
| 2. Burst specialization | 25,000 | `0.45` | 25,000 | `0.85 → 0.60` | high_1 and bursty `_3`, especially `bursty_vi_3`/`bursty_be_3` |
| 3. Stability polish | 20,000 | `0.15` | 15,000 | `0.65 → 0.45` | balanced high/bursty with enough light/medium no-op examples |

The old three-stage run ending in
`/media/sf_amjad/p4_qos_int/training_runs/20260628-223323` filled a 50K replay
buffer while the agent reached about 99K valid learning steps. That meant later
burst/polish experience displaced much of the early OSPF-baseline recovery
distribution. With the new 1200-state/14-action design, 80K replay is the
default compromise: it retains the full recommended 80K curriculum, gives K2
and `multi-k1` enough samples, and avoids pushing every checkpoint into an
unnecessarily large 100K+ replay snapshot.

Keep the default as three stages rather than one mixed 80K run. A single stage
is simpler, but it cannot reopen exploration when the traffic emphasis changes.
The stage-local resume epsilons deliberately do that: Stage 2 explores the new
K2/lockout behavior under bursty `_3` pressure, and Stage 3 reintroduces
controlled exploration while adding more light/medium no-op examples. Because
the replay buffer now spans the full 80K curriculum, the stages no longer erase
the earlier recovery distribution.

### Expected Training Time
- ~80,000 steps × ~2s/step ≈ **44 hours**
- Checkpoints saved at 25%, 50%, 75%, best, and final

### Behavioral Targets

The current enhancement targets two behaviors that should both be checked
during training:

- **Fast recovery:** high-load profiles should move from OSPF baseline routing
  to a better distribution quickly. The action space supports this with K2
  single-queue actions and `multi-k1`.
- **Burst stability:** bursty `_3` profiles should not cause the agent to chase
  the same demand unit every burst step. Successful reroutes lock
  `(qid, dst_ip, bottleneck_sid)` for five control steps while leaving other
  demand units on the same queue eligible.

The observation space exposes only minimal batch awareness:
`eligible_count_norm`, `top1_pressure_norm`, and `top2_pressure_norm` per
queue. Demand IDs are not part of the neural-network input.

### What NOT to Use (Single Topology)
- `--multi-buffer` - Not needed, single topology
- `--balanced-sampling` - Not applicable
- `--ewc-*` flags - EWC is for preventing forgetting across topologies
- `--lr` override - Default `1e-4` is optimal

---

## Multi-Topology Training

For training a **single generalized model** that performs well on both fat-tree and leaf-spine topologies.

### Strategy: Sequential Fine-Tuning with EWC

Train on each topology sequentially while using Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.

### Phase 1: Train on Fat-Tree k=4

```bash
python3 rl_agent_4.py --mode train \
    --config config/topologies/fat_tree_k4.yaml \
    --steps 40000 \
    --multi-buffer \
    --compute-ewc \
    --traffic-weights "light:0.02,medium:0.02,high:0.50,bursty:0.46"
```

**Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Steps | 40,000 | Full exploration on first topology |
| Epsilon | `1.0 → 0.05` | Default decay |
| Learning Rate | `1e-4` | Default |
| `--multi-buffer` | Yes | Separate buffer for this topology |
| `--compute-ewc` | Yes | Save Fisher matrix for Phase 2 |

**Output:**
- Model checkpoint: `training_files/YYYYMMDD-HHMMSS-dqn_v4_best.pth`
- EWC file: `training_files/YYYYMMDD-HHMMSS-ewc.pth`

> **Note:** `--resume best` and `--ewc-file` automatically find the latest timestamped file.

### Phase 2: Fine-Tune on Leaf-Spine 16x4

```bash
python3 rl_agent_4.py --mode train \
    --config config/topologies/leaf_spine_16x4.yaml \
    --steps 40000 \
    --resume best \
    --resume-eps 0.30 \
    --lr 5e-5 \
    --multi-buffer \
    --ewc-file training_files/*-ewc.pth \
    --ewc-lambda 5000 \
    --balanced-sampling \
    --traffic-weights "light:0.02,medium:0.02,high:0.50,bursty:0.46"
```

**Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Steps | 40,000 | Matches epsilon decay schedule |
| `--resume best` | Yes | Continue from Phase 1 best model |
| `--resume-eps 0.30` | Yes | Reset epsilon for new topology exploration |
| `--lr 5e-5` | Yes | Lower LR prevents overwriting Phase 1 knowledge |
| `--ewc-lambda 5000` | Yes | EWC regularization strength |
| `--balanced-sampling` | Yes | Sample from both topology buffers |

### Phase 3: Validation Fine-Tuning (Optional)

Alternate between topologies to ensure both remain performant:

```bash
# Round 1: Fat-tree validation
python3 rl_agent_4.py --mode train \
    --config config/topologies/fat_tree_k4.yaml \
    --steps 5000 \
    --resume best \
    --resume-eps 0.10 \
    --lr 5e-5 \
    --multi-buffer \
    --balanced-sampling

# Round 2: Leaf-spine validation
python3 rl_agent_4.py --mode train \
    --config config/topologies/leaf_spine_16x4.yaml \
    --steps 5000 \
    --resume best \
    --resume-eps 0.10 \
    --lr 5e-5 \
    --multi-buffer \
    --balanced-sampling
```

### Multi-Topology Summary Table

| Phase | Topology | Steps | Epsilon | LR | EWC | Balanced |
|-------|----------|-------|---------|-----|-----|----------|
| 1 | fat-tree k=4 | 40,000 | 1.0→0.05 | 1e-4 | compute | No |
| 2 | leaf-spine 16x4 | 40,000 | 0.30→0.05 | 5e-5 | load (λ=5000) | Yes |
| 3a | fat-tree k=4 | 5,000 | 0.10→0.05 | 5e-5 | - | Yes |
| 3b | leaf-spine 16x4 | 5,000 | 0.10→0.05 | 5e-5 | - | Yes |

### Expected Total Training Time
- Phase 1: ~22 hours (40K steps)
- Phase 2: ~22 hours (40K steps)
- Phase 3: ~6 hours (optional)
- **Total: ~44-50 hours**

---

## Hyperparameter Reference

### Default Hyperparameters (from `rl_agent_4.py`)

```
# Network Architecture
RAW_STATE_DIM = 61       # 3×16 queue features + 3×3 batch features + 4 global/topology
STACK_SIZE = 16          # Observation/action history frames
ACTION_DIM = 14          # No-op + 12 queue-alt-K actions + multi-k1
STATE_DIM = 1200         # 61 obs × 16 frames + 14 actions × 16 frames
HIDDEN_DIM = 128         # Network hidden layer size

# Learning
LR = 1e-4                # Learning rate
GAMMA = 0.97             # Discount factor
BATCH_SIZE = 64          # Training batch size
MIN_REPLAY_SIZE = 500    # Start learning after this many transitions
REPLAY_CAPACITY = 80,000 # Replay buffer size

# Exploration
EPS_START = 1.0          # Initial epsilon
EPS_END = 0.05           # Final epsilon
EPS_DECAY_STEPS = 45,000 # Steps to decay epsilon

# Prioritized Experience Replay
PER_ALPHA = 0.6          # Prioritization exponent
PER_BETA_START = 0.4     # Initial importance sampling
PER_BETA_END = 1.0       # Final importance sampling

# Target Network
TAU = 0.005              # Soft update rate (Polyak averaging)

# Episode
MAX_EPISODE_STEPS = 100  # Steps per episode

# Timing
WINDOW_SECONDS = 1.0     # Observation window
DELAY_AFTER_ACTION = 1.0 # Wait after action for data

# Batch rerouting
K_CHOICES = (1, 2)
TOP_N_HOT_DEMANDS = 6
DEMAND_LOCK_STEPS = 5
```

### EWC Hyperparameters

```
EWC_LAMBDA = 5000.0      # Regularization strength (tune: 1000-10000)
EWC_FISHER_SAMPLES = 200 # Samples for Fisher matrix estimation
```

### Replay Buffer Defaults

```
buffer_capacity = 80,000 # Per topology when --multi-buffer is used
```

---

## Evaluation Commands

### Evaluate on Fat-Tree

```bash
python3 rl_agent_4.py --mode eval \
    --config config/topologies/fat_tree_k4.yaml \
    --weights-tag best \
    --steps 5000
```

### Evaluate on Leaf-Spine

```bash
python3 rl_agent_4.py --mode eval \
    --config config/topologies/leaf_spine_16x4.yaml \
    --weights-tag best \
    --steps 5000
```

### Compare with Baseline (No RL)

```bash
# Run without RL agent for comparison
python3 rl_agent_4.py --mode eval \
    --config config/topologies/fat_tree_k4.yaml \
    --baseline-only \
    --steps 2000
```

### Quick Checkpoint Smoke Benchmark

Use this after creating a checkpoint to verify the production path, local
telemetry cache, benchmark wrapper, batch reroute counters, and lockout
invariant:

```bash
sudo -E env PYTHONUNBUFFERED=1 python3 -u benchmark.py \
    --config config/topologies/fat_tree_k4.yaml \
    --profiles high_1,bursty_vi_3 \
    --methods rl \
    --repetitions 1 \
    --steps 20 \
    --max-retries 0 \
    --save-dir /path/to/checkpoints \
    --weights-tag final \
    --telemetry-backend cache \
    --telemetry-cache-socket /tmp/p4_qos_int_telemetry.sock \
    --production-influx-write off \
    --no-resume
```

Success criteria for the smoke benchmark:

- All selected runs complete with `status=success`.
- `valid_fraction >= 0.80`; for a healthy collector, expect `1.0`.
- Runner summaries verify routing, traffic, and telemetry state.
- No rerouted unit repeats within the five-step `(qid, dst_ip, bottleneck_sid)`
  lock window.
- Batch columns are present in the per-step CSVs:
  `requested_batch_size`, `batch_reroute_count`, `locked_units_count`, and
  `rerouted_units`.

---

## CLI Arguments Reference

### Core Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | train | `train` or `eval` |
| `--config` | str | None | Topology config YAML file |
| `--steps` | int | 50000 | Training steps |
| `--save-dir` | str | training_files | Checkpoint directory |

### Resume Training
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--resume` | str | None | Resume from checkpoint (`best`, `final`, `50pct`, or full path). Auto-finds latest timestamped file. |
| `--resume-eps` | float | None | Override epsilon when resuming |

### Learning Rate
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | float | 1e-4 | Override learning rate |

### EWC (Multi-Topology)
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ewc-lambda` | float | 0.0 | EWC regularization strength (0=disabled) |
| `--compute-ewc` | flag | False | Compute Fisher matrix after training |
| `--ewc-file` | str | None | Load EWC file (supports glob patterns, auto-finds latest if not found) |

### Multi-Buffer (Multi-Topology)
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--multi-buffer` | flag | False | Use separate buffers per topology |
| `--buffer-capacity` | int | 80000 | Replay buffer capacity; per topology with `--multi-buffer` |
| `--balanced-sampling` | flag | False | Balance sampling across topologies |

### Traffic
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--traffic-weights` | str | None | Traffic mix (e.g., "light:0.05,medium:0.05,high:0.35,bursty:0.55") |

---

## Available Topologies

| Topology | Config File | Switches | Hosts | Structure |
|----------|-------------|----------|-------|-----------|
| Fat-Tree k=4 | `config/topologies/fat_tree_k4.yaml` | 20 | 16 | 3-tier (ToR, Agg, Core) |
| Leaf-Spine 16x4 | `config/topologies/leaf_spine_16x4.yaml` | 20 | 32 | 2-tier (Leaf, Spine) |

---

## Monitoring & Checkpoints

### Checkpoint Files (Timestamped)
- `training_files/YYYYMMDD-HHMMSS-dqn_v4_best.pth` - Best model (by rolling avg reward)
- `training_files/YYYYMMDD-HHMMSS-dqn_v4_final.pth` - Final model
- `training_files/YYYYMMDD-HHMMSS-dqn_v4_25pct.pth` - 25% progress checkpoint
- `training_files/YYYYMMDD-HHMMSS-dqn_v4_50pct.pth` - 50% progress checkpoint
- `training_files/YYYYMMDD-HHMMSS-dqn_v4_75pct.pth` - 75% progress checkpoint
- `training_files/YYYYMMDD-HHMMSS-ewc.pth` - EWC Fisher matrix (if computed)

> **Auto-resolution:** When using `--resume best` or `--ewc-file latest`, the system automatically finds the most recent timestamped file matching the pattern.

### Key Metrics to Monitor
- **Episode Reward**: Should trend upward
- **Epsilon**: Should decay from 1.0 to 0.05
- **Loss**: Should stabilize (not necessarily decrease)
- **SLA Compliance**: Target >80% across all queues
- **First all-SLA step on high profiles**: Measures recovery speed from OSPF baseline
- **Burst `_3` churn**: Watch repeated reroutes of the same `(qid, dst_ip, bottleneck_sid)` unit
- **Batch usage**: `batch_reroute_count > 1` confirms K2 or `multi-k1` paths are exercised
