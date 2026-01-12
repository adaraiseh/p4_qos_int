# DQN Training Plan for QoS Routing

This document provides training configurations for the RL-based QoS routing agent (`rl_agent_4.py`).

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
| Steps | 40,000-50,000 | Sufficient for convergence on single topology |
| Learning Rate | `1e-4` (default) | Optimal for stable training |
| Epsilon | `1.0 → 0.05` | Linear decay over 40K steps |
| Batch Size | 64 | Stable gradient estimates |
| Replay Buffer | 50,000 | Default capacity |

### Training Command

```bash
# Basic training on fat-tree k=4
python3 rl_agent_4.py --mode train \
    --config config/topologies/fat_tree_k4.yaml \
    --steps 55000

# With traffic mix (recommended, rebalanced for more stationary training)
python3 rl_agent_4.py --mode train \
    --config config/topologies/fat_tree_k4.yaml \
    --steps 55000 \
    --traffic-weights "light:0.02,medium:0.02,high:0.50,bursty:0.46"
```

### Alternative: Leaf-Spine Topology

```bash
python3 rl_agent_4.py --mode train \
    --config config/topologies/leaf_spine_16x4.yaml \
    --steps 55000 \
    --traffic-weights "light:0.02,medium:0.02,high:0.50,bursty:0.46"
```

### Expected Training Time
- ~50,000 steps × ~2s/step ≈ **28 hours**
- Checkpoints saved at 25%, 50%, 75%, best, and final

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
STATE_DIM = 464          # 50 metrics × 8 frames + 8 actions × 8 frames
ACTION_DIM = 8           # No-op + 6 single-queue + 1 multi-queue
HIDDEN_DIM = 128         # Network hidden layer size

# Learning
LR = 1e-4                # Learning rate
GAMMA = 0.97             # Discount factor
BATCH_SIZE = 64          # Training batch size
REPLAY_CAPACITY = 50,000 # Replay buffer size

# Exploration
EPS_START = 1.0          # Initial epsilon
EPS_END = 0.05           # Final epsilon
EPS_DECAY_STEPS = 40,000 # Steps to decay epsilon

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
DELAY_AFTER_ACTION = 0.8 # Wait after action for data
```

### EWC Hyperparameters

```
EWC_LAMBDA = 5000.0      # Regularization strength (tune: 1000-10000)
EWC_FISHER_SAMPLES = 200 # Samples for Fisher matrix estimation
```

### Multi-Buffer Defaults

```
buffer_capacity = 25,000 # Per-topology buffer size
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
| `--buffer-capacity` | int | 25000 | Buffer capacity per topology |
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
