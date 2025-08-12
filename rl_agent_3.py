#!/usr/bin/env python3
import os
import time
import random
import sys
import logging
from collections import deque
from math import cos, pi
from datetime import datetime, timedelta

import glob
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from influxdb_client import InfluxDBClient, Point

from controller import Controller

# =========================================
#            LOGGING
# =========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger()
log.setLevel(logging.INFO)

# =========================================
#            HYPERPARAMETERS
# =========================================
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 32
MIN_REPLAY_SIZE = 500
REPLAY_MEMORY_SIZE = 10000

# Epsilon schedule (cosine with adaptive nudges)
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 10_000

# Target updates
TAU = 0.005

ROUTE_CHANGE_PENALTY = 1.0
REWARD_ALL_MEET = 10.0

LATENCY_PENALTY_FACTOR = 0.1
DROP_PENALTY_FACTOR = 0.2
TX_UTIL_PENALTY_FACTOR = 0.05

VOICE_LAT_THRESH = 100.0
VOICE_DROP_THRESH = 2
VIDEO_LAT_THRESH = 150.0
VIDEO_DROP_THRESH = 5
BEST_EFFORT_LAT_THRESH = 200.0

SLA_NEAR_FACTOR = 0.9

QIDS = (0, 1, 7)  # voice, video, best-effort

# *** Read-safety: ignore the freshest not-yet-complete bucket ***
SAFETY_LAG_MS = 1000

# Normalization caps
DROP_RATE_CAP_PER_100MS = 20.0
PENALTY_CAP = 1.0
LATENCY_RATIO_CLIP = 2.0

# One-hot vocabulary size for worst_node_id (0 = none, 1..256 = raw id clamped)
WORST_ID_BUCKETS = 257

# =============== Models ===============

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc_common = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        z = self.fc_common(x)
        v = self.value_stream(z)
        a = self.adv_stream(z)
        return v + a - a.mean(dim=1, keepdim=True)

# =============== Replay Buffers ===============

class ReplayBuffer:
    """
    Stores (state, action, R_nstep, next_state, done, gpow)
    where gpow = GAMMA**k for the number of accumulated steps k (<= n_step).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def push(self, s, a, Rn, ns, d, gpow):
        if len(self.buf) < self.capacity:
            self.buf.append(None)
        self.buf[self.pos] = (s, a, Rn, ns, d, gpow)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d, gpow = zip(*batch)
        return np.array(s), a, r, np.array(ns), d, gpow

    def __len__(self):
        return len(self.buf)

# =============== Agent ===============

class DuelingDQNAgent:
    """
    Double DQN agent with dueling architecture.
    - SmoothL1 + grad clipping
    - Soft target updates (Polyak τ)
    - N-step returns with proper γ^k bootstrap
    - LR scheduling (cosine) with warmup; Adam eps=1e-5
    - Cooldown-aware epsilon decay + adaptive nudges
    """
    def __init__(self, state_dim, action_dim, device, n_step=3, warmup_steps=1000):
        self.device = device
        self.online = DuelingDQN(state_dim, action_dim).to(device)
        self.target = DuelingDQN(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.opt = optim.Adam(self.online.parameters(), lr=LR, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=20_000, eta_min=1e-5
        )
        self.warmup_steps = warmup_steps

        self.rb = ReplayBuffer(REPLAY_MEMORY_SIZE)
        self.action_dim = action_dim

        self.eps = EPS_START
        self.decay_steps_used = 0
        self.recent_rewards = deque(maxlen=100)
        self.prev_avg_reward = 0.0

        self.n_step = max(1, int(n_step))
        self.nstep_queue = deque()

        self.step_count = 0
        self.last_loss = None
        self.last_lr = LR

        self.loss_fn = nn.SmoothL1Loss()

    # ---------- Save/Load (for checkpoints) ----------
    def save(self, path):
        torch.save(
            {
                'online': self.online.state_dict(),
                'target': self.target.state_dict(),
                'opt': self.opt.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, path
        )

    def load(self, path, map_location=None):
        st = torch.load(path, map_location=map_location or self.device)
        self.online.load_state_dict(st['online'])
        self.target.load_state_dict(st['target'])
        self.opt.load_state_dict(st['opt'])
        if 'scheduler' in st:
            self.scheduler.load_state_dict(st['scheduler'])

    # ---------- Epsilon schedule ----------
    def _cosine_eps(self):
        p = min(1.0, self.decay_steps_used / max(1, EPS_DECAY_STEPS))
        return EPS_END + 0.5 * (EPS_START - EPS_END) * (1 + cos(pi * p))

    def _apply_adaptive_nudge(self):
        if len(self.recent_rewards) < 20:
            return 1.0
        avg_r = float(np.mean(self.recent_rewards))
        improve = 0.5
        degrade = 0.5
        nudger = 1.0
        if avg_r > self.prev_avg_reward + improve:
            nudger = 0.98
        elif avg_r < self.prev_avg_reward - degrade:
            nudger = 1.02
        self.prev_avg_reward = avg_r
        return nudger

    def maybe_decay_epsilon(self, decay_allowed: bool):
        if not decay_allowed:
            return
        self.decay_steps_used += 1
        base = self._cosine_eps()
        nudger = self._apply_adaptive_nudge()
        self.eps = max(EPS_END, min(1.0, base * nudger))

    # ---------- N-step assembly with γ^k ----------
    def _emit_nstep_if_ready(self):
        if not self.nstep_queue:
            return
        have_n = len(self.nstep_queue) >= self.n_step
        first_done = self.nstep_queue[0]["d"]
        if not have_n and not first_done:
            return

        R, g, k = 0.0, 1.0, 0
        done_any = False
        last_next_state = None
        for item in self.nstep_queue:
            R += g * item["r"]
            k += 1
            g *= GAMMA
            last_next_state = item["ns"]
            if item["d"]:
                done_any = True
                break
            if k >= self.n_step:
                break

        s0, a0 = self.nstep_queue[0]["s"], self.nstep_queue[0]["a"]
        # 'g' is GAMMA**k; for terminal transitions bootstrap will be zeroed later.
        self.rb.push(s0, a0, R, last_next_state, done_any, g)
        self.nstep_queue.popleft()

    def push_nstep(self, s, a, r, ns, d):
        self.nstep_queue.append({"s": s, "a": a, "r": r, "ns": ns, "d": d})
        self._emit_nstep_if_ready()
        if d:
            while self.nstep_queue:
                self._emit_nstep_if_ready()

    def select_action(self, state_vec, decay_allowed: bool = True):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            warm_lr = LR * (self.step_count / float(self.warmup_steps))
            for g in self.opt.param_groups:
                g['lr'] = warm_lr
        # NOTE: scheduler stepped in train_step(), not here
        self.last_lr = self.opt.param_groups[0]['lr']

        self.maybe_decay_epsilon(decay_allowed)
        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(q.argmax(dim=1).item())

    def on_step_end(self, reward: float):
        self.recent_rewards.append(float(reward))

    def train_step(self):
        if len(self.rb) < MIN_REPLAY_SIZE:
            self.last_loss = None
            return

        s, a, r, ns, d, gpow = self.rb.sample(BATCH_SIZE)
        s_t  = torch.as_tensor(s,      dtype=torch.float32, device=self.device)
        a_t  = torch.as_tensor(a,      dtype=torch.long,    device=self.device)
        r_t  = torch.as_tensor(r,      dtype=torch.float32, device=self.device)
        ns_t = torch.as_tensor(ns,     dtype=torch.float32, device=self.device)
        d_t  = torch.as_tensor(d,      dtype=torch.bool,    device=self.device)
        gp_t = torch.as_tensor(gpow,   dtype=torch.float32, device=self.device)

        q = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            na = self.online(ns_t).argmax(dim=1)
            qn = self.target(ns_t).gather(1, na.unsqueeze(1)).squeeze(1)
            qn[d_t] = 0.0
        tgt = r_t + gp_t * qn

        loss = self.loss_fn(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step()
        self.scheduler.step()  # <-- step scheduler only when we optimize
        self.last_loss = float(loss.item())

        with torch.no_grad():
            for tp, p in zip(self.target.parameters(), self.online.parameters()):
                tp.data.mul_(1.0 - TAU).add_(TAU * p.data)

# =============== Environment with CENTRALIZED STEP ===============

class RoutingRLSystem:
    def __init__(self, bucket, token, org, url="http://localhost:8086"):
        self.bucket = bucket
        self.org = org
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api()

        self.controller = Controller()

        # Topology info
        self.switch_names = list(self.controller.topo.get_p4switches().keys())
        self.switch_count = len(self.switch_names)
        
        print("switch names ", self.switch_names)
        print("switch count ", self.switch_count)

        # === Build a map: INT switch_id -> role ===
        self.rules_dir = Path("rules/test")
        self.switch_id_role = self._load_switch_id_roles(self.rules_dir)
        self.tor_ids = {sid for sid, role in self.switch_id_role.items() if role == "tor"}
        if not self.switch_id_role:
            log.warning("Could not map any switch_id to roles; worst-node filtering for ToRs disabled.")
        else:
            log.info("Role map loaded for INT switch_ids. ToR IDs filtered: %s", sorted(list(self.tor_ids)))

        # Switch-level metric weights
        self.switch_metrics = {
            'q_drop_rate_100ms': ('max', 0.6),
            'switch_latency':    ('max', 0.3),
            'tx_utilization':    ('max', 0.2),
            'queue_occupancy':   ('max', 0.2),
        }

        # State space:
        #  - 3-dim one-hot for qid
        #  - WORST_ID_BUCKETS for worst_node_id one-hot (0..256)
        #  - 7 continuous features (penalty, flowr, linkr, swr, txn, drn, near)
        self.state_dim = 3 + WORST_ID_BUCKETS + 7
        self.action_dim = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agents = {qid: DuelingDQNAgent(self.state_dim, self.action_dim, self.device, n_step=3) for qid in QIDS}

        self.action_response_delay = 1.0
        self.cooldown_secs = 5.0
        self.last_action_time = {qid: -float('inf') for qid in QIDS}
        self.last_snapshot = self.collect_snapshot()

    # ----- Query helpers -----

    def _time_window(self, seconds=1, lag_ms=SAFETY_LAG_MS):
        # IMPORTANT: shift the stop time *back* by a safety lag so we never
        # read the freshest (possibly incomplete) bucket.
        stop_dt = datetime.utcnow() - timedelta(milliseconds=lag_ms)
        start_dt = stop_dt - timedelta(seconds=seconds)
        start = start_dt.isoformat() + 'Z'
        stop  = stop_dt.isoformat() + 'Z'
        return start, stop

    def _query_metric(self, meas, qid, agg, seconds=1):
        start, stop = self._time_window(seconds)
        flux = f'''
        from(bucket:"{self.bucket}")
        |> range(start:{start}, stop:{stop})
        |> filter(fn: (r) => r._measurement == "{meas}" and r.queue_id == "{qid}")
        |> toFloat()
        |> group(columns: ["queue_id"])
        |> {agg}(column:"_value")
        '''
        tables = self.query_api.query(org=self.org, query=flux)
        if not tables:
            return 0.0
        for tbl in tables:
            if tbl.records:
                v = tbl.records[0].get_value()
                return float(v if v is not None else 0.0)
        return 0.0

    # ----- Normalization helpers -----

    def _one_hot(self, idx: int, size: int) -> np.ndarray:
        v = np.zeros(size, dtype=np.float32)
        if 0 <= idx < size:
            v[idx] = 1.0
        return v

    def _qid_one_hot(self, qid: int) -> np.ndarray:
        # map (0,1,7) -> (0,1,2)
        order = {0:0, 1:1, 7:2}
        return self._one_hot(order.get(qid, 2), 3)

    def _worst_switch_one_hot(self, switch_id: int) -> np.ndarray:
        # 0 => slot 0, otherwise 1..256 by raw ID clamped (no modulo collisions)
        idx = 0 if switch_id <= 0 else min(int(switch_id), WORST_ID_BUCKETS - 1)
        v = np.zeros(WORST_ID_BUCKETS, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _latency_ratio(self, value_ms: float, qid: int) -> float:
        sla = {0: VOICE_LAT_THRESH, 1: VIDEO_LAT_THRESH, 7: BEST_EFFORT_LAT_THRESH}[qid]
        ratio = 0.0 if sla <= 0 else float(value_ms) / float(sla)
        return float(min(max(ratio, 0.0), LATENCY_RATIO_CLIP)) / LATENCY_RATIO_CLIP

    def _norm_tx_util(self, v: float) -> float:
        return float(min(max(v, 0.0), 100.0)) / 100.0

    def _norm_drop_rate(self, v: float) -> float:
        return float(min(max(v, 0.0), DROP_RATE_CAP_PER_100MS)) / DROP_RATE_CAP_PER_100MS

    def _norm_penalty(self, v: float) -> float:
        return float(min(max(v, 0.0), PENALTY_CAP))

    # ----- Aggregations -----

    def _load_switch_id_roles(self, rules_dir: Path):
        """
        Parse rules_dir/*-commands.txt for:
        table_set_default process_int_transit.tb_int_insert init_metadata <ID>
        Infer role from filename prefix: t*=ToR, a*=Agg, c*=Core, else 'other'.
        Returns dict: { switch_id:int -> role:str }
        """
        mapping = {}
        if not rules_dir.exists():
            return mapping

        pattern = re.compile(
            r"table_set_default\s+process_int_transit\.tb_int_insert\s+init_metadata\s+(\d+)",
            re.IGNORECASE
        )

        for path in glob.glob(str(rules_dir / "*-commands.txt")):
            fname = Path(path).name  # e.g., t1-commands.txt
            role = "other"
            if fname.startswith("t"):
                role = "tor"
            elif fname.startswith("a"):
                role = "agg"
            elif fname.startswith("c"):
                role = "core"

            try:
                with open(path, "r") as f:
                    text = f.read()
                m = pattern.search(text)
                if m:
                    sid = int(m.group(1))
                    mapping[sid] = role
            except Exception as e:
                log.warning("Failed to parse %s: %s", path, e)

        return mapping


    def _compute_worst_node_for_qid(self, qid, seconds=1):
        start, stop = self._time_window(seconds)
        total_w = sum(w for _, w in self.switch_metrics.values())
        weights = {m: w / total_w for m, (_, w) in self.switch_metrics.items()}

        flux = f'''
        from(bucket:"{self.bucket}")
        |> range(start:{start}, stop:{stop})
        |> filter(fn: (r) =>
            r.queue_id == "{qid}" and
            r._measurement =~ /q_drop_rate_100ms|switch_latency|tx_utilization|queue_occupancy/
        )
        |> toFloat()
        |> group(columns: ["queue_id","switch_id","_measurement"])
        |> max(column: "_value")
        |> group(columns: ["switch_id"])
        |> pivot(rowKey:["switch_id"], columnKey:["_measurement"], valueColumn:"_value")
        '''
        tables = self.query_api.query(org=self.org, query=flux)

        worst_id, worst_score = 0, -float('inf')
        for tbl in tables:
            for rec in tbl.records:
                vals = rec.values
                sid = int(vals.get("switch_id", 0) or 0)

                # Skip ToR switches if we know the mapping
                if self.switch_id_role and sid in self.tor_ids:
                    continue

                drops = float(vals.get("q_drop_rate_100ms", 0.0) or 0.0)
                slat  = float(vals.get("switch_latency",    0.0) or 0.0)
                txu   = float(vals.get("tx_utilization",    0.0) or 0.0)
                qocc  = float(vals.get("queue_occupancy",   0.0) or 0.0)

                score = (
                    drops * weights['q_drop_rate_100ms'] +
                    slat  * weights['switch_latency'] +
                    txu   * weights['tx_utilization'] +
                    qocc  * weights['queue_occupancy']
                )
                if score > worst_score:
                    worst_id, worst_score = sid, score

        if worst_score == -float('inf'):
            return 0, 0.0

        return worst_id, worst_score

    def collect_snapshot(self, seconds=1):
        snap = {}
        for qid in QIDS:
            worst_id, penalty = self._compute_worst_node_for_qid(qid, seconds=seconds)

            flow_latency       = self._query_metric('flow_latency',     qid, 'mean', seconds)
            max_link_latency   = self._query_metric('link_latency',     qid, 'max',  seconds)
            max_switch_latency = self._query_metric('switch_latency',   qid, 'max',  seconds)
            drop_rate          = self._query_metric('q_drop_rate_100ms',qid, 'max',  seconds)
            max_tx_util        = self._query_metric('tx_utilization',   qid, 'max',  seconds)

            thresh = {0: VOICE_LAT_THRESH, 1: VIDEO_LAT_THRESH, 7: BEST_EFFORT_LAT_THRESH}[qid]
            near_sla = float(flow_latency) >= SLA_NEAR_FACTOR * float(thresh)

            snap[qid] = {
                'worst_node_id':      int(worst_id),
                'penalty_worst_node': float(penalty),
                'flow_latency':       float(flow_latency),
                'max_link_latency':   float(max_link_latency),
                'max_switch_latency': float(max_switch_latency),
                'drop_rate':          float(drop_rate),
                'max_tx_util':        float(max_tx_util),
                'near_sla':           bool(near_sla),
            }
        return snap

    # ----- Logging -----

    def _log_state(self, label: str, qid: int, d: dict):
        log.info(
            f"[State][{label}][q={qid}] "
            f"worst_node={d['worst_node_id']} "
            f"penalty={d['penalty_worst_node']:.3f} "
            f"flow_lat={d['flow_latency']:.2f}ms "
            f"link_lat_max={d['max_link_latency']:.2f}ms "
            f"switch_lat_max={d['max_switch_latency']:.2f}ms "
            f"drops_rate={d['drop_rate']:.3f} "
            f"tx_util_max={d['max_tx_util']:.2f}% "
            f"near_sla={d['near_sla']}"
        )

    # ----- State construction -----

    def build_state_vector(self, qid, data):
        qid_oh   = self._qid_one_hot(qid)
        worst_oh = self._worst_switch_one_hot(int(data['worst_node_id']))

        pen   = self._norm_penalty(data['penalty_worst_node'])
        flowr = self._latency_ratio(data['flow_latency'], qid)
        linkr = self._latency_ratio(data['max_link_latency'], qid)
        swr   = self._latency_ratio(data['max_switch_latency'], qid)
        txn   = self._norm_tx_util(data['max_tx_util'])
        drn   = self._norm_drop_rate(data['drop_rate'])
        near  = 1.0 if data['near_sla'] else 0.0

        cont = np.array([pen, flowr, linkr, swr, txn, drn, near], dtype=np.float32)
        return np.concatenate([qid_oh, worst_oh, cont], axis=0)

    # ----- Reward -----

    def compute_reward(self, qid, data, action):
        r = 0.0
        lat   = data['flow_latency']
        drops = data['drop_rate']
        tx    = data['max_tx_util']
        pen   = data['penalty_worst_node']

        if qid == 0:
            if lat <= VOICE_LAT_THRESH and drops <= VOICE_DROP_THRESH:
                r += REWARD_ALL_MEET
            else:
                r -= LATENCY_PENALTY_FACTOR * max(0.0, lat - VOICE_LAT_THRESH)
                r -= DROP_PENALTY_FACTOR * max(0.0, drops - VOICE_DROP_THRESH)
                r -= 0.05 * pen
        elif qid == 1:
            if lat <= VIDEO_LAT_THRESH and drops <= VIDEO_DROP_THRESH:
                r += REWARD_ALL_MEET
            else:
                r -= LATENCY_PENALTY_FACTOR * max(0.0, lat - VIDEO_LAT_THRESH)
                r -= DROP_PENALTY_FACTOR * max(0.0, drops - VIDEO_DROP_THRESH)
                r -= 0.05 * pen
        else:
            if lat <= BEST_EFFORT_LAT_THRESH:
                r += REWARD_ALL_MEET
            else:
                r -= LATENCY_PENALTY_FACTOR * max(0.0, lat - BEST_EFFORT_LAT_THRESH)
                r -= 0.05 * pen

        r -= TX_UTIL_PENALTY_FACTOR * max(0.0, tx - 50.0)
        if action != 0:
            r -= ROUTE_CHANGE_PENALTY

        return float(max(min(r, REWARD_ALL_MEET), -REWARD_ALL_MEET))

    # ----- Actions -----

    def apply_path_change(self, qid, worst_node_id, action):
        log.info(f"Path change q={qid} worst_node={worst_node_id} action={action}")
        # TODO: wire to self.controller.update_path(...)

    # ----- Action masking -----
    def valid_action_mask(self, qid, data):
        """
        Returns a boolean mask of shape [action_dim].
        a0 = no-op (always valid)
        a1.. = alternates; uses controller.count_alternates if available.
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        # a0 always valid
        mask[0] = True

        worst_id = int(data.get('worst_node_id', 0) or 0)
        if worst_id == 0:
            return mask  # only no-op valid

        # Try to query alternates; if controller lacks method, allow all alternates optimistically.
        n_alts = None
        if hasattr(self.controller, "count_alternates"):
            try:
                n_alts = int(max(0, self.controller.count_alternates(qid, worst_id)))
            except Exception as e:
                log.warning("count_alternates failed: %s", e)
                n_alts = None

        max_alt = (self.action_dim - 1) if n_alts is None else min(n_alts, self.action_dim - 1)
        for a in range(1, 1 + max_alt):
            mask[a] = True
        return mask

    # ----- Diagnostics -----

    def agent_stats(self):
        now_mono = time.monotonic()
        stats = {}
        for qid, agent in self.agents.items():
            avg_r = float(np.mean(agent.recent_rewards)) if len(agent.recent_rewards) else 0.0
            cooldown_remaining = max(0.0, self.cooldown_secs - (now_mono - self.last_action_time[qid]))
            stats[qid] = {
                "eps": agent.eps,
                "avg_reward_100": avg_r,
                "decay_steps_used": agent.decay_steps_used,
                "cooldown_remaining_s": cooldown_remaining,
                "last_loss": agent.last_loss if agent.last_loss is not None else float('nan'),
                "lr": agent.last_lr,
            }
        return stats

    # ----- Training metrics writer (change 9) -----

    def write_training_metrics(self, step, stats):
        ts = datetime.utcnow()
        points = []
        for qid in QIDS:
            s = stats[qid]
            p = (
                Point("rl_training")
                .tag("queue_id", str(qid))
                .field("step", int(step))
                .field("eps", float(s["eps"]))
                .field("avg_reward_100", float(s["avg_reward_100"]))
                .field("decay_steps_used", int(s["decay_steps_used"]))
                .field("cooldown_remaining_s", float(s["cooldown_remaining_s"]))
                .field("lr", float(s["lr"]))
                .time(ts)
            )
            # last_loss may be nan; Influx rejects NaN fields, so only add if finite
            if not (isinstance(s["last_loss"], float) and (np.isnan(s["last_loss"]) or np.isinf(s["last_loss"]))):
                p = p.field("last_loss", float(s["last_loss"]))
            points.append(p)
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
        except Exception as e:
            log.warning("Failed to write training metrics: %s", e)

    # ----- Main step -----

    def step_all(self):
        snap_now = self.collect_snapshot()

        actions = {}
        states  = {}
        now_mono = time.monotonic()
        for qid in QIDS:
            self._log_state("now", qid, snap_now[qid])
            s = self.build_state_vector(qid, snap_now[qid])
            states[qid] = s

            elapsed = now_mono - self.last_action_time[qid]
            in_cooldown = elapsed < self.cooldown_secs

            mask = self.valid_action_mask(qid, snap_now[qid])  # <-- action mask (change 3)

            if in_cooldown:
                _ = self.agents[qid].select_action(s, decay_allowed=False)
                a = 0
                remaining = self.cooldown_secs - elapsed
                log.info(f"[COOLDOWN] q={qid} {elapsed:.1f}s/{self.cooldown_secs:.0f}s (~{remaining:.1f}s left) -> action=0")
            else:
                a = self.agents[qid].select_action(s, decay_allowed=True)
                # Enforce mask: if chosen action invalid, fall back to no-op
                if not mask[a]:
                    a = 0
            actions[qid] = a

        any_changed = False
        for qid in QIDS:
            if actions[qid] != 0:
                self.apply_path_change(qid, snap_now[qid]['worst_node_id'], actions[qid])
                self.last_action_time[qid] = now_mono
                any_changed = True
            else:
                log.info(f"NO Path change q={qid}")
        if not any_changed:
            log.info("No path changes this step.")

        time.sleep(self.action_response_delay)

        snap_next = self.collect_snapshot()

        rewards = {}
        for qid in QIDS:
            self._log_state("next", qid, snap_next[qid])
            ns = self.build_state_vector(qid, snap_next[qid])
            r  = self.compute_reward(qid, snap_next[qid], actions[qid])
            self.agents[qid].push_nstep(states[qid], actions[qid], r, ns, False)
            self.agents[qid].train_step()
            self.agents[qid].on_step_end(r)
            rewards[qid] = r

        self.last_snapshot = snap_next
        return rewards

# ============================
#            MAIN
# ============================
if __name__ == "__main__":
    SEED = 1337
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        cudnn.deterministic = True
        cudnn.benchmark = False

    INFLUX_URL    = "http://localhost:8086"
    INFLUX_TOKEN  = "pkyJUX9Itrw-y8YuTx3kLDAQ_VYyR_MxnyvtFmHwnRQOjDb7n2QBUFt7piMNgl9TU6IujEJpi8cMEKnwGs77dA=="
    INFLUX_ORG    = "research"
    INFLUX_BUCKET = "INT"

    env = RoutingRLSystem(INFLUX_BUCKET, INFLUX_TOKEN, INFLUX_ORG, INFLUX_URL)
    total_steps = 10000
    save_points = {int(total_steps*0.5): '50%', int(total_steps*0.9): '90%', total_steps: 'final'}
    LOG_EVERY = 20

    for step in range(1, total_steps + 1):
        rewards = env.step_all()

        if step % 10 == 0:
            log.info(
                f"[Step {step}] rewards:"
                f" voice={rewards[0]:.2f}, video={rewards[1]:.2f}, best={rewards[7]:.2f}"
            )

        if step % LOG_EVERY == 0:
            stats = env.agent_stats()
            log.info(f"[Agent Stats @ step {step}]")
            for qid in QIDS:
                s = stats[qid]
                log.info(
                    f"  q={qid} | eps={s['eps']:.4f} | avgR@100={s['avg_reward_100']:.3f} "
                    f"| decay_used={s['decay_steps_used']} | cooldown_left={s['cooldown_remaining_s']:.2f}s "
                    f"| loss={s['last_loss'] if not np.isnan(s['last_loss']) else 'nan'} | lr={s['lr']:.6f}"
                )
            # Write training metrics to Influx (change 9)
            env.write_training_metrics(step, stats)

        if step in save_points:
            tag = save_points[step]
            for qid, name in ((0, 'voice'), (1, 'video'), (7, 'best')):
                # save checkpoints (change 9)
                try:
                    env.agents[qid].save(f"{name}_{tag}.pth")
                except Exception as e:
                    log.warning("Failed to save %s checkpoint: %s", name, e)
            log.info(f"Weights saved at {tag}")
