#!/usr/bin/env python3
import os
import time
import random
import sys
import logging
from collections import deque
from math import cos, pi
from datetime import datetime, timedelta
import math
import glob
import re
from pathlib import Path
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from influxdb_client import InfluxDBClient, Point
from controller import Controller

# =========================================
#                 LOGGING
# =========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

# =========================================
#              GLOBAL PARAMS
# =========================================
# Telemetry window (seconds) & safety lag
WINDOW_SECONDS = 2
SAFETY_LAG_MS = 1000
INFLUX_QUERY_TIMEOUT = 5000
# Action timing
DELAY_NO_ACTION = 0.7
DELAY_AFTER_ACTION = 1.0
ACTION_COOLDOWN_SECS = 5.0

# =========================================
#              HYPERPARAMETERS
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

# Target updates (Polyak)
TAU = 0.005

# QoS thresholds (ms / drops / util)
VOICE_LAT_THRESH = 100.0
VOICE_DROP_THRESH = 2
VIDEO_LAT_THRESH = 150.0
VIDEO_DROP_THRESH = 5
BEST_EFFORT_LAT_THRESH = 200.0
SLA_NEAR_FACTOR = 0.9

# Queues
QIDS = (0, 1, 7)  # voice, video, best-effort

# Normalization caps
DROP_RATE_CAP_PER_100MS = 20.0
PENALTY_CAP = 1.0  # kept for helper, not directly used now
LATENCY_RATIO_CLIP = 2.0

# Fallbacks when no INT data in the window
MISSING_FLOW_LAT_MS    = 2000.0
MISSING_LINK_LAT_MS    = 1000.0
MISSING_SWITCH_LAT_MS  = 200.0
MISSING_DROP_PER_100MS = DROP_RATE_CAP_PER_100MS
MISSING_TX_UTIL_PCT    = 100.0
MISSING_Q_OCC_PCT      = 100.0

# Switch metric weights (for bottleneck scoring) — normalized later
SWITCH_METRIC_WEIGHTS = {
    'q_drop_rate_100ms': ('max', 0.6),
    'switch_latency':    ('max', 0.5),
    'tx_utilization':    ('max', 0.3),
    'queue_occupancy':   ('max', 0.2),
}

# =========================================
#              REWARD SHAPING
# =========================================
MAX_REWARD = 10.0

# Exponential decay scales (ms)
LAT_K_MS    = {0: 5.0, 1: 10.0, 7: 20.0}     # voice, video, best
JITTER_K_MS = {0: 2.0, 1: 5.0, 7: 10.0}

DROP_REF_PER_100MS = 1.0     # p95 drops/100ms above this start to hurt
UTIL_SOFTCAP = 80.0          # p95 tx_util above this starts to hurt

# Reward weights
W_LAT95   = 6.0
W_IMPROVE = 3.0
W_JITTER  = 1.5
W_DROP    = 2.0
W_UTIL    = 1.0

CHANGE_PENALTY     = 0.8
IMPROVE_EPSILON_MS = 0.5

# =========================================
#                   MODELS
# =========================================
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

# =========================================
#               REPLAY BUFFER
# =========================================
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

# =========================================
#                    AGENT
# =========================================
class DuelingDQNAgent:
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

    # ---------- Save/Load ----------
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
        try:
            st = torch.load(path, map_location=map_location or self.device, weights_only=False)
        except TypeError:
            st = torch.load(path, map_location=map_location or self.device)
        self.online.load_state_dict(st['online'])
        self.target.load_state_dict(st['target'])
        self.opt.load_state_dict(st['opt'])
        if 'scheduler' in st:
            self.scheduler.load_state_dict(st['scheduler'])

    # ---------- Eval helpers ----------
    def set_eval(self):
        self.online.eval()
        self.target.eval()
        self.eps = 0.0

    # ---------- Epsilon-greedy (unmasked) ----------
    def select_action(self, state_vec, decay_allowed: bool = True):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            warm_lr = LR * (self.step_count / float(self.warmup_steps))
            for g in self.opt.param_groups:
                g['lr'] = warm_lr
        self.last_lr = self.opt.param_groups[0]['lr']

        self.maybe_decay_epsilon(decay_allowed)
        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(q.argmax(dim=1).item())

    # ---------- Masked epsilon-greedy ----------
    def select_action_masked(self, state_vec, valid_mask, eps_override=None, no_op_priority=True, margin=0.05):
        eps_now = self.eps if eps_override is None else eps_override
        if eps_override is None:
            self.maybe_decay_epsilon(decay_allowed=True)

        vm = np.asarray(valid_mask, dtype=bool)
        valid_idxs = np.flatnonzero(vm)
        if len(valid_idxs) == 0:
            return 0

        if random.random() < eps_now:
            return int(np.random.choice(valid_idxs))

        with torch.no_grad():
            s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s).squeeze(0)
        q_np = q.detach().cpu().numpy()
        q_masked = np.full_like(q_np, -1e30, dtype=np.float32)
        q_masked[valid_idxs] = q_np[valid_idxs]
        a = int(np.argmax(q_masked))

        if no_op_priority and vm[0]:
            if a != 0 and (q_masked[a] - q_masked[0]) < margin:
                a = 0
        return a

    # ---------- Epsilon schedule ----------
    def _cosine_eps(self):
        p = min(1.0, self.decay_steps_used / max(1, EPS_DECAY_STEPS))
        return EPS_END + 0.5 * (EPS_START - EPS_END) * (1 + cos(pi * p))

    def _apply_adaptive_nudge(self):
        if len(self.recent_rewards) < 20:
            return 1.0
        avg_r = float(np.mean(self.recent_rewards))
        nudger = 1.0
        if avg_r > self.prev_avg_reward + 0.5:
            nudger = 0.98
        elif avg_r < self.prev_avg_reward - 0.5:
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
        self.rb.push(s0, a0, R, last_next_state, done_any, g)
        self.nstep_queue.popleft()

    def push_nstep(self, s, a, r, ns, d):
        self.nstep_queue.append({"s": s, "a": a, "r": r, "ns": ns, "d": d})
        self._emit_nstep_if_ready()
        if d:
            while len(self.nstep_queue) >= self.n_step or (self.nstep_queue and self.nstep_queue[0]["d"]):
                self._emit_nstep_if_ready()
            self.nstep_queue.clear()

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
        self.scheduler.step()
        self.last_loss = float(loss.item())

        with torch.no_grad():
            for tp, p in zip(self.target.parameters(), self.online.parameters()):
                tp.data.mul_(1.0 - TAU).add_(TAU * p.data)

# =========================================
#      ENVIRONMENT WITH CENTRAL STEP
# =========================================
class RoutingRLSystem:
    def __init__(self, bucket, token, org, url="http://localhost:8086"):
        self.bucket = bucket
        self.org = org
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=INFLUX_QUERY_TIMEOUT)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api()

        self.controller = Controller()

        self.switch_id_name   = dict(self.controller.switch_id_to_name)
        self.name_to_switch_id = dict(self.controller.switch_name_to_id)

        self.switch_metrics = SWITCH_METRIC_WEIGHTS

        # ===== State space v2 =====
        # qid one-hot: 3
        # global features: 12 (first = bneck_score_norm)
        # path features:
        #   hop_count one-hot (0..8; 8=≥8): 9
        #   bneck_pos_norm: 1
        #   alt_exists: 1
        #   path agg mean/max (sw_lat, txu, drop, qocc): 8
        # action context:
        #   last_action_kind one-hot (noop/change/revert): 3
        #   cooldown_remaining_norm: 1
        self.state_dim = 3 + 12 + (9 + 1 + 1 + 8) + 3 + 1  # = 38

        # Actions:
        # 0: no-op
        # 1: change route for the hottest demand (this qid)
        # 2: revert last change
        self.action_dim = 3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agents = {qid: DuelingDQNAgent(self.state_dim, self.action_dim, self.device, n_step=3) for qid in QIDS}

        # Timing/delays
        self.action_response_delay_no_change = DELAY_NO_ACTION
        self.action_response_delay_change = DELAY_AFTER_ACTION
        self.cooldown_secs = ACTION_COOLDOWN_SECS
        self.last_action_time = {qid: -float('inf') for qid in QIDS}
        self.last_action_taken = {qid: 0 for qid in QIDS}  # 0=noop,1=change,2=revert

        self.last_snapshot = self.collect_snapshot()

        # Churn regularizers
        self.W_ACTION_NONZERO = 0.10
        self.W_ACTION_SWITCH  = 0.15

    # ----- Time window helper -----
    def _time_window(self, seconds=WINDOW_SECONDS, lag_ms=SAFETY_LAG_MS):
        stop_dt = datetime.utcnow() - timedelta(milliseconds=lag_ms)
        start_dt = stop_dt - timedelta(seconds=seconds)
        start = start_dt.isoformat() + 'Z'
        stop  = stop_dt.isoformat() + 'Z'
        return start, stop

    # ----- One-hot & normalization -----
    def _one_hot(self, idx: int, size: int) -> np.ndarray:
        v = np.zeros(size, dtype=np.float32)
        if 0 <= idx < size:
            v[idx] = 1.0
        return v

    def _qid_one_hot(self, qid: int) -> np.ndarray:
        order = {0:0, 1:1, 7:2}
        return self._one_hot(order.get(qid, 2), 3)

    def _last_action_one_hot(self, a: int) -> np.ndarray:
        # clamp to {0,1,2}
        a = int(a) if a in (0,1,2) else 0
        return self._one_hot(a, 3)

    def _hop_one_hot_0to8(self, hop_count: int) -> np.ndarray:
        # bin: 0..8 (8 means >=8)
        b = 8 if hop_count >= 8 else max(0, int(hop_count))
        return self._one_hot(b, 9)

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

    # ----- Lighter snapshot collection -----
    def collect_snapshot(self, seconds=WINDOW_SECONDS):
        """
        Returns snapshot: dict{ qid -> dict with:
          global metrics, hottest demand, path features, bottleneck fields
        }
        """
        start, stop = self._time_window(seconds)
        qset = 'r.queue_id == "0" or r.queue_id == "1" or r.queue_id == "7"'

        # Global aggregates (single union query for all queues)
        flux_global = f'''
        base = from(bucket:"{self.bucket}")
        |> range(start:{start}, stop:{stop})
        |> filter(fn: (r) => {qset})
        |> toFloat()

        flow_mean = base
        |> filter(fn: (r) => r._measurement == "flow_latency")
        |> mean(column:"_value")
        |> set(key:"_measurement", value:"flow_latency_agg")

        link_mean = base
        |> filter(fn: (r) => r._measurement == "link_latency")
        |> mean(column:"_value")
        |> set(key:"_measurement", value:"link_latency_agg")

        sw_mean = base
        |> filter(fn: (r) => r._measurement == "switch_latency")
        |> mean(column:"_value")
        |> set(key:"_measurement", value:"switch_latency_agg")

        drop_mean = base
        |> filter(fn: (r) => r._measurement == "q_drop_rate_100ms")
        |> mean(column:"_value")
        |> set(key:"_measurement", value:"drop_rate_agg")

        txu_mean = base
        |> filter(fn: (r) => r._measurement == "tx_utilization")
        |> mean(column:"_value")
        |> set(key:"_measurement", value:"tx_util_agg")

        pre = base
        |> filter(fn: (r) => r._measurement == "flow_latency")
        |> aggregateWindow(every: 100ms, fn: mean, createEmpty: false)

        flow_p50 = pre
        |> quantile(q:0.50, method:"estimate_tdigest")
        |> set(key:"_measurement", value:"flow_p50")

        flow_p95 = pre
        |> quantile(q:0.95, method:"estimate_tdigest")
        |> set(key:"_measurement", value:"flow_p95")

        flow_std = base
        |> filter(fn: (r) => r._measurement == "flow_latency")
        |> stddev()
        |> set(key:"_measurement", value:"flow_stddev")

        drop_p95 = base
        |> filter(fn: (r) => r._measurement == "q_drop_rate_100ms")
        |> quantile(q:0.95, method:"estimate_tdigest")
        |> set(key:"_measurement", value:"drop_p95")

        txu_p95 = base
        |> filter(fn: (r) => r._measurement == "tx_utilization")
        |> quantile(q:0.95, method:"estimate_tdigest")
        |> set(key:"_measurement", value:"tx_util_p95")

        union(tables:[
          flow_mean, link_mean, sw_mean, drop_mean, txu_mean,
          flow_p50, flow_p95, flow_std, drop_p95, txu_p95
        ])
        '''

        snap = {qid: {
            "flow_latency_agg": None, "link_latency_agg": None, "switch_latency_agg": None,
            "drop_rate_agg": None, "tx_util_agg": None, "flow_p50": None, "flow_p95": None,
            "flow_stddev": None, "drop_p95": None, "tx_util_p95": None,
            # hottest demand fields
            "hot_src_ip": None, "hot_dst_ip": None,
            # path-derived fields
            "path_nodes": [],            #  stored path at snapshot time
            "bneck_name": None,          #  bottleneck switch name at snapshot time
            "hop_count": 0,
            "hop_oh_0to8": self._one_hot(0, 9),
            "bneck_sid": 0,
            "bneck_pos_norm": 0.0,
            "bneck_score_norm": 0.0,     # replaces penalty_worst_node
            "alt_exists": 0.0,
            "sw_lat_mean": 0.0, "sw_lat_max": 0.0,
            "txu_mean": 0.0, "txu_max": 0.0,
            "drop_mean": 0.0, "drop_max": 0.0,
            "qocc_mean": 0.0, "qocc_max": 0.0,
        } for qid in QIDS}

        # Fill global aggregates
        try:
            tables = self.query_api.query(org=self.org, query=flux_global)
            for tbl in tables or []:
                for rec in tbl.records:
                    m = rec.get_measurement()
                    qid_s = rec.values.get("queue_id")
                    try:
                        qid = int(qid_s)
                    except Exception:
                        continue
                    if qid not in snap:
                        continue
                    v = rec.get_value()
                    if v is None:
                        continue
                    snap[qid][m] = float(v)
        except Exception as e:
            log.debug("Global aggregate query failed: %s", e)

        # Attach hottest demand + path features (per qid -> compact queries)
        for qid in QIDS:
            # Hottest demand
            hot = self._pick_hottest_demand(qid, seconds=seconds)
            if hot:
                src_ip, dst_ip = hot
                snap[qid]["hot_src_ip"], snap[qid]["hot_dst_ip"] = src_ip, dst_ip
                # Path features + bottleneck scoring
                self._fill_path_features_for_hottest(qid, snap[qid], seconds=seconds)
            # Derive globals fallback if missing
            if snap[qid]["flow_latency_agg"] is None:
                snap[qid].update({
                    "flow_latency_agg":  MISSING_FLOW_LAT_MS,
                    "link_latency_agg":  MISSING_LINK_LAT_MS,
                    "switch_latency_agg":MISSING_SWITCH_LAT_MS,
                    "drop_rate_agg":     MISSING_DROP_PER_100MS,
                    "tx_util_agg":       MISSING_TX_UTIL_PCT,
                    "flow_p50":          MISSING_FLOW_LAT_MS,
                    "flow_p95":          MISSING_FLOW_LAT_MS,
                    "flow_stddev":       0.0,
                    "drop_p95":          MISSING_DROP_PER_100MS,
                    "tx_util_p95":       MISSING_TX_UTIL_PCT,
                })

        # Final derivations used elsewhere
        for qid, d in snap.items():
            thresh = {0: VOICE_LAT_THRESH, 1: VIDEO_LAT_THRESH, 7: BEST_EFFORT_LAT_THRESH}[qid]
            d["flow_latency"]        = float(d["flow_latency_agg"])
            d["mean_link_latency"]   = float(d["link_latency_agg"])
            d["mean_switch_latency"] = float(d["switch_latency_agg"])
            d["drop_rate"]           = float(d["drop_rate_agg"])
            d["mean_tx_util"]        = float(d["tx_util_agg"])
            d["near_sla"]            = bool(d["flow_latency"] >= SLA_NEAR_FACTOR * float(thresh))
        return snap

    def _pick_hottest_demand(self, qid: int, seconds=WINDOW_SECONDS):
        start, stop = self._time_window(seconds)
        flux = f'''
        from(bucket:"{self.bucket}")
          |> range(start:{start}, stop:{stop})
          |> filter(fn: (r) => r._measurement == "flow_latency" and r.queue_id == "{qid}")
          |> toFloat()
          |> group(columns:["src_ip","dst_ip"])
          |> mean(column:"_value")
          |> group()
          |> sort(columns:["_value"], desc:true)
          |> limit(n:1)
        '''
        try:
            tables = self.query_api.query(org=self.org, query=flux)
        except Exception as e:
            log.debug("hottest-demand query failed qid=%s: %s", qid, e)
            return None
        for tbl in tables or []:
            for rec in tbl.records:
                src = rec.values.get("src_ip")
                dst = rec.values.get("dst_ip")
                if src and dst:
                    return str(src), str(dst)
        return None

    def _fill_path_features_for_hottest(self, qid: int, d: dict, seconds=WINDOW_SECONDS):
        """
        Mutates snapshot 'd' in-place with path features & bottleneck details for the hottest demand.
        """
        src_ip, dst_ip = d.get("hot_src_ip"), d.get("hot_dst_ip")
        if not (src_ip and dst_ip):
            return

        # Resolve the current stored path and the switches on it
        path = self.controller.get_path_by_ips(src_ip, dst_ip)
        if not path:
            return
        d["path_nodes"] = list(path)

        sw_names = [n for n in path if isinstance(n, str) and n[:1] in ("t", "a", "c")]
        sw_ids = [self.controller.switch_name_to_id.get(n) for n in sw_names]
        sw_ids = [int(s) for s in sw_ids if s is not None]
        if not sw_ids:
            log.error(
                "No switch IDs resolved for path %s (names=%s); check rules/test mappings.",
                path, sw_names
            )
            return

        hop_count = len(sw_ids)
        d["hop_count"] = hop_count
        d["hop_oh_0to8"] = self._hop_one_hot_0to8(hop_count)
        if hop_count == 0:
            return

        # Query per-switch metrics along this path (single compact flux)
        sid_filter = " or ".join([f'r.switch_id == "{sid}"' for sid in sw_ids]) or 'true'
        start, stop = self._time_window(seconds)
        flux = f'''
        from(bucket:"{self.bucket}")
        |> range(start:{start}, stop:{stop})
        |> filter(fn: (r) => r.queue_id == "{qid}" and ({sid_filter})
            and r._measurement =~ /q_drop_rate_100ms|switch_latency|tx_utilization|queue_occupancy/)
        |> toFloat()
        |> group(columns:["queue_id","switch_id","_measurement"])
        |> max(column:"_value")
        |> group(columns:["switch_id"])
        |> pivot(rowKey:["switch_id"], columnKey:["_measurement"], valueColumn:"_value")
        '''
        try:
            wtables = self.query_api.query(org=self.org, query=flux)
        except Exception as e:
            log.error("path metrics query failed: %s", e)
            return

        # Collect per-switch metric rows
        rows = {}
        for tbl in wtables or []:
            for rec in tbl.records:
                sid = int(rec.values.get("switch_id", 0) or 0)
                if sid not in sw_ids:
                    continue
                rows[sid] = {
                    "drop": float(rec.values.get("q_drop_rate_100ms", 0.0) or 0.0),
                    "swlat": float(rec.values.get("switch_latency",    0.0) or 0.0),
                    "txu":   float(rec.values.get("tx_utilization",    0.0) or 0.0),
                    "qocc":  float(rec.values.get("queue_occupancy",   0.0) or 0.0),
                }
        if not rows:
            return

        # Normalized weights for scoring
        smw = getattr(self, "switch_metrics", SWITCH_METRIC_WEIGHTS)
        total_w = max(1e-9, sum(w for _, w in smw.values()))
        wts = {m: w / total_w for m, (_, w) in smw.items()}

        # Pick bottleneck switch using normalized metrics
        best_sid, best_score = None, -1.0
        for sid in sw_ids:
            r = rows.get(sid, {"drop": 0.0, "swlat": 0.0, "txu": 0.0, "qocc": 0.0})
            drop_n = self._norm_drop_rate(r["drop"])
            slat_n = self._latency_ratio(r["swlat"], qid)
            txu_n  = self._norm_tx_util(r["txu"])
            qocc_n = self._norm_tx_util(r["qocc"])  # queue occupancy is 0..100%
            score = (
                drop_n * wts["q_drop_rate_100ms"] +
                slat_n * wts["switch_latency"] +
                txu_n  * wts["tx_utilization"] +
                qocc_n * wts["queue_occupancy"]
            )
            if score > best_score:
                best_sid, best_score = sid, score

        # Position of bottleneck on the path and feasibility of an alternate
        try:
            b_idx = sw_ids.index(best_sid) if best_sid in sw_ids else -1
        except Exception:
            b_idx = -1
        bpos = 0.0 if b_idx < 0 else (b_idx / max(1, hop_count - 1))
        alt_exists = 1.0 if (b_idx >= 0 and self.controller.has_alternate_for_worst(best_sid, path)) else 0.0

        # Aggregates over path (raw, then normalized for the state vector)
        def nzmean(xs, default=0.0): return (sum(xs) / len(xs)) if xs else default
        swlat_vals = [rows[sid]["swlat"] for sid in sw_ids if sid in rows]
        txu_vals   = [rows[sid]["txu"]   for sid in sw_ids if sid in rows]
        drop_vals  = [rows[sid]["drop"]  for sid in sw_ids if sid in rows]
        qocc_vals  = [rows[sid]["qocc"]  for sid in sw_ids if sid in rows]

        # Store snapshot fields
        d["bneck_sid"]        = int(best_sid or 0)
        d["bneck_name"]       = self.switch_id_name.get(int(best_sid)) if best_sid else None
        d["bneck_pos_norm"]   = float(bpos)
        d["bneck_score_norm"] = float(max(0.0, min(1.0, best_score)))
        d["alt_exists"]       = float(alt_exists)

        # Store normalized aggregates for the state
        d["sw_lat_mean"] = self._latency_ratio(nzmean(swlat_vals), qid)
        d["sw_lat_max"]  = self._latency_ratio(max(swlat_vals) if swlat_vals else 0.0, qid)
        d["txu_mean"]    = self._norm_tx_util(nzmean(txu_vals))
        d["txu_max"]     = self._norm_tx_util(max(txu_vals) if txu_vals else 0.0)
        d["drop_mean"]   = self._norm_drop_rate(nzmean(drop_vals))
        d["drop_max"]    = self._norm_drop_rate(max(drop_vals) if drop_vals else 0.0)
        d["qocc_mean"]   = self._norm_tx_util(nzmean(qocc_vals))
        d["qocc_max"]    = self._norm_tx_util(max(qocc_vals) if qocc_vals else 0.0)


    # ----- Logging -----
    def _log_state(self, label: str, qid: int, d: dict):
        hot = f"{d.get('hot_src_ip','-')}→{d.get('hot_dst_ip','-')}"
        log.debug(
            f"[State][{label}][q={qid}] hot={hot} "
            f"bneck_sid={d.get('bneck_sid',0)} bpos={d.get('bneck_pos_norm',0.0):.2f} "
            f"bscore={d.get('bneck_score_norm',0.0):.2f} alt={int(d.get('alt_exists',0))} "
            f"hops={d.get('hop_count',0)} "
            f"flow(lat,p50,p95)={d.get('flow_latency',0):.2f}/{d.get('flow_p50',0):.2f}/{d.get('flow_p95',0):.2f}ms "
            f"link_mean={d['mean_link_latency']:.2f}ms sw_mean={d['mean_switch_latency']:.2f}ms "
            f"jitter≈{max(0.0, d.get('flow_p95',0.0)-d.get('flow_p50',0.0)):.2f}ms "
            f"drops(mean/p95)={d['drop_rate']:.3f}/{d.get('drop_p95',0.0):.3f} "
            f"tx_util(mean/p95)={d['mean_tx_util']:.2f}%/{d.get('tx_util_p95',0.0):.2f}% "
            f"near_sla={d['near_sla']}"
        )

    # ----- State vector (v2) -----
    def build_state_vector(self, qid, data, last_action_kind: int, cooldown_remaining_s: float):
        # (A) Queue one-hot
        qid_oh = self._qid_one_hot(qid)  # 3

        # (B) Global flow features (12) — first is bneck_score_norm (replaces penalty_worst_node)
        bneck_score = float(data.get('bneck_score_norm', 0.0))
        flowr = self._latency_ratio(data['flow_latency'], qid)
        linkr = self._latency_ratio(data['mean_link_latency'], qid)
        swr   = self._latency_ratio(data['mean_switch_latency'], qid)
        txn   = self._norm_tx_util(data['mean_tx_util'])
        drn   = self._norm_drop_rate(data['drop_rate'])
        near  = 1.0 if data['near_sla'] else 0.0

        p50v = data.get('flow_p50', data['flow_latency'])
        p95v = data.get('flow_p95', data['flow_latency'])
        p50r  = self._latency_ratio(p50v, qid)
        p95r  = self._latency_ratio(p95v, qid)
        jitter = max(0.0, p95v - p50v)
        sla_for_q = {0: VOICE_LAT_THRESH, 1: VIDEO_LAT_THRESH, 7: BEST_EFFORT_LAT_THRESH}[qid]
        jitter_r = float(min(max(jitter / (2.0 * sla_for_q), 0.0), 1.0))

        drop95_n = self._norm_drop_rate(data.get('drop_p95', data['drop_rate']))
        txu95_n  = self._norm_tx_util(data.get('tx_util_p95', data['mean_tx_util']))

        global_cont = np.array(
            [bneck_score, flowr, linkr, swr, txn, drn, near, p50r, p95r, jitter_r, drop95_n, txu95_n],
            dtype=np.float32
        )  # 12

        # (C) Hottest-path features:
        hop_oh = np.asarray(data.get("hop_oh_0to8", self._one_hot(0, 9)), dtype=np.float32)  # 9
        bpos   = float(data.get("bneck_pos_norm", 0.0))  # 1
        aexists= float(data.get("alt_exists", 0.0))      # 1
        path_aggs = np.array([
            float(data.get("sw_lat_mean", 0.0)),
            float(data.get("sw_lat_max",  0.0)),
            float(data.get("txu_mean",    0.0)),
            float(data.get("txu_max",     0.0)),
            float(data.get("drop_mean",   0.0)),
            float(data.get("drop_max",    0.0)),
            float(data.get("qocc_mean",   0.0)),
            float(data.get("qocc_max",    0.0)),
        ], dtype=np.float32)  # 8

        # (D) Action context
        last_act_oh = self._last_action_one_hot(last_action_kind)  # 3
        cool_norm = max(0.0, min(1.0, cooldown_remaining_s / max(1e-6, self.cooldown_secs)))  # 1

        # Final vector: 3 + 12 + (9+1+1+8) + 3 + 1 = 38
        return np.concatenate([qid_oh, global_cont, hop_oh, np.array([bpos, aexists], np.float32),
                               path_aggs, last_act_oh, np.array([cool_norm], np.float32)], axis=0)

    # ----- Health gate -----
    def _should_consider_change(self, qid:int, d:dict) -> bool:
        lat95 = float(d.get('flow_p95', d.get('flow_latency', 0.0)))
        sla   = {0: VOICE_LAT_THRESH, 1: VIDEO_LAT_THRESH, 7: BEST_EFFORT_LAT_THRESH}[qid]
        lat_ratio = lat95 / max(1e-6, sla)
        drops95 = float(d.get('drop_p95', d.get('drop_rate', 0.0)))
        txu95   = float(d.get('tx_util_p95', d.get('mean_tx_util', 0.0)))
        if lat_ratio < 0.9 and drops95 <= 2.0 and txu95 <= 85.0:
            return False
        return True

    # ----- Actions -----
    def apply_path_change(self, qid, data, action):
        if action == 0:
            log.debug("NO change (action 0)")
            return

        if action == 2:
            ok = self.controller.revert_last_change()
            if ok:
                log.info("[REVERT] Last route change was successfully reverted.")
            else:
                log.info("[REVERT] No change to revert.")
            return

        # action == 1: operate on the hottest demand path (use snapshot data)
        src_ip, dst_ip = data.get('hot_src_ip'), data.get('hot_dst_ip')
        bneck_sid = int(data.get('bneck_sid', 0) or 0)

        if not (src_ip and dst_ip):
            log.error(f"qid {qid} no hottest demand available; skipping path change.")
            return
        if bneck_sid <= 0:
            log.error(f"qid {qid} no bottleneck detected along hottest path; skipping.")
            return

        # Always re-check against the CURRENT path (another queue may have changed it already)
        path = self.controller.get_path_by_ips(src_ip, dst_ip)
        if not path:
            log.error(f"qid {qid} no stored path for {src_ip}->{dst_ip}; skipping.")
            return

        worst_name_snapshot = data.get('bneck_name') or self.switch_id_name.get(bneck_sid, None)
        if worst_name_snapshot not in path:
            # Snapshot is stale (e.g., another queue replaced this node). Skip to avoid bad/invalid logs.
            log.debug(f"qid {qid} snapshot worst {bneck_sid}({worst_name_snapshot}) is not on current path {path}; skipping change.")
            return

        # Re-check feasibility on the CURRENT path (controller-only logic; no Influx calls)
        if not self.controller.has_alternate_for_worst(bneck_sid, path):
            log.debug(f"qid {qid} no alternate available for worst {bneck_sid}({worst_name_snapshot}) on current path {path}; skipping.")
            return

        alt = self.controller.find_alternate_for_worst(bneck_sid, path)
        if not alt:
            log.debug(f"qid {qid} no alternate found for worst {bneck_sid}({worst_name_snapshot}) on current path {path}; skipping.")
            return

        ok, details = self.controller.reroute_one_demand_symmetric(
            src_ip=src_ip,
            dst_ip=dst_ip,
            qid=qid,
            worst_switch_id=bneck_sid,
            alt_switch_name=alt
        )
        if ok:
            last_change = self.controller.change_history[-1] if self.controller.change_history else {}
            fwd = last_change.get("fwd"); rev = last_change.get("rev")
            log.info(
                f"[PATH CHANGE SUCCESS] q={qid} demand=({src_ip} → {dst_ip}) "
                f"bneck={bneck_sid} alt={alt}"
            )
            if fwd and fwd.get("old_path") and fwd.get("new_path"):
                log.info("  Forward: \n%s\n%s", " -> ".join(fwd["old_path"]), " -> ".join(fwd["new_path"]))
            if rev and rev.get("old_path") and rev.get("new_path"):
                log.info("  Reverse: \n%s\n%s", " -> ".join(rev["old_path"]), " -> ".join(rev["new_path"]))
        else:
            log.error(
                f"[PATH CHANGE FAILED] q={qid} bneck_sid={bneck_sid} "
                f"alt={alt} demand=({src_ip} → {dst_ip}) reason={details}"
            )
    # ----- Action masking -----
    def valid_action_mask(self, qid, data):
        """
        [NOOP, CHANGE, REVERT]
        'CHANGE' allowed iff we have hottest demand, a detected bottleneck, and an alternate exists for it.
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[0] = True  # no-op always allowed

        src_ip, dst_ip = data.get('hot_src_ip'), data.get('hot_dst_ip')
        bneck_sid = int(data.get('bneck_sid', 0) or 0)
        alt_exists = bool(int(data.get('alt_exists', 0)))

        if src_ip and dst_ip and bneck_sid > 0 and alt_exists:
            mask[1] = True

        if self.controller.has_pending_change():
            mask[2] = True

        return mask

    # ----- Diagnostics -----
    def list_active_demands(self, seconds=WINDOW_SECONDS):
        start, stop = self._time_window(seconds)
        flux = f'''
        from(bucket:"{self.bucket}")
          |> range(start:{start}, stop:{stop})
          |> filter(fn: (r) => r._measurement == "flow_latency")
          |> group(columns:["src_ip","dst_ip","queue_id"])
          |> count(column:"_value")
          |> filter(fn: (r) => r._value > 0)
          |> keep(columns:["src_ip","dst_ip","queue_id","_value"])
        '''
        try:
            tables = self.query_api.query(org=self.org, query=flux)
        except Exception as e:
            log.warning("Failed to list active demands: %s", e)
            return []

        out = []
        for tbl in tables or []:
            for rec in tbl.records:
                src = rec.values.get("src_ip")
                dst = rec.values.get("dst_ip")
                qid = rec.values.get("queue_id")
                if src and dst and qid is not None:
                    try:
                        out.append((int(qid), str(src), str(dst)))
                    except Exception:
                        pass
        return out

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

    # ----- Training metrics writer -----
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
            if not (isinstance(s["last_loss"], float) and (np.isnan(s["last_loss"]) or np.isinf(s["last_loss"]))):
                p = p.field("last_loss", float(s["last_loss"]))
            points.append(p)
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
        except Exception as e:
            log.warning("Failed to write training metrics: %s", e)

    # ----- Reward (unchanged semantics) -----
    def _exp_score(self, x_ms: float, k_ms: float) -> float:
        x = max(0.0, float(x_ms))
        return math.exp(-x / max(1e-6, k_ms))

    def compute_reward_v2(self, qid:int, cur:dict, prev_snap:dict, action:int, prev_action:int=None):
        lat50 = float(cur.get('flow_p50', cur['flow_latency']))
        lat95 = float(cur.get('flow_p95', cur['flow_latency']))
        jitter = max(0.0, lat95 - lat50)
        drop95 = float(cur.get('drop_p95', cur['drop_rate']))
        util95 = float(cur.get('tx_util_p95', cur['mean_tx_util']))

        prev95 = float(prev_snap.get(qid, {}).get('flow_p95', prev_snap.get(qid, {}).get('flow_latency', lat95)))
        d_improve = max(-10000.0, min(10000.0, prev95 - lat95))

        r_lat = W_LAT95 * self._exp_score(lat95, LAT_K_MS[qid])
        r_imp = W_IMPROVE * (d_improve / (LAT_K_MS[qid] * 3.0))
        r_imp = max(-W_IMPROVE, min(W_IMPROVE, r_imp))

        r_jit  = -W_JITTER * self._exp_score(jitter, JITTER_K_MS[qid]) * (jitter / (JITTER_K_MS[qid] + 1e-6))
        r_drop = -W_DROP   * max(0.0, (drop95 - DROP_REF_PER_100MS) / max(DROP_REF_PER_100MS, 1e-6))
        r_util = -W_UTIL   * max(0.0, (util95 - UTIL_SOFTCAP) / 20.0)

        r_change = 0.0
        if action != 0 and d_improve < IMPROVE_EPSILON_MS:
            r_change = -CHANGE_PENALTY

        r_total = r_lat + r_imp + r_jit + r_drop + r_util + r_change

        # Churn regularizers
        if prev_action is not None:
            if action != 0:
                r_total -= self.W_ACTION_NONZERO
            if action != prev_action:
                r_total -= self.W_ACTION_SWITCH

        r_total = max(-MAX_REWARD, min(MAX_REWARD, r_total))
        parts = {
            "r_total": r_total, "r_lat": r_lat, "r_imp": r_imp,
            "r_jit": r_jit, "r_drop": r_drop, "r_util": r_util, "r_change": r_change,
            "lat95": lat95, "lat50": lat50, "jitter": jitter, "drop95": drop95, "util95": util95,
            "prev95": prev95, "d_improve": d_improve
        }
        return r_total, parts

    def write_reward_metrics(self, qid:int, parts:dict):
        try:
            p = (Point("rl_reward")
                 .tag("queue_id", str(qid))
                 .field("r_total", float(parts["r_total"]))
                 .field("r_lat",   float(parts["r_lat"]))
                 .field("r_imp",   float(parts["r_imp"]))
                 .field("r_jit",   float(parts["r_jit"]))
                 .field("r_drop",  float(parts["r_drop"]))
                 .field("r_util",  float(parts["r_util"]))
                 .field("r_change",float(parts["r_change"]))
                 .field("lat95",   float(parts["lat95"]))
                 .field("lat50",   float(parts["lat50"]))
                 .field("jitter",  float(parts["jitter"]))
                 .field("drop95",  float(parts["drop95"]))
                 .field("util95",  float(parts["util95"]))
                 .field("prev95",  float(parts["prev95"]))
                 .field("d_improve", float(parts["d_improve"])) )
            self.write_api.write(bucket=self.bucket, org=self.org, record=[p])
        except Exception as e:
            log.debug("Failed to write rl_reward metrics: %s", e)

    # ----- Step (train) -----
    def step_all(self):
        snap_now = self.collect_snapshot()

        actions = {}
        states  = {}
        now_mono = time.monotonic()

        for qid in QIDS:
            self._log_state("now", qid, snap_now[qid])

            elapsed = now_mono - self.last_action_time[qid]
            cooldown_remaining = max(0.0, self.cooldown_secs - elapsed)
            in_cooldown = elapsed < self.cooldown_secs
            mask = self.valid_action_mask(qid, snap_now[qid])

            if in_cooldown:
                _ = self.agents[qid].select_action(np.zeros(self.state_dim, np.float32), decay_allowed=False)  # consume schedule
                a = 0
                log.debug(f"[COOLDOWN] q={qid} {elapsed:.1f}s/{self.cooldown_secs:.0f}s -> action=0")
            else:
                # Keep feasibility constraints from valid_action_mask(); 
                # exploration happens only among valid actions.
                s = self.build_state_vector(qid, snap_now[qid], self.last_action_taken[qid], cooldown_remaining)
                states[qid] = s
                a = self.agents[qid].select_action_masked(
                    s, mask, eps_override=None, no_op_priority=True, margin=0.05
                )
            if qid not in states:
                # Build state anyway for experience tuple (no-op context)
                s = self.build_state_vector(qid, snap_now[qid], self.last_action_taken[qid], cooldown_remaining)
                states[qid] = s
            actions[qid] = a

        any_changed = False
        for qid in QIDS:
            if actions[qid] != 0:
                self.apply_path_change(qid, snap_now[qid], actions[qid])
                self.last_action_time[qid] = now_mono
                any_changed = True
            else:
                log.debug(f"NO Path change q={qid}")

        delay = self.action_response_delay_change if any_changed else self.action_response_delay_no_change
        time.sleep(delay)

        snap_next = self.collect_snapshot()

        rewards = {}
        for qid in QIDS:
            self._log_state("next", qid, snap_next[qid])
            elapsed = time.monotonic() - self.last_action_time[qid]
            cooldown_remaining = max(0.0, self.cooldown_secs - elapsed)
            ns = self.build_state_vector(qid, snap_next[qid], actions[qid], cooldown_remaining)

            prev_a = self.last_action_taken[qid]
            r, parts = self.compute_reward_v2(qid, snap_next[qid], self.last_snapshot, actions[qid], prev_action=prev_a)
            self.last_action_taken[qid] = actions[qid]
            self.write_reward_metrics(qid, parts)

            done = False
            self.agents[qid].push_nstep(states[qid], actions[qid], r, ns, done)
            self.agents[qid].train_step()
            self.agents[qid].on_step_end(r)
            rewards[qid] = r

        self.last_snapshot = snap_next
        return rewards

    # ----- Step (eval-only) -----
    def step_all_eval(self):
        snap_now = self.collect_snapshot()
        actions = {}
        now_mono = time.monotonic()
        log.info("step[evaluate]")

        for qid in QIDS:
            self._log_state("now", qid, snap_now[qid])

            elapsed = now_mono - self.last_action_time[qid]
            cooldown_remaining = max(0.0, self.cooldown_secs - elapsed)
            in_cooldown = elapsed < self.cooldown_secs
            mask = self.valid_action_mask(qid, snap_now[qid])

            if in_cooldown:
                a = 0
                log.debug(f"[COOLDOWN][eval] q={qid} {elapsed:.1f}s/{self.cooldown_secs:.0f}s -> action=0")
            else:
                # More conservative in eval: only allow change if unhealthy (don’t force exploration)
                if not self._should_consider_change(qid, snap_now[qid]) and len(mask) >= 2:
                    mask[1] = False
                s = self.build_state_vector(qid, snap_now[qid], self.last_action_taken[qid], cooldown_remaining)
                a = self.agents[qid].select_action_masked(
                    s, mask, eps_override=0.0, no_op_priority=True, margin=0.05
                )
            actions[qid] = a

        any_changed = False
        for qid in QIDS:
            if actions[qid] != 0:
                self.apply_path_change(qid, snap_now[qid], actions[qid])
                self.last_action_time[qid] = now_mono
                any_changed = True
            else:
                log.debug(f"NO Path change q={qid} [eval]")

        delay = self.action_response_delay_change if any_changed else self.action_response_delay_no_change
        time.sleep(delay)

        snap_next = self.collect_snapshot()

        rewards = {}
        for qid in QIDS:
            self._log_state("next", qid, snap_next[qid])
            r, parts = self.compute_reward_v2(qid, snap_next[qid], self.last_snapshot, actions[qid], prev_action=self.last_action_taken[qid])
            self.write_reward_metrics(qid, parts)
            self.last_action_taken[qid] = actions[qid]
            rewards[qid] = r

        self.last_snapshot = snap_next
        return rewards

    # ----- Shutdown -----
    def shutdown(self):
        for api in (self.write_api, self.client):
            try:
                api.close()
            except Exception:
                pass

# =========================================
#                    MAIN
# =========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL for INT QoS – train/test (state v2)")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="train: learn and save; test: load weights and evaluate (no training)")
    parser.add_argument("--weights_dir", default="training_files",
                        help="Directory containing voice_*.pth, video_*.pth, best_*.pth")
    parser.add_argument("--tag", default="final",
                        help="Which tag to load (e.g., 50%%, 90%%, final)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Total steps in train OR eval loop")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--influx-url", default="http://192.168.201.1:8086")
    parser.add_argument("--influx-org", default="research")
    parser.add_argument("--influx-bucket", default="INT")
    parser.add_argument("--influx-token", default="0fO0ojKAANp-7aEehJHRDWEKE-cSNoIEHY2aK8dd1KI0VWpmO1GAsMJhRh_B1U8bXDIaozHMDVv1yEkCPm230w==")
    args = parser.parse_args()

    # Seeds & determinism
    SEED = 1337
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        cudnn.deterministic = True
        cudnn.benchmark = False

    env = None
    try:
        env = RoutingRLSystem(args.influx_bucket, args.influx_token, args.influx_org, args.influx_url)

        def load_final_checkpoints(weights_dir, tag, device):
            names = {0: "voice", 1: "video", 7: "best"}
            for qid, stem in names.items():
                path = os.path.join(weights_dir, f"{stem}_{tag}.pth")
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Missing checkpoint: {path}")
                env.agents[qid].load(path, map_location=device)
                env.agents[qid].set_eval()
                log.info(f"[LOAD] q={qid} ← {path}")

        if args.mode == "test":
            load_final_checkpoints(args.weights_dir, args.tag, env.device)
            for step in range(1, args.steps + 1):
                rewards = env.step_all_eval()
                if step % 10 == 0:
                    log.info(
                        f"[EVAL Step {step}] rewards:"
                        f" voice={rewards[0]:.2f}, video={rewards[1]:.2f}, best={rewards[7]:.2f}"
                    )
                if step % args.log_every == 0:
                    stats = env.agent_stats()
                    log.info(f"[Eval Agent Stats @ step {step}]")
                    for qid in (0, 1, 7):
                        s = stats[qid]
                        log.info(
                            f"  q={qid} | eps={s['eps']:.4f} | avgR@100={s['avg_reward_100']:.3f} "
                            f"| cooldown_left={s['cooldown_remaining_s']:.2f}s"
                        )
            sys.exit(0)

        # ---------- TRAIN MODE ----------
        total_steps = int(args.steps)
        save_points = {
            int(total_steps*0.5): '50%',
            int(total_steps*0.9): '90%',
            total_steps: 'final'
        }
        LOG_EVERY = int(args.log_every)

        for step in range(1, total_steps + 1):
            log.debug(f"going to step {step}")
            rewards = env.step_all()

            if step % 10 == 0:
                log.info(
                    f"[Step {step}] rewards:"
                    f" voice={rewards[0]:.2f}, video={rewards[1]:.2f}, best={rewards[7]:.2f}"
                )
                demands = env.list_active_demands(seconds=WINDOW_SECONDS)
                if not demands:
                    log.info("[Active Demands] none observed in the last window")
                else:
                    log.info("[Active Demands] %d demand(s) with resolved paths:", len(demands))
                    for qid, src_ip, dst_ip in sorted(demands):
                        path = env.controller.get_path_by_ips(src_ip, dst_ip)
                        sh = env.controller.host_from_ip(src_ip) or src_ip
                        dh = env.controller.host_from_ip(dst_ip) or dst_ip
                        if path:
                            log.info("  [q=%s] %s → %s: %s", qid, sh, dh, " -> ".join(path))
                        else:
                            log.info("  [q=%s] %s → %s: <no stored path>", qid, sh, dh)

            if step % LOG_EVERY == 0:
                stats = env.agent_stats()
                log.info(f"[Agent Stats @ step {step}]")
                for qid in (0, 1, 7):
                    s = stats[qid]
                    log.info(
                        f"  q={qid} | eps={s['eps']:.4f} | avgR@100={s['avg_reward_100']:.3f} "
                        f"| decay_used={s['decay_steps_used']} | cooldown_left={s['cooldown_remaining_s']:.2f}s "
                        f"| loss={s['last_loss'] if not np.isnan(s['last_loss']) else 'nan'} | lr={s['lr']:.6f}"
                    )
                env.write_training_metrics(step, stats)

            if step in save_points:
                tag = save_points[step]
                for qid, name in ((0, 'voice'), (1, 'video'), (7, 'best')):
                    try:
                        os.makedirs("training_files", exist_ok=True)
                        env.agents[qid].save(f"training_files/{name}_{tag}.pth")
                    except Exception as e:
                        log.warning("Failed to save %s checkpoint: %s", name, e)
                log.info(f"Weights saved at {tag}")

    except KeyboardInterrupt:
        log.info("Interrupted by user. Shutting down gracefully...")
    finally:
        if env is not None:
            env.shutdown()
