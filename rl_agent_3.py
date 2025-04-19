#!/usr/bin/env python3

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from influxdb_client import InfluxDBClient
from controller import Controller

# =========================================
#            HYPERPARAMETERS
# =========================================
LR = 1e-3                   # Learning rate
GAMMA = 0.99                # Discount factor
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_SIZE = 500       # Minimum experiences in replay buffer before training
EPS_START = 1.0             # Starting epsilon for e-greedy
EPS_END = 0.01              # Final epsilon
EPS_DECAY_STEPS = 10_000    # Steps over which epsilon is decayed
SYNC_TARGET_STEPS = 1000    # Frequency to sync online -> target net
ROUTE_CHANGE_PENALTY = 1.0  # Penalty for toggling route

# Large positive reward if all constraints are met
REWARD_ALL_MEET = 10.0

# Weighted penalty factors for violations
LATENCY_PENALTY_FACTOR = 0.1
DROP_PENALTY_FACTOR = 0.2
TX_UTIL_PENALTY_FACTOR = 0.05

# QoS thresholds [based on your specs]
VOICE_LAT_THRESH = 100.0
VOICE_DROP_THRESH = 2

VIDEO_LAT_THRESH = 150.0
VIDEO_DROP_THRESH = 5

BEST_EFFORT_LAT_THRESH = 200.0

# For near-SLA boundary checks (90% threshold)
SLA_NEAR_FACTOR = 0.9

# We assume max queue depth = 64 (can be used for normalization if needed)


# =========================================
#     DUELING DQN NETWORK DEFINITION
# =========================================
class DuelingDQN(nn.Module):
    """
    A Dueling DQN with separate Value and Advantage streams.
    """
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # Common feature extractor
        self.fc_common = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        common_out = self.fc_common(x)
        value = self.value_stream(common_out)
        advantage = self.adv_stream(common_out)
        # Q = V + (A - mean(A)) 
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals


# =========================================
#         REPLAY BUFFER
# =========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


# =========================================
#       DOUBLE DUELING DQN AGENT
# =========================================
class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.online_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

        self.eps = EPS_START
        self.step_count = 0

    def select_action(self, state):
        self.step_count += 1
        self.eps = max(EPS_END, EPS_START - (self.step_count / EPS_DECAY_STEPS)*(EPS_START - EPS_END))
        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.online_net(state_t)
            return q_vals.argmax(dim=1).item()

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return

        if self.step_count % SYNC_TARGET_STEPS == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        # Current Q
        q_vals = self.online_net(states_t)
        current_q = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_q_vals_online = self.online_net(next_states_t)
            next_actions = next_q_vals_online.argmax(dim=1)
            next_q_vals_target = self.target_net(next_states_t)
            next_q = next_q_vals_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q[dones_t] = 0.0

        target_q = rewards_t + GAMMA * next_q
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weights(self, filename):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filename)
        print(f"Weights saved to {filename}")
    
    def load_weights(self, filename):
        checkpoint = torch.load(filename)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Weights loaded from {filename}")

# =========================================
#       ROUTING RL SYSTEM (ENV)
# =========================================
class RoutingRLSystem:
    """
    This environment cycles:
      1) Query InfluxDB / gather metrics for last 1s
      2) Build state
      3) Agent picks action
      4) Compute reward
      5) Possibly update route
      6) Store transition & train
    We do that separately for each queue (0, 1, 7).
    """
    def __init__(self, bucket, token, org, url="http://localhost:8086"):
        self.bucket = bucket
        self.token = token
        self.org = org
        self.url = url

        self.controller = Controller()  # Load the P4 switches
        self.influx_client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.query_api = self.influx_client.query_api()

        #[ASSUMBTION]
        # Action space:
        #   0 => do nothing
        #   1 => swap worst node -> alt1
        #   2 => swap worst node -> alt2
        #   3 => swap worst node -> alt3
        self.action_dim = 4

        # State dimension: 10
        # [queue_id, worst_node_id, penalty_worst_node, flow_latency, near_sla,
        #  max_link_latency, max_switch_latency, max_tx_util, sum_drop_counts, other_queues_threshold]
        self.state_dim = 10

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # One agent per queue
        self.agent_voice = DuelingDQNAgent(self.state_dim, self.action_dim, self.device)  # q=0
        self.agent_video = DuelingDQNAgent(self.state_dim, self.action_dim, self.device)  # q=1
        self.agent_best_effort = DuelingDQNAgent(self.state_dim, self.action_dim, self.device)   # q=7

    def step(self, qid):
        """
        Simulate a single step for the queue 'qid'.
        """
        data = self.collect_data(qid)
        current_state = self.build_state_vector(qid, data)

        # [ASSUMPTION] TODO - Implement actions to swap worst node with alternatives...
        # pick agent
        if qid == 0: # voice
            action = self.agent_voice.select_action(current_state)
        elif qid == 1: # vedio 
            action = self.agent_video.select_action(current_state)
        else: # BE
            action = self.agent_best_effort.select_action(current_state)

        reward = self.compute_reward(qid, data, action)

        if action != 0:
            self.apply_path_change(qid, data["worst_node_id"], action)

        next_data = self.collect_data(qid)
        next_state = self.build_state_vector(qid, next_data)
        done = False

        # Store transition & train
        if qid == 0:
            self.agent_voice.push_transition(current_state, action, reward, next_state, done)
            self.agent_voice.train_step()
        elif qid == 1:
            self.agent_video.push_transition(current_state, action, reward, next_state, done)
            self.agent_video.train_step()
        else:
            self.agent_best_effort.push_transition(current_state, action, reward, next_state, done)
            self.agent_best_effort.train_step()

        return next_state, reward, done

    def collect_data(self, qid):
        """
        Gather or query from InfluxDB for the last 1 second. 
        For demonstration, we generate random data:

          'worst_node_id': int,
          'penalty_worst_node': float,
          'flow_latency': float,
          'max_link_latency': float,
          'max_switch_latency': float,
          'max_tx_util': float,
          'sum_drop_counts': float,
          'other_queues_threshold': bool

        We'll also need to decide if near_sla (bool).
        """
        # In real code, build a flux query with qid filter. 
        # Here, random:
        data_dict = {
            "worst_node_id": random.randint(1, 12),                 # ID of the node with the highest violation for this queue
            "penalty_worst_node": round(random.uniform(0, 10), 2),  # how severe that node's violation is
            "flow_latency": round(random.uniform(50, 300), 2),      # end-to-end flow latency for this queue
            "max_link_latency": round(random.uniform(0, 80), 2),
            "max_switch_latency": round(random.uniform(0, 2), 2),
            "max_tx_util": round(random.uniform(0, 100), 2),
            "sum_drop_counts": float(random.randint(0, 15)),
            "other_queues_threshold": bool(random.getrandbits(1))
        }

        # near_sla logic:
        # voice(0) => near if lat >= 90ms
        # video(1) => near if lat >= 135ms
        # best_effort(7)  => near if lat >= 180ms
        lat = data_dict["flow_latency"]
        if qid == 0:
            near_sla = (lat >= SLA_NEAR_FACTOR * VOICE_LAT_THRESH)
        elif qid == 1:
            near_sla = (lat >= SLA_NEAR_FACTOR * VIDEO_LAT_THRESH)
        else:
            near_sla = (lat >= SLA_NEAR_FACTOR * BEST_EFFORT_LAT_THRESH)

        data_dict["near_sla"] = near_sla
        return data_dict

    def build_state_vector(self, qid, data):
        """
        Construct a 10-dim vector:
          0) queue_id (float)
          1) worst_node_id (float)
          2) penalty_worst_node (float)
          3) flow_latency (float)
          4) near_sla (0 or 1)
          5) max_link_latency
          6) max_switch_latency
          7) max_tx_util
          8) sum_drop_counts
          9) other_queues_threshold (0 or 1)
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = float(qid)                                           # (float) which queue class this agent is controlling
        state[1] = float(data["worst_node_id"])                         # ID of the node with the highest violation for this queue
        state[2] = data["penalty_worst_node"]                           # how severe that node's violation is
        state[3] = data["flow_latency"]                                 # end-to-end flow latency for this queue
        state[4] = 1.0 if data["near_sla"] else 0.0                     # boolean (0/1) if we are near the SLA boundary
        state[5] = data["max_link_latency"]                             # maximum link latency across all queues
        state[6] = data["max_switch_latency"]                           # maximum switch latency across all queues
        state[7] = data["max_tx_util"]                                  # maximum tx_util across all queues
        state[8] = data["sum_drop_counts"]                              # sum of drop counts across all queues
        state[9] = 1.0 if data["other_queues_threshold"] else 0.0       # boolean (0/1) if any other queue is near threshold
        return state

    def compute_reward(self, qid, data, action):
        """
        For the chosen queue qid, measure how well it meets QoS:
          - If meets constraints => + REWARD_ALL_MEET
          - Else partial penalties
          - Subtract route-change penalty if action != 0
        We'll do partial checks for latency & drops for the given qid. 
        We'll also consider max_tx_util, etc.

        We do a simpler single-queue check. 
        If you want "all classes must meet," you'd gather data for all queues.
        """
        reward = 0.0

        lat = data["flow_latency"]
        drops = data["sum_drop_counts"]  # aggregated across the network
        max_tx = data["max_tx_util"]
        penalty_worst = data["penalty_worst_node"]

        if qid == 0:  # voice
            if lat <= VOICE_LAT_THRESH and drops <= VOICE_DROP_THRESH:
                reward += REWARD_ALL_MEET
            else:
                lat_violation = max(0, lat - VOICE_LAT_THRESH)
                drop_violation = max(0, drops - VOICE_DROP_THRESH)
                reward -= (LATENCY_PENALTY_FACTOR * lat_violation)
                reward -= (DROP_PENALTY_FACTOR * drop_violation)
                reward -= (0.05 * penalty_worst)
        elif qid == 1:  # video
            if lat <= VIDEO_LAT_THRESH and drops <= VIDEO_DROP_THRESH:
                reward += REWARD_ALL_MEET
            else:
                lat_violation = max(0, lat - VIDEO_LAT_THRESH)
                drop_violation = max(0, drops - VIDEO_DROP_THRESH)
                reward -= (LATENCY_PENALTY_FACTOR * lat_violation)
                reward -= (DROP_PENALTY_FACTOR * drop_violation)
                reward -= (0.05 * penalty_worst)
        else:  # qid=7 best-effort
            if lat <= BEST_EFFORT_LAT_THRESH:
                reward += REWARD_ALL_MEET
            else:
                lat_violation = max(0, lat - BEST_EFFORT_LAT_THRESH)
                reward -= (LATENCY_PENALTY_FACTOR * lat_violation)
                reward -= (0.05 * penalty_worst)

        # penalize if max_tx > 50
        over_util = max(0, max_tx - 50.0)
        reward -= (TX_UTIL_PENALTY_FACTOR * over_util)

        # route change penalty
        if action != 0:
            reward -= ROUTE_CHANGE_PENALTY

        return reward

    def apply_path_change(self, qid, worst_node_id, action):
        """
        If action !=0 => "swap worst_node_id with alternative." 
        We'll just print. 
        In real code, you'd call controller.update_path(sw_name, prefix, etc.).
        """
        print(f"[INFO] Path change for queue={qid}, worst_node={worst_node_id}, action={action}")
        # Example if you had the real logic:
        # self.controller.update_path(
        #     sw_name="a1",
        #     dst_prefix="10.7.1.0/24",
        #     dscp="0x2E",     # for voice or "0x18" for video, etc.
        #     next_hop_ip="10.7.1.2",
        #     egress_port=2
        # )


# =========================================
#              MAIN LOOP
# =========================================
if __name__ == "__main__":
    INFLUX_URL = "http://localhost:8086"
    INFLUX_TOKEN = "pkyJUX9Itrw-y8YuTx3kLDAQ_VYyR_MxnyvtFmHwnRQOjDb7n2QBUFt7piMNgl9TU6IujEJpi8cMEKnwGs77dA=="
    INFLUX_ORG = "research"
    INFLUX_BUCKET = "INT"

    env = RoutingRLSystem(bucket=INFLUX_BUCKET,
                          token=INFLUX_TOKEN,
                          org=INFLUX_ORG,
                          url=INFLUX_URL)

    total_episodes = 10000
    save_points = {
        int(total_episodes * 0.5): "50%",
        int(total_episodes * 0.9): "90%",
        total_episodes: "final"
    }

    for episode in range(1, total_episodes + 1):
        ns0, r0, d0 = env.step(0)
        ns1, r1, d1 = env.step(1)
        ns7, r7, d7 = env.step(7)

        # Sleep 1s to allow "last second" data to gather
        time.sleep(1)

        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"[Episode {episode}] Rewards => voice={r0:.2f}, video={r1:.2f}, best_effort={r7:.2f}")

        # Save weights at designated milestones
        if episode in save_points:
            checkpoint_name = save_points[episode]
            env.agent_voice.save_weights(f"training_files/agent_voice_weights_{checkpoint_name}.pth")
            env.agent_video.save_weights(f"training_files/agent_video_weights_{checkpoint_name}.pth")
            env.agent_best_effort.save_weights(f"training_files/agent_best_effort_weights_{checkpoint_name}.pth")
            print(f"[INFO] Weights saved at {checkpoint_name} completion.")