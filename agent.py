"""
agent.py
========
DQN Agent with:
  1. Experience Replay      — stores (s, a, r, s', done) transitions in a ring buffer
                              and samples random mini-batches to break temporal correlation.
  2. Target Network         — a slowly-updated copy of the online network used to
                              compute stable TD targets.  Updated via hard copy every
                              TARGET_UPDATE_FREQ steps.
  3. ε-greedy Exploration   — starts fully random, linearly decays to a small minimum
                              so the agent keeps exploring mildly even when well-trained.

Bellman update used (standard DQN):
    Q_target(s, a) = r  +  γ · max_{a'} Q_target(s', a')   (if not done)
    Q_target(s, a) = r                                       (if done)

    Loss = MSE( Q_online(s,a),  Q_target(s,a) )
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model import DQN


# ─── Hyper-parameters ──────────────────────────────────────────────────────────

MEMORY_SIZE        = 100_000    # max transitions stored in replay buffer
BATCH_SIZE         = 64         # transitions sampled per training step
GAMMA              = 0.99       # discount factor — how much future rewards matter
LR                 = 1e-3       # Adam learning rate
EPS_START          = 1.0        # initial exploration probability
EPS_MIN            = 0.01       # minimum exploration probability
EPS_DECAY          = 0.995      # multiplicative decay applied after each episode
TARGET_UPDATE_FREQ = 100        # hard-copy online → target every N training steps


# ─── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-size circular buffer storing (state, action, reward, next_state, done).

    `deque(maxlen=N)` automatically evicts the oldest transition when full,
    keeping memory usage bounded without explicit management.
    """

    def __init__(self, capacity: int = MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a single transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Randomly sample a mini-batch.

        Returns five numpy arrays ready for conversion to tensors:
        states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent.

    Parameters
    ----------
    state_size  : int   — observation dimensionality  (11 for Snake)
    action_size : int   — number of discrete actions   (3 for Snake)
    device      : str   — 'cpu' or 'cuda'
    """

    def __init__(self, state_size: int, action_size: int, device: str = None):
        self.state_size  = state_size
        self.action_size = action_size
        self.device      = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Agent] Using device: {self.device}")

        # Two identical networks
        self.online_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # target never trained directly

        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=LR)
        self.memory     = ReplayBuffer(MEMORY_SIZE)
        self.epsilon    = EPS_START
        self.steps_done = 0       # counts training steps (not episodes)

    # ── Interaction ────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        ε-greedy action selection.

        During training:   explore (random) with probability ε,
                           exploit (greedy) with probability 1-ε.
        During evaluation: always greedy (ε=0).
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        """Store one transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # ── Learning ───────────────────────────────────────────────────────────────

    def train_step(self) -> float | None:
        """
        Sample a mini-batch, compute the Bellman loss, and do one gradient step.

        Returns the scalar loss value (float), or None if the buffer is too small.
        """
        if len(self.memory) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Convert to tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # ── Online Q-values for the actions actually taken ─────────────────────
        # Q_online(s, a) for the a that was chosen
        current_q = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        # ── Target Q-values using the Bellman equation ────────────────────────
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            # If the episode ended (done=1), there is no future reward
            target_q = rewards_t + GAMMA * next_q * (1.0 - dones_t)

        # ── Huber loss (smooth L1) — more robust to outlier rewards ───────────
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients in early training
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1

        # Hard copy online → target periodically
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        """Call once per episode to reduce exploration rate."""
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str = "dqn_snake.pth"):
        self.online_net.save(path)

    def load(self, path: str = "dqn_snake.pth"):
        self.online_net.load(path, device=self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
