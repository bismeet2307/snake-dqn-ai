"""
train.py
========
Main training loop for the DQN Snake agent.

What happens each episode
--------------------------
1. Reset the environment → get initial state s₀
2. Loop until done:
     a. Agent selects action aₜ using ε-greedy policy
     b. Environment steps forward → (sₜ₊₁, rₜ, done, info)
     c. Transition (sₜ, aₜ, rₜ, sₜ₊₁, done) stored in replay buffer
     d. One gradient update step on a sampled mini-batch
3. Decay ε after the episode ends
4. Log metrics, save checkpoint if new best score

Run
---
    python train.py

Optional arguments can be changed via the CONFIG dict below.
"""

import os
import time
import json
import numpy as np

from environment import SnakeEnv
from agent import DQNAgent


# ─── Training config ───────────────────────────────────────────────────────────

CONFIG = {
    "n_episodes"      : 1000,       # total training episodes
    "max_steps"       : 2000,       # hard cap on steps per episode
    "log_every"       : 50,         # print summary every N episodes
    "save_every"      : 100,        # save checkpoint every N episodes
    "checkpoint_dir"  : "checkpoints",
    "results_file"    : "training_results.json",
}


# ─── Utility ───────────────────────────────────────────────────────────────────

def moving_average(values, window=20):
    """Compute centred moving average over a list."""
    if len(values) < window:
        return [np.mean(values)] * len(values)
    return [np.mean(values[max(0, i - window):i + 1]) for i in range(len(values))]


def print_header():
    print("=" * 65)
    print("   DQN Snake Agent — Training")
    print("=" * 65)
    print(f"  Episodes   : {CONFIG['n_episodes']}")
    print(f"  Max steps  : {CONFIG['max_steps']}")
    print(f"  Grid size  : 20 × 20")
    print("=" * 65)
    print(f"{'Episode':>10} {'Score':>8} {'Avg20':>8} {'ε':>8} {'Loss':>10} {'Time':>8}")
    print("-" * 65)


# ─── Main training function ────────────────────────────────────────────────────

def train():
    print_header()

    # Setup
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    env   = SnakeEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    # Logging
    scores        = []
    losses        = []
    best_score    = 0
    total_t_start = time.time()

    for episode in range(1, CONFIG["n_episodes"] + 1):
        ep_start = time.time()
        state    = env.reset()
        ep_score = 0
        ep_loss  = []

        for _ in range(CONFIG["max_steps"]):
            # 1. Choose action
            action = agent.select_action(state, training=True)

            # 2. Step environment
            next_state, reward, done, info = env.step(action)

            # 3. Store transition
            agent.remember(state, action, reward, next_state, done)

            # 4. Learn
            loss = agent.train_step()
            if loss is not None:
                ep_loss.append(loss)

            state     = next_state
            ep_score  = info["score"]

            if done:
                break

        # Post-episode updates
        agent.decay_epsilon()
        scores.append(ep_score)
        avg_loss = np.mean(ep_loss) if ep_loss else 0.0
        losses.append(avg_loss)

        # Track best model
        if ep_score > best_score:
            best_score = ep_score
            agent.save(os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))

        # Periodic checkpoint
        if episode % CONFIG["save_every"] == 0:
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"ep{episode}.pth")
            agent.save(ckpt_path)

        # Logging
        if episode % CONFIG["log_every"] == 0:
            avg20    = np.mean(scores[-20:])
            elapsed  = time.time() - ep_start
            print(
                f"{episode:>10} {ep_score:>8} {avg20:>8.1f} "
                f"{agent.epsilon:>8.3f} {avg_loss:>10.4f} {elapsed:>7.2f}s"
            )

    # Final summary
    total_time = time.time() - total_t_start
    print("=" * 65)
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Best score   : {best_score}")
    print(f"  Final avg-20 : {np.mean(scores[-20:]):.1f}")
    print("=" * 65)

    # Save all results for plotting
    results = {
        "scores"        : scores,
        "losses"        : losses,
        "moving_avg"    : moving_average(scores, 20),
        "best_score"    : best_score,
        "n_episodes"    : CONFIG["n_episodes"],
    }
    with open(CONFIG["results_file"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {CONFIG['results_file']}")

    # Save final model
    agent.save(os.path.join(CONFIG["checkpoint_dir"], "final_model.pth"))

    return agent, results


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
