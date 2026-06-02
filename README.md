# DQN Snake Agent

A Deep Q-Network (DQN) reinforcement learning agent that learns to play Snake from scratch.

## Quick Start

```bash
# 1. Clone / download this folder
cd rl_snake_agent

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train (≈1000 episodes, ~5–15 min on CPU)
python train.py

# 5. Watch the trained agent play
python play.py --render

# 6. Plot the training curves
python plot_results.py
```

## File Overview

| File | Purpose |
|---|---|
| `environment.py` | Snake game — custom Gym-style env, 11-dim state |
| `model.py` | DQN neural network (PyTorch, 2-hidden-layer FC) |
| `agent.py` | DQN agent: replay buffer, ε-greedy, target network |
| `train.py` | Training loop, checkpointing, logging |
| `play.py` | Load trained model, evaluate & render |
| `plot_results.py` | Plot score and loss curves from saved JSON |

## Key Concepts

### State Representation (11 features)
- Danger straight/right/left (3 binary)
- Current heading direction (4 binary, one-hot)
- Food location relative to head (4 binary)

### DQN Tricks Used
1. **Experience Replay** — uniform random mini-batch sampling breaks temporal correlations
2. **Target Network** — prevents the "chasing a moving target" instability
3. **ε-greedy Decay** — linear decay from 1.0 → 0.01 over training
4. **Huber Loss** — robust to reward-scale outliers vs plain MSE
5. **Gradient Clipping** — prevents exploding gradients early in training

## Expected Results

| Milestone | Episodes | Avg Score |
|---|---|---|
| Random baseline | 0 | ~0.5 |
| Learns to eat | ~200 | ~3–5 |
| Consistent | ~500 | ~8–12 |
| Trained | 1000 | ~15–25 |

Scores above 20 mean the snake is consistently growing to 20+ cells on a 20×20 grid.

## Extending This Project

- **Pixel-based input**: Replace the 11-dim state with raw grid pixels → use a CNN backbone
- **Prioritized Replay**: Weight transitions by TD error magnitude (PER)
- **Dueling DQN**: Split Q into V(s) + A(s,a) for better value estimation
- **Double DQN**: Use online net to select actions, target net to evaluate → reduces overestimation
- **Atari games**: Swap `environment.py` for `gym.make("BreakoutNoFrameskip-v4")` and upgrade the model to a CNN
