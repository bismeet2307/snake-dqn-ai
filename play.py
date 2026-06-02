"""
play.py
=======
Load a trained model and watch the agent play Snake.

Usage
-----
    # Watch in terminal (ASCII)
    python play.py --model checkpoints/best_model.pth --episodes 5 --render

    # Quiet benchmark (just print stats)
    python play.py --model checkpoints/best_model.pth --episodes 100

The --render flag draws an ASCII grid after each step and pauses briefly
so you can follow the game in real time.
"""

import argparse
import time
import numpy as np

from environment import SnakeEnv
from agent import DQNAgent


def evaluate(model_path: str, n_episodes: int = 10, render: bool = False, delay: float = 0.05):
    """
    Run the trained agent for `n_episodes` episodes and print statistics.

    Parameters
    ----------
    model_path  : path to the .pth weights file
    n_episodes  : number of evaluation games
    render      : if True, print ASCII grid each step
    delay       : seconds to wait between frames when rendering
    """
    env   = SnakeEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(model_path)

    scores = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        done  = False
        steps = 0

        if render:
            print(f"\n{'═' * 24}  Episode {ep}  {'═' * 24}")

        while not done:
            # Greedy policy (no exploration)
            action = agent.select_action(state, training=False)
            state, _, done, info = env.step(action)
            steps += 1

            if render:
                import os
                os.system("clear" if os.name != "nt" else "cls")
                env.render()
                time.sleep(delay)

        ep_score = info["score"]
        scores.append(ep_score)
        print(f"Episode {ep:>4} | Score: {ep_score:>4} | Steps: {steps:>5}")

    # Summary statistics
    print("\n" + "─" * 40)
    print(f"  Episodes    : {n_episodes}")
    print(f"  Mean score  : {np.mean(scores):.2f}")
    print(f"  Max score   : {max(scores)}")
    print(f"  Min score   : {min(scores)}")
    print(f"  Std dev     : {np.std(scores):.2f}")
    print("─" * 40)

    return scores


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained DQN Snake agent")
    p.add_argument("--model",    type=str, default="checkpoints/best_model.pth",
                   help="Path to the .pth model weights")
    p.add_argument("--episodes", type=int, default=10,
                   help="Number of evaluation episodes")
    p.add_argument("--render",   action="store_true",
                   help="Render ASCII game to terminal")
    p.add_argument("--delay",    type=float, default=0.08,
                   help="Seconds between frames when rendering")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path = args.model,
        n_episodes = args.episodes,
        render     = args.render,
        delay      = args.delay,
    )
