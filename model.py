"""
model.py
========
Deep Q-Network (DQN) architecture in PyTorch.

Architecture
------------
Input  : 11-dimensional state vector
Hidden : two fully-connected layers (256 → 256 units, ReLU activations)
Output : 3 Q-values — one per action (left, straight, right)

We use two identical networks:
  - Online  (policy) network  — updated every training step via gradient descent
  - Target              network  — updated slowly (soft or hard copy)
                                   provides stable Q-value targets and prevents
                                   the "chasing a moving target" divergence problem.

Why batch normalisation is *not* used here:
  Batch norm works best with large shuffled mini-batches.  In RL the mini-batch
  comes from the replay buffer and contains correlated transitions; LayerNorm
  would be safer, but experiments show plain ReLU networks train reliably for
  this problem size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Two-hidden-layer fully connected network.

    Parameters
    ----------
    state_size  : int   — dimensionality of the observation vector
    action_size : int   — number of discrete actions
    hidden_size : int   — width of each hidden layer (default 256)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_size,  hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        # Xavier uniform init keeps gradients well-scaled at the start
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor shape (batch, state_size)

        Returns
        -------
        q : Tensor shape (batch, action_size)  — one Q-value per action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)          # no activation on output — raw Q-values

    def save(self, path: str = "model.pth"):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
        print(f"[DQN] Weights saved → {path}")

    def load(self, path: str = "model.pth", device: torch.device = None):
        """Load model weights from disk."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"[DQN] Weights loaded ← {path}")
