import torch
from torch import nn
from .activations import Sine


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64, num_layers: int = 1):
        """
        A simple MLP with Sine activations.
        - num_layers: number of hidden layers (each of size `hidden`).
          num_layers=1 reproduces the previous behavior (Linear->Sine->Linear).
        """
        super().__init__()
        layers = []
        if num_layers <= 0:
            # direct linear projection
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(Sine())
            for _ in range(max(0, num_layers - 1)):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(Sine())
            layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) -> output (..., out_dim)
        shape = x.shape
        return self.net(x.view(-1, shape[-1])).view(*shape[:-1], -1)
