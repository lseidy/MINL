import torch
from torch import nn


class Sine(nn.Module):
    """Simple sine activation: sin(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            Sine(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) -> output (..., out_dim)
        shape = x.shape
        return self.net(x.view(-1, shape[-1])).view(*shape[:-1], -1)
