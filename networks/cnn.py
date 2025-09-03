import torch
from torch import nn


class Sine(nn.Module):
    """Simple sine activation: sin(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SmallCNN(nn.Module):
    """Refinement CNN: recebe micro-imagem (B,C,11,11) e retorna micro-imagem (B,C,11,11)."""
    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            Sine(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            Sine(),
            nn.Conv2d(hidden, in_channels, kernel_size=3, padding=1),
            Sine(),  # output in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C, H, W)
        return self.net(x)
