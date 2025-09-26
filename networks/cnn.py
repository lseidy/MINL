import torch
from torch import nn
from .activations import Sine


class SmallCNN(nn.Module):
    """Refinement CNN over micro-images (B,C,H,W)."""
    def __init__(self, in_channels: int = 3, hidden: int = 64, num_blocks: int = 2, out_activation: bool = True):
        super().__init__()
        layers = []
        # first conv
        layers.append(nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1))
        layers.append(Sine())
        # middle conv blocks
        for _ in range(max(0, num_blocks - 1)):
            layers.append(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1))
            layers.append(Sine())
        # projection back to input channels
        layers.append(nn.Conv2d(hidden, in_channels, kernel_size=3, padding=1))
        if out_activation:
            layers.append(Sine())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C, H, W)
        return self.net(x)
