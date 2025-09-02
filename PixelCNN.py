import torch
import torch.nn as nn
import torch.nn.functional as F


# Sine activation (SIREN-style)
class SineActivation(nn.Module):
        def __init__(self):
                super().__init__()

        def forward(self, x):
                return torch.sin(30.0 * x)


class PixelCNN(nn.Module):
        """
        Decoder that maps features (B, 256, 1, 1) -> microimage (B, 3, mi_h, mi_w).

        Implementation notes:
        - Apply a 1x1 conv to change channels, then a ConvTranspose2d that expands the spatial
            resolution from 1x1 directly to target (mi_h x mi_w). This avoids applying 3x3
            convolutions on 1x1 inputs which causes kernel>input errors.
        - After spatial expansion, use 3x3 convs with padding=1 to refine features while
            preserving spatial size.
        - Final 1x1 conv reduces to RGB and a sigmoid brings outputs to [0,1].
        """

        def __init__(self, in_channels=256, out_channels=3, mi_h=11, mi_w=11):
                super().__init__()
                self.mi_h = mi_h
                self.mi_w = mi_w

                # reduce channel dim from MLP
                self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1)
                self.act1 = SineActivation()

                # expand spatially from 1x1 to mi_h x mi_w in one transpose op
                # for an input of size 1, ConvTranspose2d output size = kernel_size when stride=1, padding=0
                # so kernel_size = mi_h (assumes square microimages)
                assert mi_h == mi_w, "This simple decoder expects square microimages"
                self.upsample = nn.ConvTranspose2d(128, 64, kernel_size=mi_h, stride=1, padding=0)
                self.act2 = SineActivation()

                # refinement convs (keep spatial size mi_h x mi_w)
                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.act3 = SineActivation()
                self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
                self.act4 = SineActivation()

                # final RGB output
                self.conv_out = nn.Conv2d(32, out_channels, kernel_size=1)
                self.sigmoid = nn.Sigmoid()

        def forward(self, x):
                # x: (B, in_channels, 1, 1)
                x = self.conv1(x)
                x = self.act1(x)

                x = self.upsample(x)  # -> (B, 64, mi_h, mi_w)
                x = self.act2(x)

                x = self.conv2(x)
                x = self.act3(x)

                x = self.conv3(x)
                x = self.act4(x)

                x = self.conv_out(x)
                x = self.sigmoid(x)
                return x
