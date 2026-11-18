"""Neck convolutif sur une seule carte du backbone : tronc de la detection."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvTrunk(nn.Module):
    """Une carte du backbone (B, C, H, W) -> (B, channels, H, W) : deux conv 3x3 + BN + ReLU.

    La carte lue est celle dont le stride est le plus proche de `target_stride`.
    """

    out_kind = "map"

    def __init__(
        self, feature_info: list[dict], channels: int = 128, target_stride: int = 8
    ) -> None:
        super().__init__()
        # stride le plus proche de target_stride ; a egalite (ViT simple), le plus profond
        level = min(reversed(feature_info), key=lambda f: abs(f["stride"] - target_stride))
        self.name = level["name"]
        self.in_keys = [self.name]  # entrees lues dans le dict du backbone
        self.out_dim = channels

        self.conv1 = nn.Conv2d(level["channels"], channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        x = F.relu(self.bn1(self.conv1(feats[self.name])))
        return F.relu(self.bn2(self.conv2(x)))
