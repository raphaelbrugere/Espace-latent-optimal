"""Neck convolutif sur une seule carte du backbone : tronc de la detection."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvTrunk(nn.Module):
    """Une carte du backbone (B, C, H, W) -> (B, channels, H, W) : deux conv 3x3 + BN + ReLU.

    `level` : nom de la carte de `feature_info` a lire (meme convention que `levels` de
    FPNNeck). Le defaut est dans configs/neck/conv_trunk.yaml.
    """

    out_kind = "map"

    def __init__(self, feature_info: list[dict], level: str, channels: int = 128) -> None:
        super().__init__()
        by_name = {f["name"]: f for f in feature_info}
        if level not in by_name:
            raise ValueError(f"niveaux inconnus du backbone : {[level]} (choix : {list(by_name)})")
        self.name = level
        self.in_keys = [self.name]  # entrees lues dans le dict du backbone
        self.out_dim = channels

        self.conv1 = nn.Conv2d(by_name[level]["channels"], channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        x = F.relu(self.bn1(self.conv1(feats[self.name])))
        return F.relu(self.bn2(self.conv2(x)))
