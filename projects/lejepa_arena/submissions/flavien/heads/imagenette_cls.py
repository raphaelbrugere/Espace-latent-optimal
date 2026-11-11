"""Baseline classification head: the linear probe of eval_linear_probe.py."""
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, feats):
        return self.net(feats["embed"])


def build_head(feature_info, embed_dim, num_classes):
    return LinearProbe(embed_dim, num_classes)
