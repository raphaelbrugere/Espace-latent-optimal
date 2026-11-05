"""Baseline segmentation head: the FPN decoder of train_seg_ablation.py.

Channels are read from feature_info, so it runs on any backbone. On the
baseline ViT all maps share stride 8, so the "pyramid" is flat and logits come
out at 1/8 resolution: easy room for improvement!
"""
import torch.nn as nn
import torch.nn.functional as F


class FPNHead(nn.Module):
    def __init__(self, feature_info, num_classes, channels=64):
        super().__init__()
        self.names = [f["name"] for f in feature_info]
        self.lateral = nn.ModuleList([nn.Conv2d(f["channels"], channels, 1) for f in feature_info])
        self.smooth = nn.Conv2d(channels, channels, 3, padding=1)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(channels, num_classes, 1))

    def forward(self, feats):
        laterals = [conv(feats[n]) for conv, n in zip(self.lateral, self.names)]
        fused = laterals[-1]
        for lat in reversed(laterals[:-1]):  # top-down: coarse to fine
            fused = lat + F.interpolate(fused, size=lat.shape[-2:], mode="nearest")
        return self.classifier(self.smooth(fused))


def build_head(feature_info, embed_dim, num_classes):
    return FPNHead(feature_info, num_classes)
