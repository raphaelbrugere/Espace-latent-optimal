"""Baseline head for pets_det: soft-argmax localization on one feature map.

Uses the feature map whose stride is closest to 8. At every location it predicts:
  * an attention logit ("the head is here"),
  * the box width and height,
  * a sub-cell offset of the center.
A softmax over locations turns the logits into weights; the box and the class
are weighted averages over locations. Unlike a hard argmax (CenterNet), this is
differentiable, so the task's standard box loss trains it end to end.

Ideas to beat it: combine several feature levels, a sharper decoding, a
dedicated heatmap loss, or compare with the naive alternative (regress the box
from the pooled embedding) to see how much spatial features matter.
"""
import torch
import torch.nn as nn


class SoftArgmaxHead(nn.Module):
    def __init__(self, feature_info, num_classes, channels=128):
        super().__init__()
        # closest stride to 8; on ties (plain ViT: all stride 8) take the deepest level
        level = min(reversed(feature_info), key=lambda f: abs(f["stride"] - 8))
        self.name = level["name"]
        self.trunk = nn.Sequential(
            nn.Conv2d(level["channels"], channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.attention = nn.Conv2d(channels, 1, 1)
        self.geometry = nn.Conv2d(channels, 4, 1)   # offset x, offset y, width, height
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, feats):
        x = self.trunk(feats[self.name])
        B, _, H, W = x.shape
        weights = self.attention(x).flatten(1).softmax(1)                     # (B, H*W)
        ys, xs = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
        g = self.geometry(x).flatten(2)                                        # (B, 4, H*W)
        cx = ((xs.flatten() + g[:, 0].sigmoid()) / W * weights).sum(1)       # cell + offset, in [0, 1]
        cy = ((ys.flatten() + g[:, 1].sigmoid()) / H * weights).sum(1)
        bw = (g[:, 2].sigmoid() * weights).sum(1)
        bh = (g[:, 3].sigmoid() * weights).sum(1)
        box = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).clamp(0, 1)
        pooled = (x.flatten(2) * weights[:, None]).sum(2)                      # attention-pooled features
        return {"box": box, "logits": self.classifier(pooled)}


def build_head(feature_info, embed_dim, num_classes):
    return SoftArgmaxHead(feature_info, num_classes)
