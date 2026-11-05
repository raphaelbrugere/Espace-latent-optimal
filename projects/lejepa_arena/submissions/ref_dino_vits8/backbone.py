"""Reference backbone: ViT-S/8 with DINO weights from timm (self-supervised, ImageNet-1k).

Same architecture as the baseline, so the comparison isolates the pretraining
method and data. Built WITHOUT downloading: the weights come from backbone.pt,
written by train/export_timm_weights.py. Pretrained at 224px, run here at the
league resolution (position embeddings resampled by timm).

External weights: reference only, shown unranked on the leaderboard.
"""
import timm
import torch.nn as nn

ARCH, IMG_SIZE = "vit_small_patch8_224.dino", 128


class TimmViTBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.vit = timm.create_model(ARCH, pretrained=pretrained, img_size=IMG_SIZE, num_classes=0,
                                     dynamic_img_size=True)
        self.blocks = [2, 5, 8, 11]
        self.embed_dim = self.vit.num_features            # 384: the CLS token, as in DINO's protocol
        self.feature_info = [{"name": f"block{b + 1}", "channels": self.embed_dim, "stride": 8} for b in self.blocks]

    def forward(self, x):
        final, maps = self.vit.forward_intermediates(x, indices=self.blocks, intermediates_only=False)
        out = {f["name"]: m for f, m in zip(self.feature_info, maps)}
        out["embed"] = self.vit.forward_head(final)
        return out


def build_backbone():
    return TimmViTBackbone(pretrained=False)
