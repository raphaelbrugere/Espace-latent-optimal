"""Baseline backbone: ViT-S/8 from the LeJEPA minimal example.

The embedding is timm's head output (pool + linear to 512), exactly as in the
original training script. The four "feature maps" are the outputs of blocks
3, 6, 9 and 12, all at stride 8: a plain ViT has no pyramid. That is a known
weakness for dense tasks and a good first thing to improve (see ViTDet's
"simple feature pyramid", or try a ConvNet).
"""
import timm
import torch.nn as nn

EMBED_DIM, PRETRAIN_SIZE = 512, 128


class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # dynamic_img_size: pretrain at 128px, run segmentation at 224px.
        self.vit = timm.create_model("vit_small_patch8_224", img_size=PRETRAIN_SIZE, num_classes=EMBED_DIM,
                                     drop_path_rate=0.1, dynamic_img_size=True)
        self.blocks = [2, 5, 8, 11]
        self.embed_dim = EMBED_DIM
        self.feature_info = [{"name": f"block{b + 1}", "channels": 384, "stride": 8} for b in self.blocks]

    def forward(self, x):
        final, maps = self.vit.forward_intermediates(x, indices=self.blocks, intermediates_only=False)
        out = {f["name"]: m for f, m in zip(self.feature_info, maps)}
        out["embed"] = self.vit.forward_head(final)
        return out


def build_backbone():
    return ViTBackbone()
