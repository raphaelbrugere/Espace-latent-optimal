"""ConvNeXt-Tiny backbone: a real feature pyramid (strides 4, 8, 16, 32).

The four feature maps are the outputs of the four ConvNeXt stages. The
embedding is timm's head without classifier: global average pool of the last
stage + LayerNorm, 768-d. No extra linear layer, to stay at ~27.8M parameters.
"""
import timm
import torch.nn as nn

EMBED_DIM = 768


class ConvNeXtBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("convnext_tiny", pretrained=False, num_classes=0, drop_path_rate=0.1)
        self.embed_dim = EMBED_DIM
        self.feature_info = [{"name": f"stage{i + 1}", "channels": c, "stride": s}
                             for i, (c, s) in enumerate(zip((96, 192, 384, 768), (4, 8, 16, 32)))]

    def forward(self, x):
        final, maps = self.model.forward_intermediates(x, indices=4, intermediates_only=False)
        out = {f["name"]: m for f, m in zip(self.feature_info, maps)}
        out["embed"] = self.model.forward_head(final)
        return out


def build_backbone():
    return ConvNeXtBackbone()
