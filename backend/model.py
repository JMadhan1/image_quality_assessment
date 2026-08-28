"""
Hybrid quality model: an ImageNet-pretrained CNN backbone (fine-tuned)
fused with the 8 engineered classical features, jointly predicting:
  - quality_score (regression, 0-100)
  - distortion_type (classification, 7 classes: none + 6 distortion types)

Fine-tuning a pretrained backbone rather than training from scratch is the
right call given the compute/time budget here (a from-scratch CNN would not
converge to competitive accuracy in this timeframe).

Backbone defaults to ConvNeXt-Tiny: 82.1% ImageNet top-1 vs ResNet34's 73.3%
at a comparable parameter count (29M vs 21M), and modern CNN backbones like
ConvNeXt are the better choice over transformer backbones (e.g. Swin) in
this low-data fine-tuning regime specifically -- transformers need more
data to earn their accuracy edge than this ~38k-image synthetic set
provides. ResNet18/34 stay available for comparison/rollback.
"""
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, convnext_tiny,
    ResNet18_Weights, ResNet34_Weights, ConvNeXt_Tiny_Weights,
)

DISTORTION_CLASSES = ["none", "blur", "gaussian_noise", "salt_pepper", "brightness", "jpeg", "block_corrupt"]
N_FEATURES = 8  # must match len(features.FEATURE_NAMES)

_RESNET_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
}


def _build_backbone(backbone_name: str, pretrained: bool):
    if backbone_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        feature_dim = backbone.classifier[2].in_features  # 768
        backbone.classifier[2] = nn.Identity()  # keep LayerNorm2d + Flatten, drop the classification Linear
        return backbone, feature_dim

    ctor, default_weights = _RESNET_BACKBONES[backbone_name]
    weights = default_weights if pretrained else None
    backbone = ctor(weights=weights)
    feature_dim = backbone.fc.in_features  # 512
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class HybridQualityModel(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, n_classes: int = len(DISTORTION_CLASSES),
                 pretrained: bool = True, backbone_name: str = "convnext_tiny"):
        super().__init__()
        self.backbone_name = backbone_name  # plain str, not a submodule -- safe to store on self
        backbone, self.cnn_feature_dim = _build_backbone(backbone_name, pretrained)
        self.backbone = backbone

        self.feat_norm = nn.BatchNorm1d(n_features)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        fusion_dim = self.cnn_feature_dim + 32
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.score_head = nn.Linear(256, 1)          # regression -> sigmoid*100
        self.class_head = nn.Linear(256, n_classes)   # distortion type logits

    def gradcam_target_layer(self) -> nn.Module:
        """Returns (does not store) the right Grad-CAM target layer for
        whichever backbone this instance holds. Deliberately not cached as
        a `self.` attribute -- assigning an nn.Module to a model attribute
        auto-registers it as a second copy of that submodule in the
        state_dict, which broke loading of checkpoints saved before this
        method existed (missing/duplicate key errors)."""
        if self.backbone_name == "convnext_tiny":
            return self.backbone.features[-1][-1]
        return self.backbone.layer4[-1]

    def forward(self, image: torch.Tensor, engineered_feats: torch.Tensor):
        cnn_feat = self.backbone(image)
        f = self.feat_mlp(self.feat_norm(engineered_feats))
        fused = torch.cat([cnn_feat, f], dim=1)
        h = self.trunk(fused)
        score = torch.sigmoid(self.score_head(h)).squeeze(1) * 100.0
        logits = self.class_head(h)
        return score, logits


def build_model(pretrained: bool = True, backbone_name: str = "convnext_tiny") -> HybridQualityModel:
    return HybridQualityModel(pretrained=pretrained, backbone_name=backbone_name)


def save_checkpoint(model: HybridQualityModel, backbone_name: str, path: str):
    """Checkpoints store their backbone name alongside the weights so a
    loader never has to guess/hardcode the architecture that produced them —
    this is what a resnet18 vs resnet34 checkpoint mismatch would otherwise
    silently break at load time."""
    torch.save({"backbone_name": backbone_name, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str, device) -> HybridQualityModel:
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and "backbone_name" in raw:
        backbone_name = raw["backbone_name"]
        state_dict = raw["state_dict"]
    else:
        # legacy checkpoints saved as a bare state_dict, before this
        # self-describing format existed, were all trained with resnet18.
        backbone_name = "resnet18"
        state_dict = raw
    model = build_model(pretrained=False, backbone_name=backbone_name).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
