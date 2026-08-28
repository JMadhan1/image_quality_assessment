"""
Single-image inference: runs the hybrid model, produces a quality score,
issue list (from both the CNN's distortion classification and the engineered
feature thresholds), and a Grad-CAM heatmap (base64 PNG) explaining which
region of the image drove the CNN's prediction.
"""
import base64
import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from data_pipeline.features import extract_features, extract_feature_vector, interpret_features
from dataset import eval_transform
from model import DISTORTION_CLASSES, load_checkpoint
from quality_decision import build_issues, quality_label

_MODEL = None
_DEVICE = None
_CAM = None


def load_model(checkpoint_path: str):
    global _MODEL, _DEVICE, _CAM
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint_path, _DEVICE)
    _MODEL = model
    _CAM = GradCAM(model=_ScoreWrapper(model), target_layers=[model.gradcam_target_layer()])
    return model


class _ScoreWrapper(torch.nn.Module):
    """Grad-CAM needs a single-tensor-output forward; wrap the class logits
    since defect localization is most meaningful for the classification head."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._feats = None

    def set_features(self, feats):
        self._feats = feats

    def forward(self, image):
        _, logits = self.model(image, self._feats)
        return logits


def _heatmap_to_base64(rgb_float: np.ndarray, grayscale_cam: np.ndarray) -> str:
    vis = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    pil_img = Image.fromarray(vis)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_image(image_bytes: bytes, checkpoint_path: str) -> dict:
    if _MODEL is None:
        load_model(checkpoint_path)

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    feats_dict = extract_features(img_bgr)
    engineered_issues = interpret_features(feats_dict)
    feat_vec = extract_feature_vector(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = eval_transform(img_rgb).unsqueeze(0).to(_DEVICE)
    feat_tensor = torch.from_numpy(feat_vec).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        pred_score, logits = _MODEL(img_tensor, feat_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_class = DISTORTION_CLASSES[pred_idx]

    _CAM.model.set_features(feat_tensor)
    grayscale_cam = _CAM(input_tensor=img_tensor, targets=None)[0]
    resized_rgb = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
    heatmap_b64 = _heatmap_to_base64(resized_rgb, grayscale_cam)

    score = round(float(pred_score.item()), 1)
    class_probs = {c: float(p) for c, p in zip(DISTORTION_CLASSES, probs)}
    issues = build_issues(feats_dict, engineered_issues, pred_class, class_probs, score)
    label = quality_label(score, issues)

    return {
        "quality_score": score,
        "quality_label": label,
        "issues": issues,
        "predicted_distortion": pred_class,
        "distortion_confidence": round(float(probs[pred_idx]), 3),
        "class_probabilities": {c: round(float(p), 3) for c, p in zip(DISTORTION_CLASSES, probs)},
        "engineered_features": {k: round(float(v), 3) for k, v in feats_dict.items()},
        "gradcam_heatmap_base64": heatmap_b64,
    }
