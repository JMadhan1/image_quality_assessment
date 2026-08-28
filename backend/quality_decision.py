"""
Turns raw model outputs (CNN distortion class + engineered features) into
the assessment's required decision shape: an overall ACCEPTABLE / DEGRADED /
DEFECTIVE label, and a structured issues list of
{type, severity, confidence}, using a canonical issue taxonomy that maps
directly onto the assessment's required detection capabilities:
blur, underexposure, overexposure, noise, corruption, defect.

Two signal sources feed each canonical issue type:
  - engineered features (cheap, rule-based, always computed)
  - the CNN's learned distortion classification

The CNN's "brightness" class is direction-agnostic (it only says exposure
is off, not which way) -- the engineered brightness_mean feature is used to
disambiguate underexposure vs overexposure. This is why the hybrid
combination is more informative than either signal alone.
"""
import numpy as np

# CNN distortion class -> canonical issue type(s). "brightness" is resolved
# separately since it needs the engineered feature to pick a direction.
CNN_TO_CANONICAL = {
    "blur": ["blur"],
    "gaussian_noise": ["noise"],
    "salt_pepper": ["noise"],
    "jpeg": ["corruption"],
    "block_corrupt": ["corruption", "defect"],
}

# engineered interpret_features() label -> canonical issue type
ENGINEERED_TO_CANONICAL = {
    "blur": "blur",
    "underexposed": "underexposure",
    "overexposed": "overexposure",
    "noisy": "noise",
    "low_contrast": "low_contrast",  # bonus signal, not one of the 6 required types
}


def _severity_bucket(score: float) -> str:
    if score >= 75:
        return "low"
    if score >= 40:
        return "medium"
    return "high"


def _engineered_confidence(feats: dict, canonical_type: str) -> float:
    """Heuristic confidence: how far past its trigger threshold the raw
    feature value is, normalized to [0, 1]. Thresholds match features.interpret_features."""
    if canonical_type == "blur":
        return float(np.clip((50 - feats["laplacian_var"]) / 50, 0, 1))
    if canonical_type == "underexposure":
        return float(np.clip((60 - feats["brightness_mean"]) / 60, 0, 1))
    if canonical_type == "overexposure":
        return float(np.clip((feats["brightness_mean"] - 200) / 55, 0, 1))
    if canonical_type == "noise":
        return float(np.clip((feats["noise_energy"] - 8) / 12, 0, 1))
    if canonical_type == "low_contrast":
        return float(np.clip((20 - feats["contrast"]) / 20, 0, 1))
    return 0.5


def build_issues(feats: dict, engineered_flags: list[str], pred_class: str,
                  class_probs: dict, overall_score: float) -> list[dict]:
    issues: dict[str, dict] = {}

    for flag in engineered_flags:
        canonical = ENGINEERED_TO_CANONICAL.get(flag)
        if canonical is None:
            continue
        conf = _engineered_confidence(feats, canonical)
        issues[canonical] = {
            "type": canonical,
            "severity": _severity_bucket(overall_score),
            "confidence": round(conf, 3),
            "source": "engineered_features",
        }

    if pred_class != "none":
        cnn_conf = class_probs[pred_class]
        if pred_class == "brightness":
            canonical_types = ["underexposure" if feats["brightness_mean"] < 127 else "overexposure"]
        else:
            canonical_types = CNN_TO_CANONICAL.get(pred_class, [])

        for canonical in canonical_types:
            existing = issues.get(canonical)
            if existing is None or cnn_conf > existing["confidence"]:
                issues[canonical] = {
                    "type": canonical,
                    "severity": _severity_bucket(overall_score),
                    "confidence": round(float(cnn_conf), 3),
                    "source": "cnn" if existing is None else "cnn+engineered_features",
                }
            else:
                existing["source"] = "cnn+engineered_features"

    return sorted(issues.values(), key=lambda d: -d["confidence"])


def quality_label(score: float, issues: list[dict]) -> str:
    issue_types = {i["type"] for i in issues}
    if "defect" in issue_types:
        return "DEFECTIVE"
    if score >= 75:
        return "ACCEPTABLE"
    if score >= 40:
        return "DEGRADED"
    return "DEFECTIVE"
