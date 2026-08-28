"""
Classical, interpretable image-quality features. Cheap to compute, and used
both as (a) auxiliary input fused into the CNN head and (b) directly reported
to the user as human-readable evidence alongside the CNN's prediction.
"""
import cv2
import numpy as np
import pywt

FEATURE_NAMES = [
    "laplacian_var",       # blur: sharp edges -> high variance
    "brightness_mean",     # exposure
    "brightness_std",
    "contrast",             # RMS contrast
    "saturation_mean",
    "noise_energy",         # high-frequency wavelet energy -> noise
    "edge_density",
    "colorfulness",
]


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness_stats(gray: np.ndarray) -> tuple[float, float]:
    return float(gray.mean()), float(gray.std())


def _contrast(gray: np.ndarray) -> float:
    return float(gray.astype(np.float64).std())


def _saturation_mean(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def _noise_energy(gray: np.ndarray) -> float:
    # HH sub-band of a single-level wavelet decomposition isolates
    # high-frequency content most associated with sensor/compression noise
    coeffs = pywt.dwt2(gray.astype(np.float32), "db1")
    _, (_, _, hh) = coeffs
    return float(np.mean(np.abs(hh)))


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    return float(np.mean(edges > 0))


def _colorfulness(img_bgr: np.ndarray) -> float:
    # Hasler & Susstrunk (2003) colorfulness metric
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def extract_features(img_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b_mean, b_std = _brightness_stats(gray)
    return {
        "laplacian_var": _laplacian_var(gray),
        "brightness_mean": b_mean,
        "brightness_std": b_std,
        "contrast": _contrast(gray),
        "saturation_mean": _saturation_mean(img_bgr),
        "noise_energy": _noise_energy(gray),
        "edge_density": _edge_density(gray),
        "colorfulness": _colorfulness(img_bgr),
    }


def extract_feature_vector(img_bgr: np.ndarray) -> np.ndarray:
    feats = extract_features(img_bgr)
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)


def interpret_features(feats: dict) -> list[str]:
    """Human-readable flags derived from engineered features (rule-of-thumb
    thresholds tuned on the synthetic dataset's clean-image distribution)."""
    issues = []
    if feats["laplacian_var"] < 50:
        issues.append("blur")
    if feats["brightness_mean"] < 60:
        issues.append("underexposed")
    elif feats["brightness_mean"] > 200:
        issues.append("overexposed")
    if feats["noise_energy"] > 8:
        issues.append("noisy")
    if feats["contrast"] < 20:
        issues.append("low_contrast")
    return issues
