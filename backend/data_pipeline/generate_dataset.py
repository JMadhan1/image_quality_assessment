"""
Builds a labeled image-quality-degradation dataset from a clean source set.

Source: Oxford-IIIT Pet (auto-downloaded via torchvision, ~7,349 real photos,
37 categories, hosted on a stable Oxford server). Caltech256's Google Drive
mirror is currently dead (404 as of 2026-08), so this is used instead —
still diverse, real-resolution content, unlike small/low-res sets like
STL10 or CIFAR, so blur/noise/JPEG artifacts look like real degradations.

For each sampled clean image we produce several degraded variants across
5 controlled distortion types x 5 severity levels, with ground-truth labels
(distortion_type, severity, quality_score). Splits are done by source image
so no two variants of the same photo leak across train/val/test.
"""
import argparse
import io
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import OxfordIIITPet

RNG_SEED = 42
MAX_DIM = 384  # resize longest side to this, keeps training fast on 6GB VRAM

DISTORTION_TYPES = ["blur", "gaussian_noise", "salt_pepper", "brightness", "jpeg", "block_corrupt"]
SEVERITIES = [1, 2, 3, 4, 5]  # 1 = mild, 5 = severe


def load_clean_pool(root: Path, n_images: int) -> list[Path]:
    print(f"Downloading/loading Oxford-IIIT Pet into {root} (first run only)...")
    ds = OxfordIIITPet(root=str(root), download=True)
    idxs = list(range(len(ds)))
    random.Random(RNG_SEED).shuffle(idxs)
    idxs = idxs[:n_images]

    out_dir = root / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in idxs:
        img, _ = ds[i]
        img = img.convert("RGB")
        w, h = img.size
        scale = MAX_DIM / max(w, h)
        if scale < 1:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        p = out_dir / f"clean_{i:06d}.jpg"
        img.save(p, quality=95)
        paths.append(p)
    print(f"Prepared {len(paths)} clean source images at {out_dir}")
    return paths


def apply_blur(img: np.ndarray, severity: int) -> np.ndarray:
    k = [3, 5, 9, 13, 19][severity - 1]
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = [5, 12, 22, 35, 50][severity - 1]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_salt_pepper(img: np.ndarray, severity: int) -> np.ndarray:
    amount = [0.005, 0.01, 0.03, 0.06, 0.1][severity - 1]
    out = img.copy()
    mask = np.random.random(img.shape[:2])
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return out


def apply_brightness(img: np.ndarray, severity: int) -> np.ndarray:
    # alternate under/over exposure by severity parity, magnitude grows with severity
    delta = [30, 55, 80, 110, 140][severity - 1]
    sign = -1 if severity % 2 == 0 else 1
    return np.clip(img.astype(np.int16) + sign * delta, 0, 255).astype(np.uint8)


def apply_jpeg(img: np.ndarray, severity: int) -> np.ndarray:
    quality = [40, 25, 15, 8, 3][severity - 1]
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def apply_block_corrupt(img: np.ndarray, severity: int) -> np.ndarray:
    n_blocks = [1, 2, 4, 7, 12][severity - 1]
    out = img.copy()
    h, w = img.shape[:2]
    bs = max(4, min(h, w) // 10)
    for _ in range(n_blocks):
        y, x = random.randint(0, max(0, h - bs)), random.randint(0, max(0, w - bs))
        out[y:y + bs, x:x + bs] = np.random.randint(0, 255, (min(bs, h - y), min(bs, w - x), 3), dtype=np.uint8)
    return out


DEGRADE_FN = {
    "blur": apply_blur,
    "gaussian_noise": apply_gaussian_noise,
    "salt_pepper": apply_salt_pepper,
    "brightness": apply_brightness,
    "jpeg": apply_jpeg,
    "block_corrupt": apply_block_corrupt,
}


def quality_score(severity: int) -> float:
    # simple linear mapping severity 1..5 -> score 100..20; clean images = 100
    return round(100 - (severity - 1) * 20, 1)


def build(out_root: Path, n_clean: int, variants_per_type: int):
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    clean_paths = load_clean_pool(out_root, n_clean)
    degraded_dir = out_root / "degraded"
    degraded_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in clean_paths:
        rows.append({
            "filename": str(p.relative_to(out_root)),
            "source_image": p.stem,
            "distortion_type": "none",
            "severity": 0,
            "quality_score": 100.0,
        })

        img = cv2.imread(str(p))
        for dtype in DISTORTION_TYPES:
            severities = random.sample(SEVERITIES, k=min(variants_per_type, len(SEVERITIES)))
            for sev in severities:
                out = DEGRADE_FN[dtype](img, sev)
                fname = f"{p.stem}__{dtype}_s{sev}.jpg"
                out_path = degraded_dir / fname
                cv2.imwrite(str(out_path), out)
                rows.append({
                    "filename": str(out_path.relative_to(out_root)),
                    "source_image": p.stem,
                    "distortion_type": dtype,
                    "severity": sev,
                    "quality_score": quality_score(sev),
                })

    df = pd.DataFrame(rows)

    # split by source_image so variants of one photo never cross splits
    sources = df["source_image"].unique().tolist()
    random.Random(RNG_SEED).shuffle(sources)
    n = len(sources)
    train_src = set(sources[: int(0.7 * n)])
    val_src = set(sources[int(0.7 * n): int(0.85 * n)])

    def split_of(src):
        if src in train_src:
            return "train"
        if src in val_src:
            return "val"
        return "test"

    df["split"] = df["source_image"].apply(split_of)
    labels_path = out_root / "labels.csv"
    df.to_csv(labels_path, index=False)

    print(f"\nDone. {len(df)} labeled rows -> {labels_path}")
    print(df.groupby(["split", "distortion_type"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="../../data")
    ap.add_argument("--n_clean", type=int, default=1500, help="number of clean source images to sample")
    ap.add_argument("--variants_per_type", type=int, default=3, help="how many of the 5 severities to generate per distortion type")
    args = ap.parse_args()

    build(Path(args.out).resolve(), args.n_clean, args.variants_per_type)
