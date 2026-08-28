"""
Generalization check (assessment Section 9: "evaluation should use unseen
images and provide evidence of generalization").

The train/val/test splits already hold out unseen *source images* from the
same Oxford-IIIT Pet pool. This script goes further: it scores clean,
UNDEGRADED images from a completely different dataset (Oxford Flowers102 --
different content domain, never seen during training in any form) and
checks that the model still rates them as high-quality / ACCEPTABLE, i.e.
that it generalizes beyond the training distribution's specific image
content rather than having merely memorized Pet-photo statistics.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torchvision.datasets import Flowers102

from inference import analyze_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../data")
    ap.add_argument("--checkpoint", type=str, default="../models/hybrid_model.pt")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--out", type=str, default="../models/generalization_report.json")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    print(f"Downloading/loading Oxford Flowers102 (unseen domain) into {root} ...")
    ds = Flowers102(root=str(root), split="test", download=True)

    random.seed(123)
    idxs = random.sample(range(len(ds)), min(args.n_images, len(ds)))

    scores, labels = [], []
    for i in idxs:
        img, _ = ds[i]
        img = img.convert("RGB")
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        result = analyze_image(buf.getvalue(), args.checkpoint)
        scores.append(result["quality_score"])
        labels.append(result["quality_label"])

    n = len(scores)
    mean_score = sum(scores) / n
    label_counts = {l: labels.count(l) for l in set(labels)}

    report = {
        "dataset": "Oxford Flowers102 (test split, undegraded, unseen domain)",
        "n_images": n,
        "mean_quality_score": round(mean_score, 2),
        "label_distribution": label_counts,
        "pct_acceptable_or_degraded": round(
            100 * (label_counts.get("ACCEPTABLE", 0) + label_counts.get("DEGRADED", 0)) / n, 1
        ),
    }
    print(json.dumps(report, indent=2))

    out_path = Path(args.out).resolve()
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
