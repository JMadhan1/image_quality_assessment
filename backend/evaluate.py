"""
Evaluation on the held-out synthetic test split, plus (if present) a
KonIQ-10k real-world slice for the generalization check.

Produces: per-class precision/recall/F1, confusion matrix, quality-score
MAE/RMSE, and a dump of the worst failure cases for the report.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from dataset import QualityDataset
from model import DISTORTION_CLASSES, load_checkpoint


@torch.no_grad()
def evaluate_split(model, loader, device):
    all_true_cls, all_pred_cls = [], []
    all_true_score, all_pred_score = [], []
    worst = []  # (abs_error, filename-index, true_label, pred_label)

    idx = 0
    for batch in loader:
        images = batch["image"].to(device)
        feats = batch["features"].to(device)
        scores = batch["quality_score"].to(device)
        labels = batch["distortion_label"].to(device)

        pred_score, logits = model(images, feats)
        pred_cls = logits.argmax(1)

        all_true_cls.extend(labels.cpu().numpy().tolist())
        all_pred_cls.extend(pred_cls.cpu().numpy().tolist())
        all_true_score.extend(scores.cpu().numpy().tolist())
        all_pred_score.extend(pred_score.cpu().numpy().tolist())

        errs = (pred_score - scores).abs().cpu().numpy()
        for e, t, p in zip(errs, labels.cpu().numpy(), pred_cls.cpu().numpy()):
            worst.append((float(e), idx, int(t), int(p)))
            idx += 1

    report = classification_report(
        all_true_cls, all_pred_cls, target_names=DISTORTION_CLASSES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(all_true_cls, all_pred_cls, labels=list(range(len(DISTORTION_CLASSES))))
    mae = mean_absolute_error(all_true_score, all_pred_score)
    rmse = float(np.sqrt(mean_squared_error(all_true_score, all_pred_score)))

    worst.sort(key=lambda x: -x[0])
    worst_cases = [
        {"abs_error": w[0], "true_class": DISTORTION_CLASSES[w[2]], "pred_class": DISTORTION_CLASSES[w[3]]}
        for w in worst[:20]
    ]

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": DISTORTION_CLASSES,
        "quality_score_mae": mae,
        "quality_score_rmse": rmse,
        "worst_cases": worst_cases,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../data")
    ap.add_argument("--checkpoint", type=str, default="../models/hybrid_model.pt")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", type=str, default="../models/eval_report.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)

    ds = QualityDataset(Path(args.data_root).resolve(), args.split, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = evaluate_split(model, loader, device)
    print(json.dumps({k: v for k, v in results.items() if k != "confusion_matrix"}, indent=2))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()
