from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data_pipeline.features import extract_feature_vector
from model import DISTORTION_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

CLASS_TO_IDX = {c: i for i, c in enumerate(DISTORTION_CLASSES)}


class QualityDataset(Dataset):
    def __init__(self, data_root: Path, split: str, augment: bool = False):
        self.root = Path(data_root)
        df = pd.read_csv(self.root / "labels.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = train_transform if augment else eval_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["filename"]
        img_bgr = cv2.imread(str(img_path))
        feats = extract_feature_vector(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)

        label_idx = CLASS_TO_IDX[row["distortion_type"]]
        return {
            "image": img_tensor,
            "features": torch.from_numpy(feats),
            "quality_score": torch.tensor(float(row["quality_score"]), dtype=torch.float32),
            "distortion_label": torch.tensor(label_idx, dtype=torch.long),
        }
