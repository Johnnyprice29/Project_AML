"""
data/dataset.py

PyTorch Dataset for SPair-71k semantic correspondence.

Each item returns:
  - src_img:       (3, H, W) float tensor
  - trg_img:       (3, H, W) float tensor
  - src_kps:       (N, 2) keypoint coordinates in source image  [x, y]
  - trg_kps:       (N, 2) keypoint coordinates in target image  [x, y]
  - src_bbox:      (4,) bounding box [x1, y1, x2, y2]
  - trg_bbox:      (4,) bounding box [x1, y1, x2, y2]
  - category:      string category name
"""

import os
import json
import glob
from typing import Optional, Callable, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# Standard ImageNet normalisation used by DINOv2
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD  = (0.229, 0.224, 0.225)


def get_default_transform(img_size: int = 224) -> Callable:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])


class SPairDataset(Dataset):
    """
    SPair-71k dataset loader.

    Directory structure expected:
        root/
          ImageAnnotation/
            <category>/
              <pair_id>.json    # one JSON per image pair
          JPEGImages/
            <category>/
              <image_id>.jpg
    """

    SPLITS = ("trn", "val", "test")

    def __init__(
        self,
        root: str,
        split: str = "trn",
        img_size: int = 224,
        transform: Optional[Callable] = None,
        categories: Optional[List[str]] = None,
    ):
        assert split in self.SPLITS, f"split must be one of {self.SPLITS}"
        self.root = root
        self.split = split
        self.img_size = img_size
        self.transform = transform or get_default_transform(img_size)

        # Collect annotation files
        ann_dir = os.path.join(root, "ImageAnnotation")
        if not os.path.isdir(ann_dir):
            raise FileNotFoundError(f"[ERROR] Cartella annotazioni non trovata in {ann_dir}. "
                                    f"Verifica il percorso --dataset_root.")
            
        all_jsons = sorted(glob.glob(os.path.join(ann_dir, "**", "*.json"), recursive=True))

        # Filter by split file
        split_file = os.path.join(root, "Layout", "large", f"{split}.txt")
        if not os.path.isfile(split_file):
             raise FileNotFoundError(f"[ERROR] File di split non trovato in {split_file}. "
                                     f"Assicurati che SPair-71k sia estratto correttamente.")

        with open(split_file, "r") as f:
            valid_ids = set(line.strip() for line in f if line.strip())

        self.samples = []
        for jpath in all_jsons:
            pair_id = os.path.splitext(os.path.basename(jpath))[0]
            if pair_id not in valid_ids:
                continue
            cat = os.path.basename(os.path.dirname(jpath))
            if categories and cat not in categories:
                continue
            self.samples.append(jpath)
            
        if len(self.samples) == 0:
            raise ValueError(f"[ERROR] Nessun campione trovato per lo split '{split}' in {root}. "
                             f"Controlla i file JSON in ImageAnnotation.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ann_path = self.samples[idx]
        with open(ann_path, "r") as f:
            ann = json.load(f)

        category = ann["category"]
        src_id = ann["src_imname"].replace(".jpg", "")
        trg_id = ann["trg_imname"].replace(".jpg", "")

        # Load images
        src_img = self._load_image(category, ann["src_imname"])
        trg_img = self._load_image(category, ann["trg_imname"])

        # Original image size (before resize) for keypoint scaling
        orig_src_size = np.array(src_img.size, dtype=np.float32)  # (W, H)
        orig_trg_size = np.array(trg_img.size, dtype=np.float32)

        # Keypoints: list of [x, y] in original image coords
        src_kps = np.array(ann["src_kps"], dtype=np.float32)  # (N, 2)
        trg_kps = np.array(ann["trg_kps"], dtype=np.float32)

        # Scale keypoints to resized image space
        scale_src = np.array([self.img_size, self.img_size], dtype=np.float32) / orig_src_size
        scale_trg = np.array([self.img_size, self.img_size], dtype=np.float32) / orig_trg_size
        src_kps = src_kps * scale_src
        trg_kps = trg_kps * scale_trg

        # Apply image transform
        src_tensor = self.transform(src_img)
        trg_tensor = self.transform(trg_img)

        return {
            "src_img":  src_tensor,                              # (3, H, W)
            "trg_img":  trg_tensor,                              # (3, H, W)
            "src_kps":  torch.from_numpy(src_kps),               # (N, 2)
            "trg_kps":  torch.from_numpy(trg_kps),               # (N, 2)
            "category": category,
            "pair_id":  os.path.splitext(os.path.basename(ann_path))[0],
        }

    def _load_image(self, category: str, imname: str) -> Image.Image:
        img_path = os.path.join(self.root, "JPEGImages", category, imname)
        return Image.open(img_path).convert("RGB")
