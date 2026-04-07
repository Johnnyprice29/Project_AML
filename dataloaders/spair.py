"""
dataloaders/spair.py

PyTorch Dataset for SPair-71k semantic correspondence.
Compatible with multiple directory structures (official and local mirrors).
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
    SPair-71k dataset loader. Handles multiple folder layouts.
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

        # 1. Read split file (Layout/large/trn.txt)
        split_file = os.path.join(root, "Layout", "large", f"{split}.txt")
        if not os.path.isfile(split_file):
             # Try root split file if Layout/large is missing
             split_file = os.path.join(root, "Layout", f"{split}.txt")
             
        if not os.path.isfile(split_file):
             raise FileNotFoundError(f"[ERROR] Split file not found in {root}/Layout/...")

        with open(split_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        # 2. Collect files based on split lines
        self.samples = []
        
        # Possible annotation base directories
        ann_bases = [
            os.path.join(root, "PairAnnotation", split),  # Local mirror style (seen on C:)
            os.path.join(root, "PairAnnotation"),         
            os.path.join(root, "ImageAnnotation"),        # Official style
        ]
        
        print(f"[INFO] Initializing {split} split. Scanning annotations...")
        
        for line in lines:
            # line: "pair_id:category"
            if ":" not in line: continue
            pair_id, cat = line.split(":")
            
            if categories and cat not in categories:
                continue
                
            # Try possible filenames
            # a) "pair_id:cat.json" (Linux / Official style)
            # b) "pair_id_cat.json" (Windows mirror style)
            # c) "pair_id.json"      (Other mirrors)
            found = False
            for base in ann_bases:
                if not os.path.isdir(base): continue
                
                # Check directly or in category subfolder
                for sub in ["", cat]:
                    p1 = os.path.join(base, sub, f"{pair_id}:{cat}.json")
                    p1b = os.path.join(base, sub, f"{pair_id}_{cat}.json")
                    p2 = os.path.join(base, sub, f"{pair_id}.json")
                    
                    if os.path.isfile(p1):
                        self.samples.append(p1); found = True; break
                    if os.path.isfile(p1b):
                        self.samples.append(p1b); found = True; break
                    if os.path.isfile(p2):
                        self.samples.append(p2); found = True; break
                if found: break
                
        if len(self.samples) == 0:
            raise ValueError(f"[ERROR] No samples found for split '{split}' in {root}. "
                             f"Checked bases: {ann_bases}. "
                             f"Example missing: {lines[0]}")

        print(f"[INFO] Loaded {len(self.samples)} samples for {split} split.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ann_path = self.samples[idx]
        with open(ann_path, "r") as f:
            ann = json.load(f)

        category = ann.get("category", os.path.basename(os.path.dirname(ann_path)))
        
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
        # Check in JPEGImages/<cat>/<imname>
        img_path = os.path.join(self.root, "JPEGImages", category, imname)
        if not os.path.exists(img_path):
            # Try global JPEGImages folder
            img_path = os.path.join(self.root, "JPEGImages", imname)
        return Image.open(img_path).convert("RGB")


def collate_spair(batch: List[dict]) -> dict:
    """
    Custom collate function to handle variable number of keypoints per image.
    Pads keypoints with -1.0 and returns a boolean mask.
    """
    src_imgs = torch.stack([item["src_img"] for item in batch])
    trg_imgs = torch.stack([item["trg_img"] for item in batch])
    categories = [item["category"] for item in batch]
    pair_ids = [item["pair_id"] for item in batch]

    # Find max number of keypoints in this batch
    max_kps = max([item["src_kps"].shape[0] for item in batch])
    B = len(batch)

    # Prepare padded tensors
    padded_src_kps = torch.full((B, max_kps, 2), -1.0)
    padded_trg_kps = torch.full((B, max_kps, 2), -1.0)
    kps_mask = torch.zeros((B, max_kps), dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["src_kps"].shape[0]
        padded_src_kps[i, :n] = item["src_kps"]
        padded_trg_kps[i, :n] = item["trg_kps"]
        kps_mask[i, :n] = True

    return {
        "src_img": src_imgs,
        "trg_img": trg_imgs,
        "src_kps": padded_src_kps,
        "trg_kps": padded_trg_kps,
        "kps_mask": kps_mask,
        "category": categories,
        "pair_id": pair_ids,
    }
