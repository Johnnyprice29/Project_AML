"""
dataloaders/pfpascal.py

PyTorch Dataset for PF-Pascal semantic correspondence.
Output format is identical to SPairDataset so the same collate_fn
and evaluation loop work without changes.

Structure expected after download:
    root/
        Annotations/
            <category>/
                <image_id>.mat        (keys: 'kps', 'bbox', 'imsize', …)
        JPEGImages/
            <image_id>.jpg            (flat – NO category subfolders)
"""

import os
import glob
from typing import Optional, Callable, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import scipy.io as sio

# Same normalisation as SPair / DINOv2
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD  = (0.229, 0.224, 0.225)


def get_default_transform(img_size: int = 224) -> Callable:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])


class PFPascalDataset(Dataset):
    """
    PF-Pascal dataset for semantic correspondence evaluation.

    Returns dicts with the **same keys** as SPairDataset so that
    ``collate_spair`` can be reused unchanged:
        src_img   : (3, H, W) float tensor
        trg_img   : (3, H, W) float tensor
        src_kps   : (N, 2) float tensor   – scaled to img_size
        trg_kps   : (N, 2) float tensor   – scaled to img_size
        category  : str
        pair_id   : str
    """

    def __init__(self, root: str, img_size: int = 224,
                 transform: Optional[Callable] = None):
        self.root = root
        self.img_size = img_size
        self.transform = transform or get_default_transform(img_size)
        self.pairs: List[tuple] = []

        # --- Locate annotation directory (case-insensitive) ---
        anno_dir = os.path.join(root, "Annotations")
        if not os.path.exists(anno_dir):
            anno_dir = os.path.join(root, "annotations")

        if not os.path.exists(anno_dir):
            print(f"[WARNING] Annotation directory not found in {root}")
            return

        anno_files = glob.glob(os.path.join(anno_dir, "*", "*.mat"))

        # --- Group annotations by category ---
        categories: dict[str, list] = {}
        for f in anno_files:
            cat = os.path.basename(os.path.dirname(f))
            categories.setdefault(cat, []).append(f)

        # --- Build pairs within each category (cap at 10 neighbours) ---
        for cat, files in categories.items():
            files.sort()
            for i in range(len(files)):
                for j in range(i + 1, min(i + 11, len(files))):
                    self.pairs.append((files[i], files[j], cat))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        anno1_path, anno2_path, category = self.pairs[idx]

        # --- Load keypoints from .mat ---
        kps1 = self._load_kps(anno1_path)
        kps2 = self._load_kps(anno2_path)

        # Only keep keypoints that are visible in BOTH images
        # A keypoint with coords (0,0) or (-1,-1) is typically invisible
        n = min(len(kps1), len(kps2))
        kps1, kps2 = kps1[:n], kps2[:n]

        # --- Load images ---
        img1_name = os.path.basename(anno1_path).replace(".mat", ".jpg")
        img2_name = os.path.basename(anno2_path).replace(".mat", ".jpg")
        
        p1 = os.path.join(self.root, "JPEGImages", img1_name)
        if not os.path.exists(p1): p1 = os.path.join(self.root, "JPEGImages", category, img1_name)
            
        p2 = os.path.join(self.root, "JPEGImages", img2_name)
        if not os.path.exists(p2): p2 = os.path.join(self.root, "JPEGImages", category, img2_name)
            
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")

        # --- Scale keypoints to resized image space ---
        orig1 = np.array(img1.size, dtype=np.float32)  # (W, H)
        orig2 = np.array(img2.size, dtype=np.float32)
        scale1 = np.array([self.img_size, self.img_size], dtype=np.float32) / orig1
        scale2 = np.array([self.img_size, self.img_size], dtype=np.float32) / orig2
        kps1 = kps1 * scale1
        kps2 = kps2 * scale2

        # --- Transform images to tensors ---
        src_tensor = self.transform(img1)
        trg_tensor = self.transform(img2)

        pair_id = (os.path.splitext(os.path.basename(anno1_path))[0] + "__"
                   + os.path.splitext(os.path.basename(anno2_path))[0])

        return {
            "src_img":  src_tensor,                          # (3, H, W)
            "trg_img":  trg_tensor,                          # (3, H, W)
            "src_kps":  torch.from_numpy(kps1).float(),      # (N, 2)
            "trg_kps":  torch.from_numpy(kps2).float(),      # (N, 2)
            "category": category,
            "pair_id":  pair_id,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _load_kps(mat_path: str) -> np.ndarray:
        """Load keypoints from a PF-Pascal .mat file.  Returns (N, 2)."""
        mat = sio.loadmat(mat_path)
        kps = mat["kps"] if "kps" in mat else mat["keypoints"]
        kps = np.array(kps, dtype=np.float32)
        if kps.ndim == 1:
            kps = kps.reshape(-1, 2)
        return kps
