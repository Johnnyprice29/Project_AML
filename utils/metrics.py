"""
utils/metrics.py

Evaluation metrics for semantic correspondence.
"""

import torch
import numpy as np
from typing import Union, Optional, List, Dict


def pck(
    pred_kps: torch.Tensor,
    gt_kps: torch.Tensor,
    img_size: Union[int, tuple] = 224,
    alpha: float = 0.1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Percentage of Correct Keypoints (PCK) @ alpha.
    """
    if isinstance(img_size, int):
        H = W = img_size
    else:
        H, W = img_size

    # Euclidean distance
    dist = torch.norm(pred_kps - gt_kps, dim=-1)  # (B, N)
    threshold = alpha * float(max(H, W))
    correct = (dist <= threshold).float()

    if mask is not None:
        # Only compute mean over valid keypoints
        return correct[mask].mean()
    
    return correct.mean()


def pck_per_category(
    pred_kps: list,
    gt_kps: list,
    categories: list,
    img_size: Union[int, tuple] = 224,
    alpha: float = 0.1,
) -> dict:
    """
    Compute per-category PCK.

    Args:
        pred_kps:   list of (N_i, 2) tensors, one per sample.
        gt_kps:     list of (N_i, 2) tensors, one per sample.
        categories: list of category strings, one per sample.
        img_size:   reference size.
        alpha:      threshold.

    Returns:
        dict mapping category → mean PCK score (float).
    """
    if isinstance(img_size, int):
        H = W = img_size
    else:
        H, W = img_size

    threshold = alpha * max(H, W)
    results: dict = {}

    for pred, gt, cat in zip(pred_kps, gt_kps, categories):
        dist    = torch.norm(pred - gt, dim=-1)    # (N,)
        correct = (dist <= threshold).float().mean().item()
        if cat not in results:
            results[cat] = []
        results[cat].append(correct)

    return {cat: float(np.mean(scores)) for cat, scores in results.items()}
