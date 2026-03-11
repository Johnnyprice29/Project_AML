"""
utils/metrics.py

Evaluation metrics for semantic correspondence.
"""

import torch
import numpy as np
from typing import Union


def pck(
    pred_kps: torch.Tensor,
    gt_kps: torch.Tensor,
    img_size: Union[int, tuple] = 224,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Percentage of Correct Keypoints (PCK) @ alpha.

    A predicted keypoint is "correct" if its Euclidean distance to the
    ground-truth is ≤ α × max(H, W).

    Args:
        pred_kps:  (B, N, 2)   predicted [x, y] pixel coordinates.
        gt_kps:    (B, N, 2)   ground-truth [x, y] pixel coordinates.
        img_size:  int or (H, W) — reference image size.
        alpha:     threshold radius as a fraction of max(H, W).

    Returns:
        pck_score: scalar tensor, mean over all valid keypoints in the batch.
    """
    if isinstance(img_size, int):
        H = W = img_size
    else:
        H, W = img_size

    threshold = alpha * max(H, W)

    # Euclidean distance between predicted and GT keypoints
    dist = torch.norm(pred_kps - gt_kps, dim=-1)  # (B, N)

    correct = (dist <= threshold).float()          # (B, N)
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
