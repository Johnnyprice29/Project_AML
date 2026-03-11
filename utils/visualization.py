"""
utils/visualization.py

Visualization utilities for semantic correspondence results.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, List, Tuple
from PIL import Image


DINO_MEAN = np.array([0.485, 0.456, 0.406])
DINO_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor (C,H,W) → numpy uint8 (H,W,3)."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * DINO_STD + DINO_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def draw_keypoint_matches(
    src_img: torch.Tensor,
    trg_img: torch.Tensor,
    src_kps: torch.Tensor,
    trg_kps_gt: torch.Tensor,
    trg_kps_pred: torch.Tensor,
    alpha: float = 0.1,
    img_size: int = 224,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw source keypoints and their predicted / GT correspondences side-by-side.

    Green lines  → correct predictions (within PCK threshold)
    Red   lines  → incorrect predictions

    Args:
        src_img:      (3, H, W) source image tensor (normalised).
        trg_img:      (3, H, W) target image tensor (normalised).
        src_kps:      (N, 2) source keypoints [x, y].
        trg_kps_gt:   (N, 2) ground-truth target keypoints.
        trg_kps_pred: (N, 2) predicted target keypoints.
        alpha:        PCK threshold (fraction of max(H, W)).
        img_size:     image size used.
        title:        plot title.
        save_path:    if given, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    src_np = denormalize(src_img)
    trg_np = denormalize(trg_img)

    # Side-by-side canvas
    canvas = np.concatenate([src_np, trg_np], axis=1)  # (H, 2W, 3)
    W = img_size

    threshold = alpha * img_size

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(canvas)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    colors = cm.tab10(np.linspace(0, 1, len(src_kps)))

    for i, (s, g, p) in enumerate(zip(src_kps, trg_kps_gt, trg_kps_pred)):
        sx, sy = s.cpu().tolist()
        gx, gy = g.cpu().tolist()
        px, py = p.cpu().tolist()

        dist    = np.linalg.norm([px - gx, py - gy])
        correct = dist <= threshold
        lc      = "lime" if correct else "red"
        c       = colors[i % len(colors)]

        # Source keypoint (left image)
        ax.plot(sx, sy, "o", color=c, markersize=5)
        # GT target keypoint (right image)
        ax.plot(gx + W, gy, "s", color=c, markersize=5, alpha=0.5)
        # Predicted target (right image)
        ax.plot(px + W, py, "^", color=c, markersize=5)
        # Line: src → pred (coloured by correctness)
        ax.plot([sx, px + W], [sy, py], "-", color=lc, linewidth=0.8, alpha=0.7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cost_volume(
    cost_volume: torch.Tensor,
    src_kp_idx: int,
    h: int,
    w: int,
    trg_img: torch.Tensor,
    title: str = "Cost Volume",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise the similarity heatmap in the target image for a chosen
    source feature index.

    Args:
        cost_volume: (Ns, Nt) similarity matrix.
        src_kp_idx:  index into source feature grid.
        h, w:        feature grid dimensions.
        trg_img:     (3, H, W) target image tensor (normalised).
    """
    trg_np  = denormalize(trg_img)
    sim_row = cost_volume[src_kp_idx].reshape(h, w).cpu().numpy()

    # Resize heatmap to image size
    from PIL import Image as PILImage
    heatmap = PILImage.fromarray(
        ((sim_row - sim_row.min()) / (sim_row.max() - sim_row.min() + 1e-8) * 255).astype(np.uint8)
    ).resize((trg_np.shape[1], trg_np.shape[0]), PILImage.BILINEAR)
    heatmap = np.array(heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(trg_np)
    axes[0].set_title("Target Image")
    axes[0].axis("off")
    axes[1].imshow(trg_np)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
