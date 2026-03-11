"""
utils/matching.py

Nearest-neighbour feature matching utilities.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def cosine_similarity_cost_volume(
    src_feats: torch.Tensor,
    trg_feats: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Compute the cosine similarity cost volume between two dense feature maps.

    Args:
        src_feats: (B, Ns, C) L2-normalised source features.
        trg_feats: (B, Nt, C) L2-normalised target features.
        temperature: softmax temperature.

    Returns:
        cost_volume: (B, Ns, Nt) similarity scores.
    """
    # dot product of unit vectors = cosine similarity
    cost_volume = torch.bmm(src_feats, trg_feats.transpose(1, 2))  # (B, Ns, Nt)
    return cost_volume / temperature


def mutual_nearest_neighbour(
    cost_volume: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find mutually nearest-neighbour matches in the cost volume.

    A match (i, j) is valid only if:
        argmax_j(cost[i, :]) == j  AND  argmax_i(cost[:, j]) == i

    Args:
        cost_volume: (Ns, Nt) similarity matrix (single image pair, no batch dim).

    Returns:
        src_idx: (M,) indices of matched source features.
        trg_idx: (M,) indices of matched target features.
    """
    # Forward: best target for each source
    fwd_matches = cost_volume.argmax(dim=1)   # (Ns,)
    # Backward: best source for each target
    bwd_matches = cost_volume.argmax(dim=0)   # (Nt,)

    src_indices = torch.arange(cost_volume.shape[0], device=cost_volume.device)
    mutual_mask = bwd_matches[fwd_matches] == src_indices

    src_idx = src_indices[mutual_mask]
    trg_idx = fwd_matches[mutual_mask]
    return src_idx, trg_idx


def soft_argmax2d(
    heatmap: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Differentiable soft-argmax over a (h*w,) heatmap.

    Converts a probability distribution over grid cells to a continuous
    2D expected coordinate.

    Args:
        heatmap: (B, N_kp, h*w) — similarity scores (will be softmax'd).
        h, w:   grid height and width.

    Returns:
        coords: (B, N_kp, 2) — expected [x, y] coordinates in grid space.
    """
    probs = F.softmax(heatmap, dim=-1)                   # (B, N_kp, h*w)

    # Build grid of (x, y) coordinates
    ys = torch.arange(h, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.arange(w, device=heatmap.device, dtype=heatmap.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(-1)   # (h*w,)
    grid_y = grid_y.reshape(-1)   # (h*w,)

    exp_x = (probs * grid_x).sum(-1)   # (B, N_kp)
    exp_y = (probs * grid_y).sum(-1)   # (B, N_kp)

    return torch.stack([exp_x, exp_y], dim=-1)   # (B, N_kp, 2)
