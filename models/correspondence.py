"""
models/correspondence.py

Full semantic correspondence model:
    DINOv2 (+ LoRA)  →  dense features  →  [SAM masking]  →  Adaptive Window Soft-Argmax

New features vs. baseline:
  • Segment-Aware Correspondence: object masks from SAM filter the cost volume
    so matches are constrained inside the target object.
  • Adaptive Window Soft-Argmax: window radius dynamically adapts to heatmap
    entropy for sub-pixel precision without being over-confident on hard pairs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.extractor import DINOv2Extractor
from utils.adaptive_window import batched_adaptive_softargmax
from utils.segment_aware import apply_masks_to_cost_volume


class SemanticCorrespondenceModel(nn.Module):
    """
    End-to-end model for semantic correspondence.

    Pipeline:
        1. Extract dense DINOv2 features for source & target images.
        2. (Optional) Apply a lightweight projection head.
        3. Compute cosine-similarity cost volume.
        4. [Optional] Apply SAM segment masks to cost volume  (Segment-Aware).
        5. Adaptive Window Soft-Argmax per keypoint             (Adaptive Windowing).

    Args:
        backbone:          DINOv2Extractor instance.
        proj_dim:          Output dim of projection head. None to skip.
        temperature:       Softmax temperature for the cost volume.
        use_adaptive_win:  If True, use entropy-based adaptive soft-argmax.
                           If False, fall back to plain argmax (faster).
        aw_min_radius:     Minimum window half-size (feature cells).
        aw_max_radius:     Maximum window half-size (feature cells).
    """

    def __init__(
        self,
        backbone: DINOv2Extractor,
        proj_dim: Optional[int] = 256,
        temperature: float = 0.05,
        use_adaptive_win: bool = True,
        aw_min_radius: int = 2,
        aw_max_radius: int = 7,
    ):
        super().__init__()
        self.backbone         = backbone
        self.temperature      = temperature
        self.use_adaptive_win = use_adaptive_win
        self.aw_min_radius    = aw_min_radius
        self.aw_max_radius    = aw_max_radius
        feat_dim              = backbone.feat_dim

        # Optional projection head (shared weights for src & trg)
        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
            )
            self.out_dim = proj_dim
        else:
            self.proj    = nn.Identity()
            self.out_dim = feat_dim

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feats: (B, h*w, C)  — L2-normalised, ready for dot-product similarity
        """
        feat_map = self.backbone(x)            # (B, D, h, w)
        B, D, h, w = feat_map.shape
        feats = feat_map.permute(0, 2, 3, 1)   # (B, h, w, D)
        feats = feats.reshape(B, h * w, D)     # (B, N, D)
        feats = self.proj(feats)               # (B, N, C)
        feats = F.normalize(feats, dim=-1)     # L2 norm
        return feats, (h, w)

    def forward(
        self,
        src_img: torch.Tensor,
        trg_img: torch.Tensor,
        src_kps: Optional[torch.Tensor] = None,
        trg_masks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> dict:
        """
        Args:
            src_img:   (B, 3, H, W)
            trg_img:   (B, 3, H, W)
            src_kps:   (B, N, 2) keypoint pixel coordinates in source [x, y].
                       If None, returns cost volume only (no keypoint prediction).
            trg_masks: List[Optional[ndarray(H,W)]] of length B — SAM binary
                       masks for the target object. None entries are skipped.
                       When provided, activates Segment-Aware Correspondence.

        Returns:
            dict with:
              'cost_volume':   (B, Ns, Nt) — similarity matrix (after masking if used)
              'pred_kps':      (B, N, 2)   — predicted target keypoints in pixel space
              'entropies':     (B, N)      — per-keypoint heatmap entropy (for logging)
              'feat_grid_hw':  (h, w)
        """
        src_feats, (h, w) = self.extract_features(src_img)   # (B, Ns, C)
        trg_feats, _      = self.extract_features(trg_img)   # (B, Nt, C)

        # --- Cosine similarity cost volume ---
        cost_volume = torch.bmm(src_feats, trg_feats.transpose(1, 2))  # (B, Ns, Nt)
        cost_volume = cost_volume / self.temperature

        # --- Segment-Aware: mask out-of-object target positions ---
        if trg_masks is not None:
            cost_volume = apply_masks_to_cost_volume(cost_volume, trg_masks, h, w)

        output = {
            "cost_volume":  cost_volume,
            "feat_grid_hw": (h, w),
        }

        if src_kps is not None:
            pred_kps, entropies = self._match_keypoints(
                cost_volume, src_kps, h, w, src_img
            )
            output["pred_kps"]  = pred_kps
            output["entropies"] = entropies

        return output

    # ------------------------------------------------------------------
    # Keypoint matching
    # ------------------------------------------------------------------

    def _match_keypoints(
        self,
        cost_volume: torch.Tensor,   # (B, Ns_grid, Nt_grid)
        src_kps: torch.Tensor,       # (B, N_kp, 2)  in pixel coords [x, y]
        h: int,
        w: int,
        src_img: torch.Tensor,       # (B, 3, H, W) — to get img size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each source keypoint, find the best-matching target position and
        convert back to pixel coordinates.

        If use_adaptive_win=True:  Adaptive Window Soft-Argmax (sub-pixel).
        If use_adaptive_win=False: Plain hard-argmax (faster, coarser).

        Returns:
            pred_kps:  (B, N_kp, 2) in pixel coords.
            entropies: (B, N_kp) heatmap entropy per keypoint.
        """
        B, N_kp, _ = src_kps.shape
        _, H, W    = src_img.shape[1:]

        # Convert source keypoints → flat feature-grid indices
        kp_grid_x = (src_kps[..., 0] / W * w).long().clamp(0, w - 1)  # (B, N_kp)
        kp_grid_y = (src_kps[..., 1] / H * h).long().clamp(0, h - 1)
        kp_idx    = kp_grid_y * w + kp_grid_x                          # (B, N_kp)

        # Gather similarity rows: (B, N_kp, Nt_grid)
        kp_idx_exp    = kp_idx.unsqueeze(-1).expand(-1, -1, h * w)
        kp_similarity = cost_volume.gather(1, kp_idx_exp)              # (B, N_kp, h*w)

        if self.use_adaptive_win:
            # --- Adaptive Window Soft-Argmax (sub-pixel) ---
            # Returns coords in feature-grid space
            grid_coords, entropies = batched_adaptive_softargmax(
                kp_similarity, h, w,
                min_radius=self.aw_min_radius,
                max_radius=self.aw_max_radius,
            )   # grid_coords: (B, N_kp, 2) [x,y] in grid space

            # Scale from grid space → pixel space
            pred_x = (grid_coords[..., 0] + 0.5) / w * W
            pred_y = (grid_coords[..., 1] + 0.5) / h * H

        else:
            # --- Plain hard-argmax (baseline fallback) ---
            best_trg_idx = kp_similarity.argmax(dim=-1)   # (B, N_kp)
            trg_grid_y   = best_trg_idx // w
            trg_grid_x   = best_trg_idx % w
            pred_x = (trg_grid_x.float() + 0.5) / w * W
            pred_y = (trg_grid_y.float() + 0.5) / h * H
            entropies = torch.zeros(B, N_kp, device=src_kps.device)

        pred_kps = torch.stack([pred_x, pred_y], dim=-1)   # (B, N_kp, 2)
        return pred_kps, entropies
