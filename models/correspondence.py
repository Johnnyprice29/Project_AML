"""
models/correspondence.py

Full semantic correspondence model:
    DINOv2 (optionally with LoRA)  →  dense feature maps  →  nearest-neighbour matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.extractor import DINOv2Extractor


class SemanticCorrespondenceModel(nn.Module):
    """
    End-to-end model for semantic correspondence.

    Pipeline:
        1. Extract dense DINOv2 features for source & target images.
        2. (Optional) Apply a lightweight projection head.
        3. Compute cosine-similarity cost volume.
        4. Return predicted target coordinates for each source keypoint.

    Args:
        backbone:       DINOv2Extractor instance.
        proj_dim:       Output dimension of the projection head.
                        Set to None to skip projection.
        temperature:    Softmax temperature for the cost volume.
                        Lower → sharper/more confident matching.
    """

    def __init__(
        self,
        backbone: DINOv2Extractor,
        proj_dim: Optional[int] = 256,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.backbone    = backbone
        self.temperature = temperature
        feat_dim         = backbone.feat_dim

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
    ) -> dict:
        """
        Args:
            src_img:  (B, 3, H, W)
            trg_img:  (B, 3, H, W)
            src_kps:  (B, N, 2)  keypoint pixel coordinates in source image [x, y]
                      If None, returns the full cost volume only.

        Returns:
            dict with keys:
              'cost_volume':   (B, h*w, h*w) — raw similarity matrix
              'pred_kps':      (B, N, 2) — predicted target keypoints  [optional]
              'feat_grid_hw':  (h, w) — feature grid size
        """
        src_feats, (h, w) = self.extract_features(src_img)   # (B, Ns, C)
        trg_feats, _      = self.extract_features(trg_img)   # (B, Nt, C)

        # Cosine similarity cost volume: (B, Ns, Nt)
        cost_volume = torch.bmm(src_feats, trg_feats.transpose(1, 2))  # (B, N, N)
        cost_volume = cost_volume / self.temperature

        output = {
            "cost_volume":  cost_volume,
            "feat_grid_hw": (h, w),
        }

        if src_kps is not None:
            pred_kps = self._match_keypoints(cost_volume, src_kps, h, w, src_img)
            output["pred_kps"] = pred_kps

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
    ) -> torch.Tensor:
        """
        For each source keypoint, find the best-matching target grid cell
        and convert back to pixel coordinates.

        Returns:
            pred_kps: (B, N_kp, 2) in pixel coords.
        """
        B, N_kp, _ = src_kps.shape
        _, H, W = src_img.shape[1:]

        # Convert keypoint pixel coords → feature grid indices
        # grid_x ∈ [0, w-1],  grid_y ∈ [0, h-1]
        kp_grid_x = (src_kps[..., 0] / W * w).long().clamp(0, w - 1)  # (B, N_kp)
        kp_grid_y = (src_kps[..., 1] / H * h).long().clamp(0, h - 1)
        kp_idx    = kp_grid_y * w + kp_grid_x                          # (B, N_kp) flat index

        # Gather rows of cost_volume corresponding to source keypoints
        # kp_idx: (B, N_kp) → expand to (B, N_kp, Nt_grid)
        kp_idx_exp    = kp_idx.unsqueeze(-1).expand(-1, -1, h * w)
        kp_similarity = cost_volume.gather(1, kp_idx_exp)              # (B, N_kp, Nt_grid)

        # Argmax → best matching target grid cell
        best_trg_idx = kp_similarity.argmax(dim=-1)                    # (B, N_kp)

        # Convert flat index → (grid_x, grid_y) → pixel coords
        trg_grid_y = best_trg_idx // w                                 # (B, N_kp)
        trg_grid_x = best_trg_idx % w

        # Map back to image pixel space (centre of the patch)
        pred_x = (trg_grid_x.float() + 0.5) / w * W
        pred_y = (trg_grid_y.float() + 0.5) / h * H

        pred_kps = torch.stack([pred_x, pred_y], dim=-1)               # (B, N_kp, 2)
        return pred_kps
