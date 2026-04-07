"""
utils/segment_aware.py

Segment-Aware Correspondence (Stage 4-B).

Idea:
  Use SAM (Segment Anything Model) to generate an object mask for the TARGET
  image. Before computing the argmax / soft-argmax over the similarity row,
  mask out all positions that do NOT belong to the object. This dramatically
  reduces false positives in cluttered scenes.

Pipeline:
  1. Run SAM on target image → binary mask (H, W).
  2. Downsample mask to feature-grid resolution (h, w).
  3. Zero-out similarity scores outside the mask (set to -inf).
  4. Proceed with normal soft-argmax / adaptive windowing.

Requirements:
  pip install segment-anything
  # Download SAM weights:
  # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Usage:
  segmentor = SAMSegmentor(checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b")
  mask_hw   = segmentor.get_object_mask(trg_img_pil, point_coords=[(cx, cy)])
  filtered  = apply_mask_to_sim_row(sim_row, mask_hw, h, w)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from PIL import Image


# ---------------------------------------------------------------------------
# SAM wrapper
# ---------------------------------------------------------------------------

class SAMSegmentor:
    """
    Thin wrapper around SAM (Segment Anything Model).

    Falls back gracefully if `segment_anything` is not installed.

    Args:
        checkpoint:  Path to SAM .pth weights file.
        model_type:  One of 'vit_b', 'vit_l', 'vit_h'.
        device:      'cuda' or 'cpu'.
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cuda",
    ):
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment_anything is not installed.\n"
                "Install with:  pip install segment-anything\n"
                "Weights from:  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )

        self.device = device
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)

    # ------------------------------------------------------------------

    def get_object_mask(
        self,
        image: Image.Image,
        point_coords: List[Tuple[int, int]],
        point_labels: Optional[List[int]] = None,
        multimask_output: bool = False,
    ) -> np.ndarray:
        """
        Generate a binary segmentation mask for the object containing the
        given point prompts.

        Args:
            image:            PIL Image (RGB).
            point_coords:     List of (x, y) prompt points [image pixel space].
            point_labels:     1 = foreground, 0 = background. Defaults to all 1.
            multimask_output: If True, returns the mask with highest IoU score.

        Returns:
            mask: (H, W) numpy bool array — True inside the object mask.
        """
        img_np = np.array(image.convert("RGB"))
        self.predictor.set_image(img_np)

        if point_labels is None:
            point_labels = [1] * len(point_coords)

        coords_np  = np.array(point_coords,  dtype=np.float32)
        labels_np  = np.array(point_labels,  dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=coords_np,
            point_labels=labels_np,
            multimask_output=True,
        )
        # Pick the mask with the highest confidence score
        best_idx = scores.argmax()
        return masks[best_idx].astype(bool)   # (H, W)

    # ------------------------------------------------------------------

    def get_mask_for_keypoint(
        self,
        image: Image.Image,
        keypoint_xy: Tuple[float, float],
    ) -> np.ndarray:
        """
        Convenience: generate mask for the object at a single keypoint location.

        Args:
            image:         PIL Image (RGB).
            keypoint_xy:   (x, y) in image pixel coordinates.

        Returns:
            mask: (H, W) numpy bool array.
        """
        return self.get_object_mask(image, [keypoint_xy])


# ---------------------------------------------------------------------------
# Mask application to similarity row
# ---------------------------------------------------------------------------

def downsample_mask(
    mask: np.ndarray,       # (H_img, W_img) bool
    h: int,
    w: int,
    threshold: float = 0.3,
) -> torch.Tensor:
    """
    Resize a boolean mask to the feature-grid resolution (h, w).

    A feature cell is considered "inside the object" if at least
    `threshold` fraction of its corresponding image pixels are inside the mask.

    Args:
        mask:      (H, W) bool numpy array.
        h, w:      target feature-grid size.
        threshold: fraction of pixels inside mask to count the cell as "in".

    Returns:
        grid_mask: (h*w,) bool tensor.
    """
    mask_float = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    # Average pool: (1, 1, H, W) → (1, 1, h, w)
    pooled = F.adaptive_avg_pool2d(mask_float, (h, w)).squeeze()   # (h, w)
    return (pooled >= threshold).reshape(-1)                          # (h*w,)


def apply_mask_to_sim_row(
    sim_row:   torch.Tensor,   # (h*w,) raw similarity scores
    mask:      np.ndarray,     # (H_img, W_img) bool — object mask in image space
    h: int,
    w: int,
    fill_value: float = float("-inf"),
) -> torch.Tensor:
    """
    Zero-out (mask out) similarity scores for positions outside the object mask.

    Args:
        sim_row:    (h*w,) float tensor — raw or temperature-scaled similarities.
        mask:       (H, W) bool numpy array — True = inside object.
        h, w:       feature grid dimensions.
        fill_value: value to assign for out-of-mask positions.
                    Use -inf so softmax gives ~0 probability there.

    Returns:
        masked_sim: (h*w,) tensor — scores outside mask set to fill_value.
    """
    grid_mask = downsample_mask(mask, h, w).to(sim_row.device)   # (h*w,) bool
    masked    = sim_row.clone()
    masked[~grid_mask] = fill_value
    return masked


# ---------------------------------------------------------------------------
# Batched helper for the model forward pass
# ---------------------------------------------------------------------------

def apply_masks_to_cost_volume(
    cost_volume: torch.Tensor,    # (B, Ns, Nt)  where Nt = h*w
    trg_masks:   List[Optional[np.ndarray]],  # list of B masks (or None to skip)
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Apply per-sample object masks to every row of the cost volume.

    Args:
        cost_volume: (B, Ns, Nt) similarity tensor.
        trg_masks:   List[Optional[ndarray (H,W)]] of length B.
                     If None for a sample, that sample is left unmasked.
        h, w:        feature grid dimensions (Nt = h*w).

    Returns:
        masked_cv: (B, Ns, Nt) — modified in-place clone.
    """
    masked_cv = cost_volume.clone()
    for b, mask in enumerate(trg_masks):
        if mask is None:
            continue
        grid_mask = downsample_mask(mask, h, w).to(cost_volume.device)  # (h*w,)
        # Broadcast: mask entire Ns rows at once
        masked_cv[b, :, ~grid_mask] = float("-inf")
    return masked_cv
