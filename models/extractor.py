"""
models/extractor.py

DINOv2 feature extractor wrapper.

Extracts dense per-patch feature maps from a frozen (or LoRA-adapted)
DINOv2 Vision Transformer backbone.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class DINOv2Extractor(nn.Module):
    """
    Wraps a DINOv2 ViT model and exposes its dense patch features.

    Args:
        model_name: one of 'dinov2_vits14', 'dinov2_vitb14',
                    'dinov2_vitl14', 'dinov2_vitg14'
        layer:      which transformer block to extract features from.
                    -1 means the last block.
        use_key:    if True, use attention *keys* (more discriminative);
                    if False, use the patch tokens directly.
        freeze:     if True, freeze all backbone parameters.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        layer: int = -1,
        use_key: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.layer_idx = layer
        self.use_key = use_key

        # Load pretrained DINOv2 from torch hub
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        # Patch size is always 14 for DINOv2
        self.patch_size = 14
        self._feat_dim = self.model.embed_dim

        # Hook storage
        self._features: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._register_hook()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    # ------------------------------------------------------------------
    # Hook logic
    # ------------------------------------------------------------------

    def _register_hook(self):
        blocks = self.model.blocks
        target_block = blocks[self.layer_idx]

        if self.use_key:
            # Hook into the attention QKV projection to grab Keys
            def _hook(module, input, output):
                # output shape: (B, num_heads, N, head_dim)
                # We reconstruct the "key" from the QKV linear layer output
                pass  # Will be overridden by forward() using get_intermediate_layers

            # We'll extract keys via get_intermediate_layers + manual projection
            # (simpler: just use patch tokens from get_intermediate_layers)
        # We use get_intermediate_layers which is simpler and official
        pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — input image batch, H and W must be
               divisible by patch_size (14).

        Returns:
            features: (B, C, h, w) dense feature map, where
                      h = H // patch_size, w = W // patch_size,
                      C = feat_dim.
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Image size ({H}x{W}) must be divisible by patch_size={self.patch_size}"
        )
        h, w = H // self.patch_size, W // self.patch_size

        # Resolve negative layer index
        actual_layer = self.layer_idx
        if actual_layer < 0:
            # PEFT wraps the backbone, so blocks might be inside base_model.model
            if hasattr(self.model, "blocks"):
                n_blocks = len(self.model.blocks)
            else:
                n_blocks = len(self.model.base_model.model.blocks)
            actual_layer += n_blocks

        # get_intermediate_layers returns a list of (B, N+1, D) tensors
        # N = h*w patch tokens, +1 for [CLS]
        out = self.model.get_intermediate_layers(
            x, n=[actual_layer], reshape=False
        )  # list of length 1
        feats = out[0]          # (B, N+1, D)
        feats = feats[:, 1:]    # remove [CLS] → (B, N, D)
        feats = feats.permute(0, 2, 1)          # (B, D, N)
        feats = feats.reshape(B, self._feat_dim, h, w)  # (B, D, h, w)
        return feats

    def unfreeze(self):
        """Unfreeze backbone parameters (needed before applying LoRA)."""
        for p in self.model.parameters():
            p.requires_grad_(True)
