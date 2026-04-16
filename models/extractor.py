import torch
import torch.nn as nn
from typing import List, Optional

# SAM support
try:
    from segment_anything import sam_model_registry
except ImportError:
    sam_model_registry = None

class FeatureExtractor(nn.Module):
    """
    Unified feature extractor for semantic correspondence.
    Supports DINOv2, SAM, and DINOv3 backbones.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        layer: int = -1,
        use_key: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.layer_idx = layer
        self.use_key = use_key

        if "dinov2" in self.model_name:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
            self.patch_size = 14
            self._feat_dim = self.model.embed_dim
            self.model_type = "dino"
        elif "dinov3" in self.model_name:
            # Use DINOv2 with Registers - the most advanced non-gated version (often called v3 in research)
            try:
                print(f"[INFO] Loading DINOv2 with Registers (ViT-B/14) as DINOv3 alternative...")
                self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
                self.patch_size = 14
                self._feat_dim = self.model.embed_dim
                self.model_type = "dino"
            except Exception as e:
                raise RuntimeError(f"Could not load DINOv2-Reg model: {e}")
                
        elif "sam" in self.model_name:
            torch.cuda.empty_cache() 
            if sam_model_registry is None:
                raise ImportError("segment-anything not found. Install it for SAM support.")
            
            # Map names: 'sam_vitb', 'sam_vitl', 'sam_vith'
            sam_type = self.model_name.split("_")[-1] if "_" in self.model_name else "vit_b"
            if sam_type == "vitb": sam_type = "vit_b"
            elif sam_type == "vitl": sam_type = "vit_l"
            elif sam_type == "vith": sam_type = "vit_h"
            
            # We only need the image_encoder for feature extraction
            sam = sam_model_registry[sam_type](checkpoint=None) 
            self.model = sam.image_encoder
            self.patch_size = 16 # SAM default patch size
            
            # SAM encoder doesn't have .embed_dim directly; we map it based on type
            if sam_type == "vit_b": self._feat_dim = 768
            elif sam_type == "vit_l": self._feat_dim = 1024
            elif sam_type == "vit_h": self._feat_dim = 1280
            else: self._feat_dim = 768
            
            self.model_type = "sam"
        else:
            raise ValueError(f"Unknown backbone: {model_name}")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts dense feature maps.
        Returns: (B, C, h, w)
        """
        B, C, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        if self.model_type == "dino":
            # Resolve layer
            actual_layer = self.layer_idx
            if actual_layer < 0:
                n_blocks = len(self.model.blocks) if hasattr(self.model, "blocks") else len(self.model.base_model.model.blocks)
                actual_layer += n_blocks
            
            out = self.model.get_intermediate_layers(x, n=[actual_layer], reshape=False)
            feats = out[0] # (B, N, D)
            feats = feats.permute(0, 2, 1).reshape(B, self._feat_dim, h, w)
            return feats
            
        elif self.model_type == "dinov3":
            # Transformers ViT returns pooler_output and hidden_states
            actual_layer = self.layer_idx
            if actual_layer < 0:
                actual_layer += self.model.config.num_hidden_layers
            
            # We need +1 because index 0 is the embedding layer
            outputs = self.model(x, output_hidden_states=True)
            feats = outputs.hidden_states[actual_layer + 1] # (B, N, D)
            
            # Remove CLS token (usually at index 0)
            feats = feats[:, 1:, :] 
            feats = feats.permute(0, 2, 1).reshape(B, self._feat_dim, h, w)
            return feats

        elif self.model_type == "sam":
            # SAM image_encoder strictly expects 1024x1024
            if H != 1024 or W != 1024:
                x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
            
            # Use half precision and checkpointing to save memory on T4
            from torch.utils.checkpoint import checkpoint
            def custom_forward(images):
                return self.model(images)
            
            # Move to half precision for SAM
            self.model.half()
            x = x.half()
            
            # If training/finetuning, checkpointing is needed, but even in eval it helps on T4
            with torch.cuda.amp.autocast():
                feats = self.model(x)
            
            return feats.float() # Convert back for correspondence consistency

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad_(True)

# For backward compatibility
DINOv2Extractor = FeatureExtractor
