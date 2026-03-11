"""
models/lora.py

LoRA (Low-Rank Adaptation) utilities for fine-tuning DINOv2.

Instead of re-implementing LoRA from scratch, we leverage the
`peft` library which provides a battle-tested implementation.
This module provides convenience wrappers for applying LoRA
to a DINOv2 backbone.

Reference:
    Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
    https://arxiv.org/abs/2106.09685
"""

from typing import List, Optional
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARNING] `peft` not installed. LoRA will not be available. "
          "Install with: pip install peft")


def apply_lora_to_dinov2(
    model: nn.Module,
    rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA adapters to the attention layers of a DINOv2 backbone.

    Args:
        model:          The DINOv2 nn.Module.
        rank:           LoRA rank r. Higher r = more capacity, more params.
        lora_alpha:     Scaling factor. Output = (alpha/r) * BA * x.
        lora_dropout:   Dropout applied to LoRA path during training.
        target_modules: Which linear layers to adapt. Defaults to
                        the QKV and output projections of every attention block.

    Returns:
        A PEFT-wrapped model with LoRA adapters.
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "The `peft` package is required for LoRA. "
            "Install it with:  pip install peft"
        )

    if target_modules is None:
        # DINOv2 attention layer names (may vary slightly across versions)
        target_modules = ["qkv", "proj"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        # NOTE: TaskType.FEATURE_EXTRACTION is used because DINOv2 is
        # an encoder-only model with no causal masking.
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Manual LoRA (no peft dependency) — simple reference implementation
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear with an added LoRA bypass.

        y = W @ x  +  (alpha/r) * B @ A @ x

    W is frozen; only A and B are trained.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank         = rank
        self.scaling      = lora_alpha / rank

        # Frozen original weights
        self.weight = original_linear.weight  # not a nn.Parameter — reference
        self.bias   = original_linear.bias

        # Trainable LoRA matrices
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        self.dropout = nn.Dropout(p=lora_dropout)

        # Initialise: A ~ N(0, 0.02), B = 0 → no change at start
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out
