"""
utils/curriculum.py

Curriculum Learning for Semantic Correspondence (Stage 2-C).

Strategy:
  - "Easy" pairs:  small viewpoint change, similar scale, same image category.
  - "Hard" pairs:  large viewpoint / scale change, cluttered background.

Difficulty is estimated from SPair-71k annotation metadata:
  - `vpvar`  : viewpoint variation level  (0 = same, 3 = extreme)
  - `scvar`  : scale variation level      (0 = same, 3 = extreme)
  - `trncvar`: truncation variation       (0 = none, 3 = heavy)

During training:
  Epoch 1..curriculum_epochs → sample only pairs with difficulty ≤ current_threshold
  After curriculum_epochs    → use all pairs (standard training)

Usage in train.py:
    from utils.curriculum import CurriculumSampler
    sampler = CurriculumSampler(train_ds, total_epochs=args.epochs,
                                curriculum_epochs=args.curriculum_epochs)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, ...)
    # At the start of each epoch:
    sampler.set_epoch(epoch)
"""

import json
import os
import math
import random
from typing import List, Iterator

import numpy as np
from torch.utils.data import Sampler

from dataloaders.spair import SPairDataset


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

def compute_pair_difficulty(ann: dict) -> float:
    """
    Compute a scalar difficulty score ∈ [0, 1] for a single annotation dict.

    Uses viewpoint variation (vpvar), scale variation (scvar), and
    truncation variation (trncvar) from SPair-71k metadata.
    Unknown fields default to mid-range (0.5).
    """
    vpvar   = ann.get("vpvar",   1)   # int 0-3
    scvar   = ann.get("scvar",   1)
    trncvar = ann.get("trncvar", 0)

    # Normalise each to [0, 1] and compute weighted mean
    score = (0.5 * vpvar + 0.3 * scvar + 0.2 * trncvar) / 3.0
    return float(np.clip(score, 0.0, 1.0))


def score_dataset(dataset: SPairDataset) -> List[float]:
    """
    Score every sample in the dataset and return a list of difficulty scores.
    Scores are cached in memory (no disk I/O after first call).
    """
    scores = []
    for ann_path in dataset.samples:
        with open(ann_path, "r") as f:
            ann = json.load(f)
        scores.append(compute_pair_difficulty(ann))
    return scores


# ---------------------------------------------------------------------------
# Curriculum Sampler
# ---------------------------------------------------------------------------

class CurriculumSampler(Sampler):
    """
    Epoch-aware batch sampler that progressively exposes harder training pairs.

    Args:
        dataset:            SPairDataset instance.
        batch_size:         Number of samples per batch.
        total_epochs:       Total number of training epochs.
        curriculum_epochs:  Number of epochs to run curriculum (ramp-up phase).
                            After this, all samples are used.
        start_fraction:     Fraction of easiest samples used at epoch 1.
        end_fraction:       Fraction of samples used at epoch `curriculum_epochs`.
        drop_last:          Drop the last incomplete batch.
        seed:               Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: SPairDataset,
        batch_size: int,
        total_epochs: int,
        curriculum_epochs: int = 10,
        start_fraction: float = 0.3,
        end_fraction: float = 1.0,
        drop_last: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset           = dataset
        self.batch_size        = batch_size
        self.total_epochs      = total_epochs
        self.curriculum_epochs = curriculum_epochs
        self.start_fraction    = start_fraction
        self.end_fraction      = end_fraction
        self.drop_last         = drop_last
        self.seed              = seed
        self._epoch            = 0

        print("[Curriculum] Scoring dataset difficulty …", flush=True)
        self._scores = np.array(score_dataset(dataset))
        
        if len(self._scores) == 0:
            raise ValueError("[Curriculum ERROR] Dataset vuoto o campioni non trovati. Verifica il path del dataset.")
            
        # Indices sorted from easiest (0) to hardest (1)
        self._sorted_idx = np.argsort(self._scores).tolist()
        print(f"[Curriculum] Done. Score range: "
              f"[{self._scores.min():.3f}, {self._scores.max():.3f}]")

    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        """Call at the beginning of each epoch (1-indexed)."""
        self._epoch = epoch

    def _active_fraction(self) -> float:
        """Linear ramp from start_fraction to end_fraction over curriculum_epochs."""
        if self._epoch >= self.curriculum_epochs:
            return self.end_fraction
        t = (self._epoch - 1) / max(self.curriculum_epochs - 1, 1)
        return self.start_fraction + t * (self.end_fraction - self.start_fraction)

    def _active_indices(self) -> List[int]:
        frac = self._active_fraction()
        n    = max(1, int(math.ceil(frac * len(self._sorted_idx))))
        return self._sorted_idx[:n]

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[List[int]]:
        rng     = random.Random(self.seed + self._epoch)
        indices = self._active_indices()
        rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batches.append(indices[start : start + self.batch_size])
        if not self.drop_last and len(indices) % self.batch_size != 0:
            batches.append(indices[-(len(indices) % self.batch_size):])

        return iter(batches)

    def __len__(self) -> int:
        n = len(self._active_indices())
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)
