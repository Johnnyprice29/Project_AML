"""
evaluate.py

Evaluation script — computes PCK@0.1 and PCK@0.05 on the SPair-71k test split,
broken down by category.

Usage:
    python evaluate.py \\
        --dataset_root ./datasets/SPair-71k \\
        --checkpoint   ./checkpoints/best.pth \\
        --alpha        0.1
"""

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataloaders.spair import SPairDataset, collate_spair
from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel
from utils.metrics import pck, pck_per_category


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Semantic Correspondence Model")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--alpha",        type=float, default=0.1)
    parser.add_argument("--img_size",     type=int, default=224)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--num_workers",  type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})

    backbone = DINOv2Extractor(
        model_name=saved_args.get("backbone", "dinov2_vitb14"),
        freeze=True,
    )
    # Applica LoRA con gli stessi parametri del training (default rank=16)
    backbone.model = apply_lora_to_dinov2(
        backbone.model, 
        rank=saved_args.get("lora_rank", 16),
        lora_alpha=saved_args.get("lora_alpha", 32)
    )

    model = SemanticCorrespondenceModel(
        backbone=backbone,
        proj_dim=saved_args.get("proj_dim", 256),
        temperature=saved_args.get("temperature", 0.05),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ---- Dataset ----
    test_ds     = SPairDataset(args.dataset_root, split="test", img_size=args.img_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_spair)
    print(f"[INFO] Test set: {len(test_ds)} pairs")
    # ---- Evaluation ----
    all_pred, all_gt, all_cats = [], [], []

    for batch in tqdm(test_loader, desc="Evaluating"):
        src_img = batch["src_img"].to(device)
        trg_img = batch["trg_img"].to(device)
        src_kps = batch["src_kps"].to(device)
        trg_kps = batch["trg_kps"].to(device)
        mask    = batch["kps_mask"].to(device)

        out = model(src_img, trg_img, src_kps=src_kps)
        pred_kps = out["pred_kps"]   # (B, N, 2)

        for b in range(len(src_img)):
            # Solo i keypoint validi per questo sample del batch
            n_valid = int(mask[b].sum().item())
            all_pred.append(pred_kps[b, :n_valid].cpu())
            all_gt.append(trg_kps[b, :n_valid].cpu())
            all_cats.append(batch["category"][b])

    # ---- Metrics ----
    for alpha in [0.1, 0.05]:
        scores = [
            pck(p.unsqueeze(0), g.unsqueeze(0), img_size=args.img_size, alpha=alpha).item()
            for p, g in zip(all_pred, all_gt)
        ]
        mean_pck = np.mean(scores)
        print(f"\nPCK @ {alpha:.2f} = {mean_pck * 100:.2f}%")

    per_cat = pck_per_category(all_pred, all_gt, all_cats,
                               img_size=args.img_size, alpha=args.alpha)
    print(f"\nPer-category PCK @ {args.alpha}:")
    for cat, score in sorted(per_cat.items()):
        print(f"  {cat:<20s}  {score * 100:.2f}%")
    print(f"\n  Mean: {np.mean(list(per_cat.values())) * 100:.2f}%")


if __name__ == "__main__":
    main()
