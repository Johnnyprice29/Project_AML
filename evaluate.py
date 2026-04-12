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

import os
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
    parser.add_argument("--checkpoint",   type=str, default="", help="Path to best.pth")
    parser.add_argument("--baseline_only", action="store_true", help="Test pure DINOv2 without LoRA/weights")
    parser.add_argument("--layer",        type=int, default=-1, help="Transformer layer to extract features from (-1=last)")
    parser.add_argument("--alpha",        type=float, default=0.1)
    parser.add_argument("--img_size",     type=int, default=224)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--no_adaptive_win", action="store_true", help="Disable adaptive window")
    parser.add_argument("--results_file", type=str, default="", help="Path to save text results")
    return parser.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- Load checkpoint ----
    if not args.baseline_only:
        assert args.checkpoint, "Must provide --checkpoint unless --baseline_only is used"
        ckpt = torch.load(args.checkpoint, map_location=device)
        saved_args = ckpt.get("args", {})
        # Se l'utente non ha specificato un layer, usa quello del checkpoint
        if args.layer == -1 and "layer" in saved_args:
            args.layer = saved_args["layer"]
    else:
        print("[INFO] BASELINE MODE: Testing pure DINOv2 without LoRA!")
        ckpt = {}
        saved_args = {}

    backbone = DINOv2Extractor(
        model_name=saved_args.get("backbone", "dinov2_vitb14"),
        layer=args.layer,
        freeze=True,
    )
    
    if not args.baseline_only:
        peft_type = saved_args.get("peft_type", "lora")
        if peft_type == "lora":
            print(f"[INFO] Applying LoRA (rank={saved_args.get('lora_rank', 16)}) to backbone.")
            backbone.model = apply_lora_to_dinov2(
                backbone.model, 
                rank=saved_args.get("lora_rank", 16),
                lora_alpha=saved_args.get("lora_alpha", 32)
            )
        elif peft_type == "bitfit":
            print("[INFO] BitFit detected: unfreezing bias parameters before loading.")
            for n, p in backbone.model.named_parameters():
                if "bias" in n: p.requires_grad = True

    model = SemanticCorrespondenceModel(
        backbone=backbone,
        proj_dim=saved_args.get("proj_dim", 256),
        temperature=saved_args.get("temperature", 0.05),
        use_adaptive_win=not args.no_adaptive_win,
    ).to(device)

    if not args.baseline_only:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    model.eval()

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
    final_output = []
    ckpt_name = os.path.basename(args.checkpoint) if args.checkpoint else "Baseline"
    final_output.append(f"--- Evaluation Results: {ckpt_name} ---")

    for alpha in [0.1, 0.05]:
        scores = [
            pck(p.unsqueeze(0), g.unsqueeze(0), img_size=args.img_size, alpha=alpha).item()
            for p, g in zip(all_pred, all_gt)
        ]
        mean_pck = np.mean(scores)
        line = f"PCK @ {alpha:.2f} = {mean_pck * 100:.2f}%"
        print(f"\n{line}")
        final_output.append(line)

    per_cat = pck_per_category(all_pred, all_gt, all_cats,
                               img_size=args.img_size, alpha=args.alpha)
    print(f"\nPer-category PCK @ {args.alpha}:")
    final_output.append(f"\nPer-category PCK @ {args.alpha}:")
    for cat, score in sorted(per_cat.items()):
        line = f"  {cat:<20s}  {score * 100:.2f}%"
        print(line)
        final_output.append(line)
    
    mean_val = f"\n  Mean: {np.mean(list(per_cat.values())) * 100:.2f}%"
    print(mean_val)
    final_output.append(mean_val)

    if args.results_file:
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        with open(args.results_file, "w") as f:
            f.write("\n".join(final_output))
        print(f"\n[INFO] Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
