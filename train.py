"""
train.py

Training script for the Semantic Correspondence model (Project 5).

Usage:
    python train.py --dataset_root ./datasets/SPair-71k \\
                    --backbone dinov2_vitb14 \\
                    --lora_rank 16 \\
                    --batch_size 16 \\
                    --lr 1e-4 \\
                    --epochs 20 \\
                    --output_dir ./checkpoints
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SPairDataset
from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel
from utils.metrics import pck


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def correspondence_loss(
    pred_kps: torch.Tensor,   # (B, N, 2)
    gt_kps:   torch.Tensor,   # (B, N, 2)
) -> torch.Tensor:
    """
    Simple L2 regression loss on keypoint coordinates.
    TODO: replace / augment with a contrastive or cyclic consistency loss.
    """
    return F.mse_loss(pred_kps, gt_kps)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_pck  = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
    for batch in pbar:
        src_img  = batch["src_img"].to(device)
        trg_img  = batch["trg_img"].to(device)
        src_kps  = batch["src_kps"].to(device)   # (B, N, 2)
        trg_kps  = batch["trg_kps"].to(device)

        optimizer.zero_grad()
        out = model(src_img, trg_img, src_kps=src_kps)

        loss = correspondence_loss(out["pred_kps"], trg_kps)
        loss.backward()
        optimizer.step()

        pck_score = pck(out["pred_kps"].detach(), trg_kps,
                        img_size=src_img.shape[-1], alpha=0.1)

        running_loss += loss.item()
        running_pck  += pck_score.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", pck=f"{pck_score.item():.4f}")

    n = len(loader)
    return running_loss / n, running_pck / n


@torch.no_grad()
def validate(model, loader, device, alpha=0.1):
    model.eval()
    total_pck = 0.0

    for batch in tqdm(loader, desc="  Val", unit="batch"):
        src_img = batch["src_img"].to(device)
        trg_img = batch["trg_img"].to(device)
        src_kps = batch["src_kps"].to(device)
        trg_kps = batch["trg_kps"].to(device)

        out = model(src_img, trg_img, src_kps=src_kps)
        total_pck += pck(out["pred_kps"], trg_kps,
                         img_size=src_img.shape[-1], alpha=alpha).item()

    return total_pck / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Semantic Correspondence Model")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--backbone",     type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--img_size",     type=int, default=224)
    parser.add_argument("--lora_rank",    type=int, default=16)
    parser.add_argument("--lora_alpha",   type=int, default=32)
    parser.add_argument("--proj_dim",     type=int, default=256)
    parser.add_argument("--temperature",  type=float, default=0.05)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--epochs",       type=int, default=20)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--output_dir",   type=str, default="./checkpoints")
    parser.add_argument("--seed",         type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Datasets & Loaders ----
    train_ds = SPairDataset(args.dataset_root, split="trn", img_size=args.img_size)
    val_ds   = SPairDataset(args.dataset_root, split="val", img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"[INFO] Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs")

    # ---- Model ----
    backbone = DINOv2Extractor(model_name=args.backbone, freeze=False)
    backbone.model = apply_lora_to_dinov2(
        backbone.model,
        rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    model = SemanticCorrespondenceModel(
        backbone=backbone,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
    ).to(device)

    # Only train LoRA params + projection head
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- Training ----
    best_pck = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pck = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_pck               = validate(model, val_loader, device, alpha=0.1)
        scheduler.step()

        print(f"Epoch {epoch:03d}  "
              f"loss={train_loss:.4f}  train_pck@0.1={train_pck:.4f}  "
              f"val_pck@0.1={val_pck:.4f}")

        if val_pck > best_pck:
            best_pck = val_pck
            ckpt_path = os.path.join(args.output_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pck": val_pck,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}  (PCK@0.1={val_pck:.4f})")

    print(f"\n[DONE] Best val PCK@0.1 = {best_pck:.4f}")


if __name__ == "__main__":
    main()
