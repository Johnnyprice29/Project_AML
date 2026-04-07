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
import shutil

from dataloaders.spair import SPairDataset, collate_spair
from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel
from utils.metrics import pck
from utils.curriculum import CurriculumSampler


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def correspondence_loss(
    pred_kps: torch.Tensor,   # (B, N, 2)
    gt_kps:   torch.Tensor,   # (B, N, 2)
    mask:     torch.Tensor,   # (B, N) boolean mask
) -> torch.Tensor:
    """
    MSE loss on keypoint coordinates, computed only on valid (non-padded) keypoints.
    """
    mse = F.mse_loss(pred_kps, gt_kps, reduction='none').mean(dim=-1) # (B, N)
    return mse[mask].mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, config, scaler):
    model.train()
    running_loss = 0.0
    running_pck  = 0.0
    running_ent  = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
    for i, batch in enumerate(pbar):
        if config.get("max_batches") and i >= config["max_batches"]:
            break
            
        src_img, trg_img = batch["src_img"].to(device), batch["trg_img"].to(device)
        src_kps  = batch["src_kps"].to(device)   # (B, N, 2) padded
        trg_kps  = batch["trg_kps"].to(device)
        kps_mask = batch["kps_mask"].to(device)  # (B, N)

        optimizer.zero_grad()
        
        # AMP Mixed Precision per raddoppiare la velocità
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            out = model(src_img, trg_img, src_kps=src_kps)
            loss = correspondence_loss(out["pred_kps"], trg_kps, kps_mask)

        scaler.scale(loss).backward()
        
        # Gradient clipping deve avvenire dopo unscale
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        pck_score = pck(out["pred_kps"].detach(), trg_kps,
                        img_size=src_img.shape[-1], alpha=0.1, mask=kps_mask)
        mean_ent  = out.get("entropies", torch.zeros(1)).mean().item()

        running_loss += loss.item()
        running_pck  += pck_score.item()
        running_ent  += mean_ent
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            pck=f"{pck_score.item():.4f}",
            entropy=f"{mean_ent:.3f}"
        )

    n = len(loader) if not config.get("max_batches") else config["max_batches"]
    return running_loss / n, running_pck / n, running_ent / n


@torch.no_grad()
def validate(model, loader, device, alpha=0.1):
    model.eval()
    total_pck = 0.0

    for batch in tqdm(loader, desc="  Val", unit="batch"):
        src_img = batch["src_img"].to(device)
        trg_img = batch["trg_img"].to(device)
        src_kps = batch["src_kps"].to(device)
        trg_kps = batch["trg_kps"].to(device)
        mask    = batch["kps_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            out = model(src_img, trg_img, src_kps=src_kps)
            
        total_pck += pck(out["pred_kps"], trg_kps, img_size=src_img.shape[-1], 
                         alpha=alpha, mask=mask).item()

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
    parser.add_argument("--max_batches",  type=int, default=None,
                        help="Limita il numero di batch per epoch (utile per debug veloce)")
    parser.add_argument("--output_dir",   type=str, default="./checkpoints")
    parser.add_argument("--backup_dir",   type=str, default="",
                        help="Cartella opzionale (es. su Drive) dove copiare il checkpoint.")
    parser.add_argument("--seed",         type=int, default=42)
    # --- Curriculum Learning ---
    parser.add_argument("--curriculum_epochs",   type=int,   default=10,
                        help="Epochs for the difficulty ramp-up phase (0 = no curriculum)")
    parser.add_argument("--curriculum_start_frac", type=float, default=0.3,
                        help="Fraction of easiest pairs used at epoch 1")
    # --- Model options ---
    parser.add_argument("--no_adaptive_win", action="store_true",
                        help="Disable Adaptive Window Soft-Argmax (use hard argmax)")
    parser.add_argument("--aw_min_radius",   type=int, default=2)
    parser.add_argument("--aw_max_radius",   type=int, default=7)
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

    # ---- Config (from Lab 3 style) ----
    config = {
        "dataset_root": args.dataset_root,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "proj_dim": args.proj_dim,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "curriculum_epochs": args.curriculum_epochs,
        "curriculum_start_frac": args.curriculum_start_frac,
        "no_adaptive_win": args.no_adaptive_win,
        "aw_min_radius": args.aw_min_radius,
        "aw_max_radius": args.aw_max_radius,
        "max_batches": args.max_batches,
        "backup_dir": args.backup_dir,
    }

    # ---- Datasets & Loaders ----
    train_ds = SPairDataset(config["dataset_root"], split="trn", img_size=config["img_size"])
    val_ds   = SPairDataset(config["dataset_root"], split="val", img_size=config["img_size"])

    # Curriculum sampler: progressively expose harder pairs
    use_curriculum = config["curriculum_epochs"] > 0
    if use_curriculum:
        print(f"[Curriculum] Enabled: ramp over {config['curriculum_epochs']} epochs, "
              f"starting from {config['curriculum_start_frac']*100:.0f}% easiest pairs.")
        curriculum_sampler = CurriculumSampler(
            dataset=train_ds,
            batch_size=config["batch_size"],
            total_epochs=config["epochs"],
            curriculum_epochs=config["curriculum_epochs"],
            start_fraction=config["curriculum_start_frac"],
            end_fraction=1.0,
            drop_last=True,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=curriculum_sampler, 
            num_workers=config["num_workers"], collate_fn=collate_spair
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                                  shuffle=True, num_workers=config["num_workers"], 
                                  collate_fn=collate_spair, pin_memory=True)
        curriculum_sampler = None

    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, 
        num_workers=config["num_workers"], collate_fn=collate_spair, pin_memory=True
    )

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
        use_adaptive_win=not args.no_adaptive_win,
        aw_min_radius=args.aw_min_radius,
        aw_max_radius=args.aw_max_radius,
    ).to(device)

    # Only train LoRA params + projection head
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- Training ----
    scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
    best_pck = 0.0
    for epoch in range(1, args.epochs + 1):
        # Advance curriculum sampler
        if curriculum_sampler is not None:
            curriculum_sampler.set_epoch(epoch)
            active_frac = curriculum_sampler._active_fraction()
            print(f"[Curriculum] Epoch {epoch}: using {active_frac*100:.1f}% of training pairs")

        train_loss, train_pck, mean_ent = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config, scaler
        )
        val_pck = validate(model, val_loader, device, alpha=0.1)
        scheduler.step()

        print(f"Epoch {epoch:03d}  "
              f"loss={train_loss:.4f}  train_pck@0.1={train_pck:.4f}  "
              f"val_pck@0.1={val_pck:.4f}  entropy={mean_ent:.3f}")

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
            
            # Backup opzionale su Google Drive
            if config.get("backup_dir"):
                try:
                    os.makedirs(config["backup_dir"], exist_ok=True)
                    shutil.copy(ckpt_path, os.path.join(config["backup_dir"], "best.pth"))
                    print(f"  ✓ Backup copiato in → {config['backup_dir']}")
                except Exception as e:
                    print(f"  [Warning] Fallito il backup su Drive: {e}")

    print(f"\n[DONE] Best val PCK@0.1 = {best_pck:.4f}")


if __name__ == "__main__":
    main()
