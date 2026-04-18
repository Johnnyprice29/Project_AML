import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.spair import SPairDataset, collate_spair
from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel
from utils.metrics import pck

def main():
    parser = argparse.ArgumentParser(description="Temperature Calibration Ablation")
    parser.add_argument("--dataset_root", type=str, default="./data/SPair-71k")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best LoRA checkpoint")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.5], help="List of temperatures to test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--results_file", type=str, default="", help="Path to save text results")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("[INFO] Loading validation data...")
    val_ds = SPairDataset(args.dataset_root, split='val', img_size=224)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=collate_spair)

    print(f"[INFO] Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Baseline instancing once to avoid re-downloading/loading overhead
    backbone = DINOv2Extractor(model_name='dinov2_vitb14', freeze=True)
    backbone.model = apply_lora_to_dinov2(backbone.model, rank=16)

    results = {}

    for T in args.temperatures:
        print(f"\n======================================")
        print(f"--- Testing Temperature: T={T} ---")
        print(f"======================================")
        
        # Instantiate model with the specific temperature
        model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True, temperature=T).to(device)
        model.load_state_dict(ckpt['model_state_dict'])    
        model.eval()
        
        scores = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Eval T={T}"):
                src_img, trg_img = batch["src_img"].to(device), batch["trg_img"].to(device)
                src_kps, trg_kps = batch["src_kps"].to(device), batch["trg_kps"].to(device)
                mask = batch["kps_mask"].to(device)

                out = model(src_img, trg_img, src_kps=src_kps)
                
                s = pck(out["pred_kps"].detach(), trg_kps, img_size=224, alpha=args.alpha, mask=mask)
                scores.append(s.item())
                
        mean_pck = np.mean(scores) * 100
        results[T] = mean_pck
        print(f">> PCK@{args.alpha} @ T={T}  =>  {mean_pck:.2f}%\n")

    print("\n--- Final Calibration Summary ---")
    summary_lines = ["--- Temperature Ablation Results ---"]
    for T, score in results.items():
        weight = "(Default/Sweet Spot)" if T == 0.05 else ""
        line = f"Temperature {T}: {score:.2f}% {weight}"
        print(line)
        summary_lines.append(line)
        
    if args.results_file:
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        with open(args.results_file, "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        print(f"[INFO] Results saved to {args.results_file}")

if __name__ == '__main__':
    main()
