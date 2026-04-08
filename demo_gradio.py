import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import torchvision.transforms as T

from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--share", action="store_true", help="Create a public link for the demo")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---- Load Model ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    
    backbone = DINOv2Extractor(
        model_name=saved_args.get("backbone", "dinov2_vitb14"),
        freeze=True
    )
    # Re-apply LoRA
    backbone.model = apply_lora_to_dinov2(
        backbone.model,
        rank=saved_args.get("lora_rank", 16),
        lora_alpha=saved_args.get("lora_alpha", 32)
    )
    
    model = SemanticCorrespondenceModel(
        backbone=backbone,
        proj_dim=saved_args.get("proj_dim", 256),
        use_adaptive_win=True,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("[INFO] Model loaded successfully.")

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def predict(src_img, trg_img, src_x, src_y):
        # src_x, src_y are coords in the original image size
        # 1. Preprocess images
        src_t = transform(src_img).unsqueeze(0).to(device)
        trg_t = transform(trg_img).unsqueeze(0).to(device)
        
        # 2. Scale clicked point to model img_size
        scale_x = args.img_size / src_img.width
        scale_y = args.img_size / src_img.height
        src_kp = torch.tensor([[[src_x * scale_x, src_y * scale_y]]], device=device).float() # (1, 1, 2)
        
        # 3. Predict
        out = model(src_t, trg_t, src_kps=src_kp)
        pred_kp = out["pred_kps"][0, 0].cpu().numpy() # (2,) [x, y]
        
        # 4. Scale back to target image size
        unscale_x = trg_img.width / args.img_size
        unscale_y = trg_img.height / args.img_size
        pred_x, pred_y = pred_kp[0] * unscale_x, pred_kp[1] * unscale_y
        
        return pred_x, pred_y

    def gradio_interface(src_img, trg_img, evt: gr.SelectData):
        # Click on Source Image (the left one)
        # However, Gradio's SelectData doesn't distinguish between images easily
        # if they are separate components. We'll assume the click is on the Source.
        src_x, src_y = evt.index[0], evt.index[1]
        
        trg_x, trg_y = predict(src_img, trg_img, src_x, src_y)
        
        # Draw on source
        src_draw = src_img.copy()
        d1 = ImageDraw.Draw(src_draw)
        r = 5
        d1.ellipse([src_x-r, src_y-r, src_x+r, src_y+r], fill="red", outline="white")
        
        # Draw on target
        trg_draw = trg_img.copy()
        d2 = ImageDraw.Draw(trg_draw)
        d2.ellipse([trg_x-r, trg_y-r, trg_x+r, trg_y+r], fill="green", outline="white")
        
        return src_draw, trg_draw

    with gr.Blocks(title="Semantic Correspondence Demo") as demo:
        gr.Markdown("# 🐈 Semantic Correspondence with DINOv2 + LoRA")
        gr.Markdown("Seleziona due immagini e clicca su un punto dell'immagine **SORGENTE** (sinistra) per trovare il punto corrispondente sulla **TARGET** (destra).")
        
        with gr.Row():
            src_input = gr.Image(type="pil", label="Source Image")
            trg_input = gr.Image(type="pil", label="Target Image")
            
        with gr.Row():
            src_output = gr.Image(type="pil", label="Source with Click")
            trg_output = gr.Image(type="pil", label="Target Prediction")
            
        # Trigger on clicking source input
        src_input.select(gradio_interface, inputs=[src_input, trg_input], outputs=[src_output, trg_output])
        
    demo.launch(share=args.share, inline=False)

if __name__ == "__main__":
    main()
