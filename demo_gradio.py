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
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory containing .pth files")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(ckpt_path):
        if not ckpt_path or ckpt_path == "Baseline (No LoRA)":
            backbone = DINOv2Extractor(model_name="dinov2_vitb14", freeze=True)
            model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
            model.eval()
            return model

        ckpt = torch.load(ckpt_path, map_location=device)
        saved_args = ckpt.get("args", {})
        backbone = DINOv2Extractor(model_name=saved_args.get("backbone", "dinov2_vitb14"), freeze=True)
        backbone.model = apply_lora_to_dinov2(backbone.model, rank=saved_args.get("lora_rank", 16))
        model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cache per i modelli caricati
    models_cache = {}

    def predict(src_img, trg_img, src_x, src_y, ckpt_name, use_aw):
        if ckpt_name not in models_cache:
            path = os.path.join(args.checkpoint_dir, ckpt_name) if ckpt_name != "Baseline (No LoRA)" else None
            models_cache[ckpt_name] = load_model(path)
        
        m = models_cache[ckpt_name]
        m.use_adaptive_win = use_aw

        src_t = transform(src_img).unsqueeze(0).to(device)
        trg_t = transform(trg_img).unsqueeze(0).to(device)
        
        scale_x, scale_y = args.img_size / src_img.width, args.img_size / src_img.height
        src_kp = torch.tensor([[[src_x * scale_x, src_y * scale_y]]], device=device).float()
        
        out = m(src_t, trg_t, src_kps=src_kp)
        pred_kp = out["pred_kps"][0, 0].cpu().numpy()
        
        unscale_x, unscale_y = trg_img.width / args.img_size, trg_img.height / args.img_size
        return pred_kp[0] * unscale_x, pred_kp[1] * unscale_y

    def gradio_fn(src_img, trg_img, ckpt_name, use_aw, evt: gr.SelectData):
        src_x, src_y = evt.index[0], evt.index[1]
        trg_x, trg_y = predict(src_img, trg_img, src_x, src_y, ckpt_name, use_aw)
        
        s_draw, t_draw = src_img.copy(), trg_img.copy()
        r = 6
        ImageDraw.Draw(s_draw).ellipse([src_x-r, src_y-r, src_x+r, src_y+r], fill="red", outline="white")
        ImageDraw.Draw(t_draw).ellipse([trg_x-r, trg_y-r, trg_x+r, trg_y+r], fill="green", outline="white")
        return s_draw, t_draw

    # Cerchiamo i checkpoint disponibili
    available_ckpts = ["Baseline (No LoRA)"]
    if os.path.exists(args.checkpoint_dir):
        available_ckpts += [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")]

    with gr.Blocks(title="AML Project 5 Demo") as demo:
        gr.Markdown("# 🐈 Multi-Stage Semantic Correspondence")
        with gr.Row():
            ckpt_dropdown = gr.Dropdown(choices=available_ckpts, value=available_ckpts[0], label="Seleziona Versione Modello (Checkpoint)")
            aw_toggle = gr.Checkbox(value=True, label="Usa Adaptive Window (Stage 3)")
        
        with gr.Row():
            src_in = gr.Image(type="pil", label="Source (Clicca qui)")
            trg_in = gr.Image(type="pil", label="Target Image")
        with gr.Row():
            src_out = gr.Image(type="pil", label="Source Selection")
            trg_out = gr.Image(type="pil", label="Target Prediction")
            
        src_in.select(gradio_fn, inputs=[src_in, trg_in, ckpt_dropdown, aw_toggle], outputs=[src_out, trg_out])

    demo.launch(share=args.share, inline=False)

if __name__ == "__main__":
    main()
