import json
import os

SETUP_CODE = r'''# ── Setup Logica Demo (Stage 4) ─────────────────────────────────
import torch, os, gradio as gr
from PIL import Image, ImageDraw
import torchvision.transforms as T
from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def launch_stage_demo(title, ckpt_name=None, layer_idx=-1, show_layer_slider=False):
    def get_prediction(src_img, trg_img, x, y, current_layer):
        backbone = DINOv2Extractor(model_name='dinov2_vitb14', layer=int(current_layer), freeze=True)
        if ckpt_name:
            path = f'/content/drive/MyDrive/AML/Checkpoints/{ckpt_name}/best.pth'
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=device)
                backbone.model = apply_lora_to_dinov2(backbone.model, rank=ckpt['args'].get('lora_rank', 16))
                model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
                model.load_state_dict(ckpt['model_state_dict'])
            else: model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
        else: model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
        model.eval()
        s_t, t_t = transform(src_img).unsqueeze(0).to(device), transform(trg_img).unsqueeze(0).to(device)
        src_kp = torch.tensor([[[x*(224/src_img.width), y*(224/src_img.height)]]], device=device).float()
        with torch.no_grad(): out = model(s_t, t_t, src_kps=src_kp)
        pkp = out['pred_kps'][0,0].cpu().numpy()
        return pkp[0]*(trg_img.width/224), pkp[1]*(trg_img.height/224)

    def gradio_fn(s_img, t_img, lyr, evt: gr.SelectData):
        tx, ty = get_prediction(s_img, t_img, evt.index[0], evt.index[1], lyr)
        so, to = s_img.copy(), t_img.copy()
        ImageDraw.Draw(so).ellipse([evt.index[0]-6, evt.index[1]-6, evt.index[0]+6, evt.index[1]+6], fill='red', outline='white')
        ImageDraw.Draw(to).ellipse([tx-6, ty-6, tx+6, ty+6], fill='green', outline='white')
        return so, to

    with gr.Blocks() as d:
        gr.Markdown(f"### {title}")
        with gr.Row(): lyr_ctrl = gr.Slider(0, 11, value=layer_idx if layer_idx != -1 else 11, step=1, label='Layer', visible=show_layer_slider)
        with gr.Row(): si, ti = gr.Image(type='pil', label='Source'), gr.Image(type='pil', label='Target')
        with gr.Row(): so, to = gr.Image(type='pil'), gr.Image(type='pil')
        si.select(gradio_fn, [si, ti, lyr_ctrl], [so, to])
    d.launch(share=True, inline=False)
'''

nb = {
    "nbformat": 4, "nbformat_minor": 0, "metadata": {"accelerator": "GPU"},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 🧬 Master Pipeline Project 5\n", "**Team:** Johnprice Osagie · Mario Lapadula · Giorgia Pugliese · Riccardo Bellanca"]},
        {"cell_type": "code", "metadata": {}, "source": ["from google.colab import drive\ndrive.mount('/content/drive')\n\n!git clone -b osagie5 https://github.com/Johnnyprice29/Project_AML.git /content/Project_AML\n%cd /content/Project_AML\n!pip install -r requirements.txt -q\n!pip install gradio -q\n!python dataloaders/download_spair.py --root ./data\n\n" + SETUP_CODE.strip()]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1. Stage 1: Backbone Analysis"]},
        {"cell_type": "code", "metadata": {}, "source": ["!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --results_file /content/drive/MyDrive/AML/Results/baseline.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('DINOv2 Baseline', layer_idx=11)"]},
        {"cell_type": "code", "metadata": {}, "source": ["for layer in [4, 8, 11]:\n    !python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --layer {layer} --results_file /content/drive/MyDrive/AML/Results/layer_{layer}.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('DINOv2 Layer Explorer', show_layer_slider=True)"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Stage 2: Fine-Tuning Efficiente (LoRA)"]},
        {"cell_type": "code", "metadata": {}, "source": ["!python train.py --dataset_root ./data/SPair-71k --epochs 5 --curriculum_epochs 0 --num_workers 4 --output_dir ./checkpoints/lora_only --backup_dir /content/drive/MyDrive/AML/Checkpoints/lora_only"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('LoRA Only', ckpt_name='lora_only')"]},
        {"cell_type": "code", "metadata": {}, "source": ["!python train.py --dataset_root ./data/SPair-71k --epochs 5 --curriculum_epochs 2 --num_workers 4 --output_dir ./checkpoints/lora_curriculum --backup_dir /content/drive/MyDrive/AML/Checkpoints/lora_curriculum"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🎯 3. Stage 3: Raffinamento Adattivo"]},
        {"cell_type": "code", "metadata": {}, "source": ["CKPT = '/content/drive/MyDrive/AML/Checkpoints/lora_curriculum/best.pth'\n!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT} --results_file /content/drive/MyDrive/AML/Results/s3_with.txt\n!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT} --no_adaptive_win --results_file /content/drive/MyDrive/AML/Results/s3_without.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🌟 4. Stage 4: Valutazione ed Estensioni"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('Modello Finale: LoRA + Curriculum', ckpt_name='lora_curriculum')"]}
    ]
}

with open("Project_Final_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Notebook con --dataset_root corretto creato!")
