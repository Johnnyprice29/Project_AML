import json
import os

output_path = r"G:\My Drive\Magistrale\2year2semester\AML\Project_AML\Project_Notebook.ipynb"

nb = {
    "nbformat": 4, "nbformat_minor": 0, "metadata": {"accelerator": "GPU"},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 🧬 Project 5 — Semantic Correspondence\n", "**Team:** Johnprice Osagie · Mario Lapadula · Giorgia Pugliese · Riccardo Bellanca"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 📦 0. Setup"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n\n",
            "!git clone -b main https://github.com/Johnnyprice29/Project_AML.git /content/Project_AML\n",
            "%cd /content/Project_AML\n",
            "!git fetch origin && git reset --hard origin/main\n",
            "!pip install -r requirements.txt -q\n",
            "!pip install gradio -q\n",
            "!python dataloaders/download_spair.py --root ./data\n\n",
            "import os\n",
            "from utils.demo_utils import launch_stage_demo, launch_comparison_demo, launch_robustness_demo"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1.1 Evaluate Backbones (Baseline Analysis)\n", "Analisi comparativa dei modelli pre-addestrati ViT-B (Non-distillati)."]},
        {"cell_type": "code", "metadata": {}, "source": ["# 1.1.1 DINOv2 Baseline (ViT-B/14)\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone dinov2_vitb14 --results_file /content/drive/MyDrive/AML/Results/baseline_dinov2.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 1.1.2 DINOv3 Baseline (ViT-B/16)\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone dinov3 --results_file /content/drive/MyDrive/AML/Results/baseline_dinov3.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 1.1.3 SAM Baseline (ViT-B)\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone sam_vitb --results_file /content/drive/MyDrive/AML/Results/baseline_sam.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔦 1.2 Layer-wise Explorer for DINOv2"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('DINOv2 Layer Explorer', show_layer_slider=True)"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Training DINOv2 (LoRA, LoRA+Curriculum, BitFit)"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "DRIVE_CKPTS = '/content/drive/MyDrive/AML/Checkpoints'\n\n",
            "# 2.1 Training LoRA\n",
            "if not os.path.exists(f'{DRIVE_CKPTS}/lora_only/lora_only_best.pth'):\n",
            "    !python train.py --peft_type lora --dataset_root ./data/SPair-71k --epochs 5 --exp_name lora_only --output_dir ./checkpoints/lora_only --backup_dir {DRIVE_CKPTS}/lora_only\n",
            "else: print('[OK] LoRA già presente.')"
        ]},
        {"cell_type": "code", "metadata": {}, "source": [
            "# 2.2 Training LoRA + Curriculum\n",
            "if not os.path.exists(f'{DRIVE_CKPTS}/lora_curriculum/lora_curriculum_best.pth'):\n",
            "    !python train.py --peft_type lora --dataset_root ./data/SPair-71k --epochs 5 --curriculum_epochs 3 --exp_name lora_curriculum --output_dir ./checkpoints/lora_curriculum --backup_dir {DRIVE_CKPTS}/lora_curriculum\n",
            "else: print('[OK] LoRA Curriculum già presente.')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🎯 3. Raffinamento (Ablation Study: Adaptive Window)"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT_LORA = f'{DRIVE_CKPTS}/lora_only/lora_only_best.pth'\n",
            "print('--- LoRA + Adaptive Window ---')\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_LORA} --results_file /content/drive/MyDrive/AML/Results/lora_with_aw.txt"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🌍 4.1 Test su altri dataset (PF-Pascal)"]},
        {"cell_type": "code", "metadata": {}, "source": ["print('Test di generalizzazione su PF-Pascal...')\n# !python evaluate.py --dataset_root ./data/PF-Pascal --dataset_type pfpascal --checkpoint {CKPT_LORA} --results_file /content/drive/MyDrive/AML/Results/gen_pfpascal.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 📐 4.2 Test con Robustezza Geometrica (Quantitative)"]},
        {"cell_type": "code", "metadata": {}, "source": ["print('Valutazione quantitativa robustezza...')\n# !python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_LORA} --geometric_test rotate --results_file /content/drive/MyDrive/AML/Results/robust_geom.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## ⚖️ 5. Demo Comparison & Robustness Demo"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 5.1 Demo Comparison (Baseline vs LoRA+AW)\nlaunch_comparison_demo(ckpt_name='lora_only')"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 5.2 Demo Robustezza Geometrica (Rotazioni)\nlaunch_robustness_demo(ckpt_name='lora_only')"]}
    ]
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"Project_Notebook.ipynb (Versione Robusta Finale) generato in {output_path}")
