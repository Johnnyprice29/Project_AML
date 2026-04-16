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
            "import os, torch, gc\n",
            "from utils.demo_utils import launch_stage_demo, launch_comparison_demo, launch_robustness_demo\n\n",
            "def clear_gpu():\n",
            "    gc.collect()\n",
            "    torch.cuda.empty_cache()\n",
            "    print('[INFO] GPU Memory cleared.')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1.1 Evaluate Backbones (Baseline Analysis)"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 1.1.1 DINOv2 Baseline\nclear_gpu()\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone dinov2_vitb14 --results_file /content/drive/MyDrive/AML/Results/baseline_dinov2.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 1.1.2 DINOv2 with Registers (The real DINOv3 alternative)\nclear_gpu()\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone dinov3 --results_file /content/drive/MyDrive/AML/Results/baseline_dinov3.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 1.1.3 SAM Baseline (ViT-B) - Aggressive Memory Management\nclear_gpu()\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone sam_vitb --batch_size 1 --results_file /content/drive/MyDrive/AML/Results/baseline_sam.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔦 1.2 Layer-wise Explorer for DINOv2"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('DINOv2 Layer Explorer', show_layer_slider=True)"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Training DINOv2 (LoRA, LoRA+Curriculum)"]},
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
            "else: print('[OK] Modello Curriculum già presente.')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🎯 3. Raffinamento (Ablation Study: Adaptive Window)"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT_LORA = f'{DRIVE_CKPTS}/lora_only/lora_only_best.pth'\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_LORA} --results_file /content/drive/MyDrive/AML/Results/lora_final_aw.txt"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🌍 4. Generalizzazione e Robustezza"]},
        {"cell_type": "code", "metadata": {}, "source": ["# !python evaluate.py --dataset_root ./data/PF-Pascal --dataset_type pfpascal --checkpoint {CKPT_LORA} --results_file /content/drive/MyDrive/AML/Results/gen_pascal.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## ⚖️ 5. Showcase Finali"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_comparison_demo(ckpt_name='lora_only')"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_robustness_demo(ckpt_name='lora_only')"]}
    ]
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"Project_Notebook.ipynb (Versione Ultra-Robusta) generato in {output_path}")
