import json
import os

# Cambiamo nome per forzare Google Drive a vedere un file nuovo
output_path = r"G:\My Drive\Magistrale\2year2semester\AML\Project_AML\Project_Final_Notebook.ipynb"

nb = {
    "nbformat": 4, "nbformat_minor": 0, "metadata": {"accelerator": "GPU"},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 🧬 Project 5 — Semantic Correspondence (OFFICIAL FINAL)\n", "**Team:** Johnprice Osagie · Mario Lapadula · Giorgia Pugliese · Riccardo Bellanca"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 📦 0. Setup"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n\n",
            "!git clone -b main https://github.com/Johnnyprice29/Project_AML.git /content/Project_AML\n",
            "%cd /content/Project_AML\n",
            "!git fetch origin && git reset --hard origin/main\n",
            "!pip install -r requirements.txt -q\n",
            "!pip install gradio -q\n",
            "!python dataloaders/download_spair.py --root ./data\n",
            "# Per PF-Pascal (Stage 4), caricare i dati in ./data/PF-Pascal\n",
            "!mkdir -p ./data/PF-Pascal\n\n",
            "import os, torch, gc\n",
            "DRIVE_CKPTS = '/content/drive/MyDrive/AML/Checkpoints'\n",
            "def clear_gpu():\n",
            "    gc.collect()\n",
            "    torch.cuda.empty_cache()\n",
            "    print('[INFO] GPU Cleared.')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1. Evaluation Baseline"]},
        {"cell_type": "code", "metadata": {}, "source": ["clear_gpu()\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone dinov2_vitb14 --results_file /content/drive/MyDrive/AML/Results/baseline_dinov2.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["clear_gpu()\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone dinov3 --results_file /content/drive/MyDrive/AML/Results/baseline_dinov3.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["clear_gpu()\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone sam_vitb --batch_size 1 --results_file /content/drive/MyDrive/AML/Results/baseline_sam.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Training Stage"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT_PATH = f'{DRIVE_CKPTS}/lora_only/lora_only_best.pth'\n",
            "if not os.path.exists(f'/content/drive/MyDrive/AML/Checkpoints/lora_only/lora_only_best.pth'):\n",
            "    !python train.py --peft_type lora --dataset_root ./data/SPair-71k --epochs 5 --exp_name lora_only --output_dir ./checkpoints/lora_only --backup_dir \"/content/drive/MyDrive/AML/Checkpoints/lora_only\"\n",
            "else: print(f'[OK] Checkpoint già presente su Drive.')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🌍 4. Generalizzazione e Robustezza"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT_LORA = f'{DRIVE_CKPTS}/lora_only/lora_only_best.pth'\n",
            "!python evaluate.py --dataset_root ./data/PF-Pascal --dataset_type pfpascal --checkpoint \"$CKPT_LORA\" --results_file /content/drive/MyDrive/AML/Results/gen_pfpascal.txt"
        ]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT_LORA = f'{DRIVE_CKPTS}/lora_only/lora_only_best.pth'\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint \"$CKPT_LORA\" --results_file /content/drive/MyDrive/AML/Results/robustness_rot.txt"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## ⚖️ 5. Demo"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_comparison_demo(ckpt_name='lora_only')"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_robustness_demo(ckpt_name='lora_only')"]}
    ]
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"Creato nuovo file: {output_path}")
