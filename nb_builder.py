import json
import os

output_path = r"G:\My Drive\Magistrale\2year2semester\AML\Project_AML\Project_Final_v2.ipynb"

nb = {
    "nbformat": 4, "nbformat_minor": 0, "metadata": {"accelerator": "GPU"},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 🧬 Project 5 — Semantic Correspondence (OFFICIAL FINAL V2)\n", "**Team:** Johnprice Osagie · Mario Lapadula · Giorgia Pugliese · Riccardo Bellanca"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 📦 0. Setup"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n\n",
            "!git clone -b main https://github.com/Johnnyprice29/Project_AML.git /content/Project_AML\n",
            "%cd /content/Project_AML\n",
            "!git fetch origin && git reset --hard origin/main\n",
            "!pip install -r requirements.txt -q\n",
            "!pip install gradio -q\n\n",
            "# Scaricamento Dataset\n",
            "!python dataloaders/download_spair.py --root ./data\n",
            "!python dataloaders/download_pfpascal.py --root ./data\n\n",
            "import os, torch, gc\n",
            "DRIVE_CKPTS = '/content/drive/MyDrive/AML/Checkpoints'\n",
            "def clear_gpu():\n",
            "    gc.collect()\n",
            "    torch.cuda.empty_cache()\n",
            "    print('[INFO] GPU Cleared.')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1. Baseline Evaluation"]},
        {"cell_type": "code", "metadata": {}, "source": ["clear_gpu()\n!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone dinov2_vitb14 --results_file /content/drive/MyDrive/AML/Results/baseline_dinov2.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Training (Comparison PEFT)"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "DRIVE_CKPTS = '/content/drive/MyDrive/AML/Checkpoints'\n",
            "# 2.1 Training LoRA\n",
            "if not os.path.exists(f'{DRIVE_CKPTS}/lora_only/lora_only_best.pth'):\n",
            "    !python train.py --peft_type lora --dataset_root ./data/SPair-71k --epochs 5 --exp_name lora_only --output_dir ./checkpoints/lora_only --backup_dir \"$DRIVE_CKPTS/lora_only\"\n",
            "else: print('[OK] LoRA trovato.')"
        ]},
        {"cell_type": "code", "metadata": {}, "source": [
            "# 2.2 Training BitFit\n",
            "if not os.path.exists(f'{DRIVE_CKPTS}/bitfit_only/bitfit_only_best.pth'):\n",
            "    !python train.py --peft_type bitfit --dataset_root ./data/SPair-71k --epochs 5 --exp_name bitfit_only --output_dir ./checkpoints/bitfit_only --backup_dir \"$DRIVE_CKPTS/bitfit_only\"\n",
            "else: print('[OK] BitFit trovato.')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🌍 4. Generalization: PF-Pascal"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "import os\n",
            "# Identifichiamo la cartella corretta di PF-Pascal\n",
            "base_pascal = './data/PF-Pascal'\n",
            "contents = os.listdir(base_pascal)\n",
            "PASCAL_ROOT = os.path.join(base_pascal, contents[0]) if contents else base_pascal\n",
            "print(f'[INFO] Utilizzando PASCAL_ROOT: {PASCAL_ROOT}')\n\n",
            "print('--- PF-Pascal: LoRA ---')\n",
            "!python evaluate.py --dataset_root \"$PASCAL_ROOT\" --dataset_type pfpascal --checkpoint \"$DRIVE_CKPTS/lora_only/lora_only_best.pth\" --results_file /content/drive/MyDrive/AML/Results/gen_pascal_lora.txt\n",
            "print('--- PF-Pascal: BitFit ---')\n",
            "!python evaluate.py --dataset_root \"$PASCAL_ROOT\" --dataset_type pfpascal --checkpoint \"$DRIVE_CKPTS/bitfit_only/bitfit_only_best.pth\" --results_file /content/drive/MyDrive/AML/Results/gen_pascal_bitfit.txt"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🎯 3. Ablation Study: Adaptive Window (AW)"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "print('--- LoRA ---')\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint \"$DRIVE_CKPTS/lora_only/lora_only_best.pth\" --results_file /content/drive/MyDrive/AML/Results/lora_aw.txt\n",
            "print('--- BitFit ---')\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint \"$DRIVE_CKPTS/bitfit_only/bitfit_only_best.pth\" --results_file /content/drive/MyDrive/AML/Results/bitfit_aw.txt"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## ⚖️ 5. Visual Showcase"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_comparison_demo(ckpt_name='lora_only')"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_robustness_demo(ckpt_name='lora_only')"]}
    ]
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Project_Final_v2.ipynb sincronizzato con rilevamento automatico PASCAL_ROOT.")
