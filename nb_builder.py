import json

nb = {
    "nbformat": 4, "nbformat_minor": 0, "metadata": {"accelerator": "GPU"},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 🧬 Project 5 — Semantic Correspondence\n", "**Team:** Johnprice Osagie · Mario Lapadula · Giorgia Pugliese · Riccardo Bellanca"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 📦 0. Setup"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n\n",
            "!git clone -b osagie5 https://github.com/Johnnyprice29/Project_AML.git /content/Project_AML\n",
            "%cd /content/Project_AML\n",
            "!git fetch origin && git reset --hard origin/osagie5\n",
            "!pip install -r requirements.txt -q\n",
            "!pip install gradio -q\n",
            "!python dataloaders/download_spair.py --root ./data\n\n",
            "from utils.demo_utils import launch_stage_demo"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1. Stage 1: Backbone Analysis"]},
        {"cell_type": "code", "metadata": {}, "source": ["!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --results_file /content/drive/MyDrive/AML/Results/baseline.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Stage 2: Fine-Tuning Efficiente (PEFT)"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 2.2 BitFit (Ablation Study)\n!python train.py --peft_type bitfit --dataset_root ./data/SPair-71k --epochs 5 --curriculum_epochs 0 --exp_name bitfit_only --output_dir ./checkpoints/bitfit_only --backup_dir /content/drive/MyDrive/AML/Checkpoints/bitfit_only"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🎯 3. Stage 3: Raffinamento e Confronto Finale"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["### 🎯 3.2 Analisi BitFit (Con vs Senza Adaptive Window)"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT_BITFIT = '/content/drive/MyDrive/AML/Checkpoints/bitfit_only/bitfit_only_best.pth'\n",
            "print('--- Valutazione CON Adaptive Window ---')\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_BITFIT} --results_file /content/drive/MyDrive/AML/Results/s3_bitfit_with_aw.txt\n",
            "print('\\n--- Valutazione SENZA Adaptive Window ---')\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_BITFIT} --no_adaptive_win --results_file /content/drive/MyDrive/AML/Results/s3_bitfit_no_aw.txt"
        ]}
    ]
}

with open("Project_Final_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Notebook aggiornato con BitFit ablation (AW vs No-AW)!")
