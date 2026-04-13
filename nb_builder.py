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
            "from utils.demo_utils import launch_stage_demo, launch_comparison_demo"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1. Stage 1: Backbone Analysis"]},
        {"cell_type": "code", "metadata": {}, "source": ["!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --results_file /content/drive/MyDrive/AML/Results/baseline.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["# Esplorazione interattiva dei layer di DINOv2\nlaunch_stage_demo('DINOv2 Layer Explorer', show_layer_slider=True)"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Stage 2: Fine-Tuning Efficiente (PEFT)"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 2.1 Training LoRA\n!python train.py --peft_type lora --dataset_root ./data/SPair-71k --epochs 5 --exp_name lora_only --output_dir ./checkpoints/lora_only --backup_dir /content/drive/MyDrive/AML/Checkpoints/lora_only"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 2.2 Training BitFit (Ablation Study)\n!python train.py --peft_type bitfit --dataset_root ./data/SPair-71k --epochs 5 --exp_name bitfit_only --output_dir ./checkpoints/bitfit_only --backup_dir /content/drive/MyDrive/AML/Checkpoints/bitfit_only"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🎯 3. Stage 3: Raffinamento e Ablazione"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["### 🎯 3.1 Raffinamento LoRA (Modello di Punta)"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT_LORA = '/content/drive/MyDrive/AML/Checkpoints/lora_only/lora_only_best.pth'\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_LORA} --results_file /content/drive/MyDrive/AML/Results/s3_lora_with_aw.txt\n",
            "launch_stage_demo('LoRA + Adaptive Window', ckpt_name='lora_only')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["### 🎯 3.2 Ablazione BitFit (AW vs No-AW)"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT_BITFIT = '/content/drive/MyDrive/AML/Checkpoints/bitfit_only/bitfit_only_best.pth'\n",
            "print('--- Valutazione CON Adaptive Window ---')\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_BITFIT} --results_file /content/drive/MyDrive/AML/Results/s3_bitfit_with_aw.txt\n",
            "print('\\n--- Valutazione SENZA Adaptive Window ---')\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_BITFIT} --no_adaptive_win --results_file /content/drive/MyDrive/AML/Results/s3_bitfit_no_aw.txt\n",
            "launch_stage_demo('BitFit (Comparison Demo)', ckpt_name='bitfit_only')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🌟 4. Stage 4: Demo Finale e Segment-Aware"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('Modello Finale (LoRA) + SAM Integration', ckpt_name='lora_only')"]},

        {"cell_type": "markdown", "metadata": {}, "source": ["## ⚖️ 5. Stage 5: Comparison Showcase (Baseline vs LoRA+AW)"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["Questa demo permette di confrontare istantaneamente il modello base con la nostra versione ottimizzata."]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_comparison_demo(ckpt_name='lora_only')"]}
    ]
}

with open("Project_Final_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Notebook aggiornato con lo Stage 5: Comparison Demo!")
