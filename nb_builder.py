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
            "!pip install -r requirements.txt -q\n",
            "!pip install gradio -q\n",
            "!python dataloaders/download_spair.py --root ./data\n\n",
            "from utils.demo_utils import launch_stage_demo"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1. Stage 1: Backbone Analysis"]},
        {"cell_type": "code", "metadata": {}, "source": ["!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --results_file /content/drive/MyDrive/AML/Results/baseline.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('DINOv2 Baseline', layer_idx=11)"]},
        {"cell_type": "code", "metadata": {}, "source": ["for layer in [4, 8, 11]:\n    !python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --layer {layer} --results_file /content/drive/MyDrive/AML/Results/layer_{layer}.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('DINOv2 Layer Explorer', show_layer_slider=True)"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Stage 2: Fine-Tuning (LoRA)"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 2.1 Solo LoRA\n!python train.py --dataset_root ./data/SPair-71k --epochs 5 --curriculum_epochs 0 --num_workers 4 --exp_name lora_only --output_dir ./checkpoints/lora_only --backup_dir /content/drive/MyDrive/AML/Checkpoints/lora_only"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('LoRA Only', ckpt_name='lora_only')"]},
        {"cell_type": "code", "metadata": {}, "source": ["# 2.2 LoRA + Curriculum\n!python train.py --dataset_root ./data/SPair-71k --epochs 5 --curriculum_epochs 2 --num_workers 4 --exp_name lora_curriculum --output_dir ./checkpoints/lora_curriculum --backup_dir /content/drive/MyDrive/AML/Checkpoints/lora_curriculum"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('LoRA + Curriculum', ckpt_name='lora_curriculum')"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🎯 3. Stage 3: Raffinamento Adattivo"]},
        {"cell_type": "code", "metadata": {}, "source": ["CKPT = '/content/drive/MyDrive/AML/Checkpoints/lora_curriculum/lora_curriculum_best.pth'\n!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT} --results_file /content/drive/MyDrive/AML/Results/s3_with_aw.txt\n!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT} --no_adaptive_win --results_file /content/drive/MyDrive/AML/Results/s3_without_aw.txt"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🌟 4. Stage 4: Demo Finale"]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_stage_demo('Modello Finale', ckpt_name='lora_curriculum')"]}
    ]
}

with open("Project_Final_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Notebook rigenerato con --exp_name!")
