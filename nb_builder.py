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
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1. Stage 1: Backbone Analysis\n", "In questa fase analizziamo la capacità 'zero-shot' di DINOv2. Confrontiamo diversi layer per capire dove risiede l'informazione semantica più utile per il matching."]},
        {"cell_type": "code", "metadata": {}, "source": ["!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --results_file /content/drive/MyDrive/AML/Results/baseline.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["for layer in [4, 8, 11]:\n    !python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --layer {layer} --results_file /content/drive/MyDrive/AML/Results/layer_{layer}.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["# Visualizzazione interattiva: come cambia il matching in base al layer?\nlaunch_stage_demo('DINOv2 Layer Explorer', show_layer_slider=True)"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Stage 2: Fine-Tuning Efficiente (PEFT)\n", "Confrontiamo due tecniche di adattamento: **LoRA** (Low-Rank Adaptation) e **BitFit** (Bias-only Fine-Tuning)."]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["### 2.1 Solo LoRA"]},
        {"cell_type": "code", "metadata": {}, "source": ["!python train.py --peft_type lora --dataset_root ./data/SPair-71k --epochs 5 --curriculum_epochs 0 --num_workers 4 --exp_name lora_only --output_dir ./checkpoints/lora_only --backup_dir /content/drive/MyDrive/AML/Checkpoints/lora_only"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["### 2.2 LoRA + Curriculum Learning\n", "**Nota Scientifica:** Testiamo se facilitare l'apprendimento con coppie semplici aiuta la convergenza."]},
        {"cell_type": "code", "metadata": {}, "source": ["!python train.py --peft_type lora --dataset_root ./data/SPair-71k --epochs 5 --curriculum_epochs 2 --num_workers 4 --exp_name lora_curriculum --output_dir ./checkpoints/lora_curriculum --backup_dir /content/drive/MyDrive/AML/Checkpoints/lora_curriculum"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["### 2.3 BitFit (Ablation Study)\n", "Metodo ultra-efficiente: addestriamo solo i bias del modello. Vediamo se è sufficiente per il task."]},
        {"cell_type": "code", "metadata": {}, "source": ["!python train.py --peft_type bitfit --dataset_root ./data/SPair-71k --epochs 5 --curriculum_epochs 0 --num_workers 4 --exp_name bitfit_only --output_dir ./checkpoints/bitfit_only --backup_dir /content/drive/MyDrive/AML/Checkpoints/bitfit_only"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["### 📊 Confronto PEFT (Gradio)\n", "Mettiamo a confronto visivo il modello LoRA e quello BitFit."]},
        {"cell_type": "code", "metadata": {}, "source": [
            "print('Caricamento demo comparativa...')\n",
            "launch_stage_demo('Confronto LoRA vs BitFit', ckpt_name='lora_only') # Puoi cambiare in bitfit_only per ispezionare l'altro"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🎯 3. Stage 3: Raffinamento Adattivo\n", "In questa fase applichiamo l'**Adaptive Window Soft-Argmax** per ottenere precisione sub-pixel."]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT = '/content/drive/MyDrive/AML/Checkpoints/lora_only/lora_only_best.pth'\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT} --num_workers 4 --results_file /content/drive/MyDrive/AML/Results/s3_with_aw.txt\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT} --no_adaptive_win --num_workers 4 --results_file /content/drive/MyDrive/AML/Results/s3_without_aw.txt"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🌟 4. Stage 4: Valutazione Finale ed Estensioni\n", "Valutazione finale su Test Set e Demo interattiva con **SAM (Segment Anything)** attivo."]},
        {"cell_type": "code", "metadata": {}, "source": ["# Valutazione finale sul test set\nCKPT_FINAL = '/content/drive/MyDrive/AML/Checkpoints/lora_only/lora_only_best.pth'\n!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_FINAL} --alpha 0.1 --results_file /content/drive/MyDrive/AML/Results/final_test_results.txt"]},
        {"cell_type": "code", "metadata": {}, "source": ["# Demo Finale: Include Adaptive Window e Segment-Aware Correspondence\nlaunch_stage_demo('Modello Finale Completo', ckpt_name='lora_only')"]}
    ]
}

with open("Project_Final_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Notebook rigenerato con Sezioni Scientifiche (Layer Analysis + PEFT Comparison)!")
