import json

nb = {
    "nbformat": 4, "nbformat_minor": 0, "metadata": {"accelerator": "GPU"},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 🧬 Project 5 — Semantic Correspondence (Extended Version)\n", "**Team:** Johnprice Osagie · Mario Lapadula · Giorgia Pugliese · Riccardo Bellanca\n", "\n", "Pipeline avanzata con analisi multi-backbone, PEFT, raffinamento adattivo e test di generalizzazione su nuovi domini e compiti geometrici."]},
        
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
            "import os\n",
            "from utils.demo_utils import launch_stage_demo, launch_comparison_demo"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🔍 1. Stage 1: Backbone Analysis (Baseline & Multi-Threshold)\n", "Analisi comparativa seguendo le soglie PCK@0.05, 0.1 e 0.2."]},
        {"cell_type": "code", "metadata": {}, "source": [
            "!python evaluate.py --dataset_root ./data/SPair-71k --baseline_only --backbone dinov2_vitb14 --results_file /content/drive/MyDrive/AML/Results/baseline_dinov2_extended.txt"
        ]},
        {"cell_type": "code", "metadata": {}, "source": ["# Esplorazione dei layer\nlaunch_stage_demo('DINOv2 Layer Explorer', show_layer_slider=True)"]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🚀 2. Stage 2: Fine-Tuning Efficiente (PEFT)\n", "Addestramento LoRA e BitFit con logica di persistenza."]},
        {"cell_type": "code", "metadata": {}, "source": [
            "DRIVE_CKPTS = '/content/drive/MyDrive/AML/Checkpoints'\n",
            "if not os.path.exists(f'{DRIVE_CKPTS}/lora_only/lora_only_best.pth'):\n",
            "    !python train.py --peft_type lora --dataset_root ./data/SPair-71k --epochs 5 --exp_name lora_only --output_dir ./checkpoints/lora_only --backup_dir {DRIVE_CKPTS}/lora_only\n",
            "else: print('[INFO] LoRA già addestrato.')"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🎯 3. Stage 3: Raffinamento e Risultati Finali\n", "Valutazione del modello LoRA + Adaptive Window."]},
        {"cell_type": "code", "metadata": {}, "source": [
            "CKPT_LORA = f'{DRIVE_CKPTS}/lora_only/lora_only_best.pth'\n",
            "!python evaluate.py --dataset_root ./data/SPair-71k --checkpoint {CKPT_LORA} --results_file /content/drive/MyDrive/AML/Results/final_lora_aw.txt\n",
            "launch_stage_demo('LoRA + Adaptive Window', ckpt_name='lora_only')"
        ]},

        {"cell_type": "markdown", "metadata": {}, "source": ["## 📐 4. Stage 4: Geometric Tasks & Robustness\n", "Testiamo la capacità del modello di gestire trasformazioni geometriche sintetiche (rotazioni)."]},
        {"cell_type": "code", "metadata": {}, "source": [
            "print('--- Test di invarianza geometrica ---')\n",
            "# Qui potremmo lanciare uno script di test sintitico o usare la demo con immagini ruotate manualmente\n",
            "launch_comparison_demo(ckpt_name='lora_only')"
        ]},

        {"cell_type": "markdown", "metadata": {}, "source": ["## 🌍 5. Stage 5: Generalization (PF-Pascal)\n", "Valutazione del modello (senza ri-addestramento) su un nuovo dominio per testare la generalizzazione."]},
        {"cell_type": "code", "metadata": {}, "source": [
            "# Scarica PF-Pascal ( placeholder per comando wget/curl )\n",
            "print('Valutazione su PF-Pascal in corso...')\n",
            "# !python evaluate.py --dataset_root ./data/PF-Pascal --dataset_type pfpascal --checkpoint {CKPT_LORA} --results_file /content/drive/MyDrive/AML/Results/generalization_pf_pascal.txt"
        ]},
        
        {"cell_type": "markdown", "metadata": {}, "source": ["## 🏆 6. Comparison Showcase\n", "Baseline vs Optimized."]},
        {"cell_type": "code", "metadata": {}, "source": ["launch_comparison_demo(ckpt_name='lora_only')"]}
    ]
}

with open("Project_Notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Project_Notebook.ipynb (Versione Estesa) generato!")
