# Project 5 — Semantic Correspondence with DINOv2 + LoRA

> **Advanced Machine Learning — A.Y. 2025/2026**  
> Politecnico di Torino

---

## 👥 Team

| Name | Student ID |
|------|-----------|
| Johnprice Osagie | s344613 |
| Mario Lapadula | s338300 |
| Giorgia Pugliese | s344683 |
| Riccardo Bellanca | s346229 |

---

## 🎯 Task Description

**Semantic Correspondence** is the problem of matching pixels across images that depict the same *semantic concept* (e.g., the eye of a Chihuahua → the eye of a Husky), even when the two images differ in appearance, viewpoint, or background.

Given an image pair $(I_A, I_B)$ and a set of source keypoints in $I_A$, the goal is to predict the corresponding keypoints in $I_B$.

---

## 🗂️ Repository Structure

```
Project_AML/
├── data/                        # Dataset download scripts & loaders
│   ├── download_spair.py        # Download SPair-71k
│   └── dataset.py               # PyTorch Dataset class
│
├── models/                      # Model definitions
│   ├── extractor.py             # DINOv2 feature extractor wrapper
│   ├── lora.py                  # LoRA adapter implementation
│   └── correspondence.py        # Full correspondence model
│
├── utils/                       # Utility functions
│   ├── metrics.py               # PCK metric
│   ├── visualization.py         # Match / keypoint visualization
│   └── matching.py              # Nearest-neighbour matching logic
│
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── demo.ipynb                   # Interactive demo notebook
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 📦 Setup

```bash
# 1. Create and activate a conda environment
conda create -n sem_corr python=3.10 -y
conda activate sem_corr

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset — SPair-71k

[SPair-71k](https://cvlab.postech.ac.kr/research/SPair-71k/) is the standard benchmark for semantic correspondence.

- **18** object categories
- **70,958** image pairs with annotated keypoints
- Download via: `python data/download_spair.py --root ./datasets`

---

## 🧠 Approach

### 1. Feature Extraction — DINOv2
We use **DINOv2** (ViT-B/14) as a frozen feature extractor. The **key** features from the self-attention layers encode rich semantic structure.

### 2. Parameter-Efficient Fine-Tuning — LoRA
Since DINOv2 has ~86M parameters, we use **LoRA (Low-Rank Adaptation)** to fine-tune only ~1% of the parameters. LoRA inserts trainable low-rank matrices into the attention heads:

$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times d}$$

with rank $r \ll d$ (e.g., $r=16$).

### 3. Dense Matching
For each source keypoint, we find its nearest neighbour in the target feature map using cosine similarity. A softmax temperature controls the sharpness of the matching distribution.

---

## 📐 Evaluation — PCK

**Percentage of Correct Keypoints (PCK)** at threshold $\alpha$:

$$\text{PCK}@\alpha = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[\|\hat{p}_i - p_i\| \leq \alpha \cdot \max(H, W)\right]$$

We report **PCK@0.1** and **PCK@0.05**.

---

## 🚀 Training

```bash
python train.py \
    --dataset_root ./datasets/SPair-71k \
    --backbone dinov2_vitb14 \
    --lora_rank 16 \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 20 \
    --output_dir ./checkpoints
```

---

## 🧪 Evaluation

```bash
python evaluate.py \
    --dataset_root ./datasets/SPair-71k \
    --checkpoint ./checkpoints/best.pth \
    --alpha 0.1
```

---

## 📚 References

- Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Min et al. (2019). *SPair-71k: A Large-scale Benchmark for Semantic Correspondence.* [arXiv:1908.10543](https://arxiv.org/abs/1908.10543)
- Zhang et al. (2024). *Tale of Two Features: Stable Diffusion Complemented by ViT-based Features for Semantic Correspondence.* CVPR.