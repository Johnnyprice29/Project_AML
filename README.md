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
├── dataloaders/                 # Python code for data loading
│   ├── __init__.py
│   ├── download_spair.py        # Script to download SPair-71k
│   ├── download_pfpascal.py     # Script to download PF-Pascal
│   ├── spair.py                 # SPairDataset class
│   └── pfpascal.py              # PFPascalDataset class
│
├── models/
│   ├── extractor.py             # DINOv2 feature extractor
│   ├── lora.py                  # LoRA implementation
│   └── correspondence.py        # Cost volume + Adaptive Window + SAM
│
├── utils/
│   ├── metrics.py               # PCK@α evaluation
│   ├── matching.py              # Cost volume & NN matching
│   ├── visualization.py         # Drawing matches & heatmaps
│   ├── adaptive_window.py       # ★ Adaptive Window Soft-Argmax
│   ├── curriculum.py            # ★ Curriculum Learning sampler
│   ├── segment_aware.py         # ★ SAM masking
│   └── demo_utils.py            # ★ Gradio Demos (Comparison, Robustness, SAM)
│
├── data/                        # Datasets (SPair-71k, PF-Pascal)
├── checkpoints/                 # Saved model weights
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation on SPair/Pascal
├── ablate_temperature.py        # Ablation study on softmax temperature
├── Notebook_training_eval.ipynb # Development notebook
├── requirements.txt
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

# 3. Download the datasets
python dataloaders/download_spair.py --root ./data
python dataloaders/download_pfpascal.py --root ./data

# 4. (Optional) For Segment-Aware: install SAM and download weights
pip install git+https://github.com/facebookresearch/segment-anything.git
# Download sam_vit_b_01ec64.pth to project root
```

---

## 📊 Datasets

### SPair-71k
[SPair-71k](https://cvlab.postech.ac.kr/research/SPair-71k/) is the standard benchmark for semantic correspondence.
- **18** categories, **70k** pairs.
- Metadata (viewpoint, scale, etc.) used for **Curriculum Learning**.

### PF-Pascal
[PF-Pascal](https://www.di.ens.fr/willow/research/proposalflow/) is a smaller, classic benchmark.
- **20** Pascal VOC categories.
- Used for cross-dataset validation and robustness testing.

---

## 🧠 Approach

### 1. Feature Extraction — DINOv2
We use **DINOv2** (ViT-B/14) as the backbone. Its dense patch features encode rich semantic structure learned via self-supervised training.

### 2. Parameter-Efficient Fine-Tuning — LoRA
Since DINOv2 has ~86M parameters, we use **LoRA (Low-Rank Adaptation)** to fine-tune only ~1% of the parameters by inserting trainable low-rank bypass matrices into every attention layer:

$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times d}$$

with rank $r \ll d$ (default $r=16$).

### 3. ★ Curriculum Learning (`utils/curriculum.py`)
Instead of training on all pairs from the start, we rank pairs by difficulty using SPair-71k metadata (viewpoint + scale + truncation variation) and expose progressively harder examples:

- **Epoch 1**: only the 30% easiest pairs (similar pose, same scale)
- **Linear ramp**: difficulty increases over `--curriculum_epochs` epochs
- **After ramp**: all pairs used (standard training)

### 4. Dense Matching + Cost Volume
For each source keypoint, we gather its similarity row from the cosine-similarity cost volume $(B \times N_s \times N_t)$ and find the best matching position in the target.

### 5. ★ Adaptive Window Soft-Argmax (`utils/adaptive_window.py`)
Instead of a fixed-size window or plain argmax, we compute the **Shannon entropy** of the similarity distribution:

- **Low entropy** (confident/sharp peak) → small window radius → precise sub-pixel localisation
- **High entropy** (flat/uncertain) → large window radius → wider exploration

The window radius is linearly interpolated between `aw_min_radius` and `aw_max_radius` based on entropy.

### 6. ★ Segment-Aware Correspondence (`utils/segment_aware.py`)
We integrate **SAM (Segment Anything Model)** to restrict matching to the target object:

1. SAM generates a binary mask for the target image using the keypoint location as a prompt.
2. The mask is downsampled to feature-grid resolution $(h \times w)$.
3. Cost-volume columns outside the mask are set to $-\infty$ before soft-argmax.

This eliminates false positives in cluttered scenes.

---

## 📐 Evaluation — PCK

**Percentage of Correct Keypoints (PCK)** at threshold $\alpha$:

$$\text{PCK}@\alpha = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[\|\hat{p}_i - p_i\| \leq \alpha \cdot \max(H, W)\right]$$

We report **PCK@0.1** and **PCK@0.05**, both globally and per category.

---

## 🚀 Training

**Standard run (GPU — tutte le estensioni abilitate):**
```bash
python train.py \
    --dataset_root ./data/SPair-71k \
    --backbone dinov2_vitb14 \
    --lora_rank 16 \
    --epochs 20 \
    --curriculum_epochs 10 \
    --curriculum_start_frac 0.3 \
    --aw_min_radius 2 \
    --aw_max_radius 7 \
    --output_dir ./checkpoints
```

**Quick test (CPU, 1 epoch, verifica che tutto funzioni):**
```bash
python train.py \
    --dataset_root ./data/SPair-71k \
    --epochs 1 \
    --batch_size 4 \
    --curriculum_epochs 0 \
    --num_workers 0
```

**Disable specific extensions:**
```bash
# No curriculum (train on all pairs from epoch 1)
python train.py ... --curriculum_epochs 0

# No adaptive windowing (use plain hard-argmax)
python train.py ... --no_adaptive_win
```

---

## 🧪 Evaluation

```bash
python evaluate.py \
    --dataset_root ./data/SPair-71k \
    --checkpoint ./checkpoints/best.pth \
    --alpha 0.1
```

---

## 🎮 Interactive Demos

We provide several interactive Gradio demos in `utils/demo_utils.py`, primarily designed for use in Jupyter Notebooks/Colab:

- **Comparison Demo**: Side-by-side comparison between the Baseline and our fine-tuned model.
- **Robustness Demo**: Test how the model handles arbitrary geometric rotations.
- **Segment-Aware Demo**: Visualise how SAM-based masking improves matching precision.

To launch them in a notebook:
```python
from utils.demo_utils import launch_comparison_demo
launch_comparison_demo(ckpt_name='my_best_model')
```

---

## 📚 References

- Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Min et al. (2019). *SPair-71k: A Large-scale Benchmark for Semantic Correspondence.* [arXiv:1908.10543](https://arxiv.org/abs/1908.10543)
- Kirillov et al. (2023). *Segment Anything.* [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- Bengio et al. (2009). *Curriculum Learning.* ICML.
- Zhang et al. (2024). *Tale of Two Features: Stable Diffusion Complemented by ViT-based Features for Semantic Correspondence.* CVPR.