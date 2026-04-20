import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = r"G:\My Drive\Magistrale\2year2semester\AML\Project_AML\reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Imposta uno stile carino e accademico
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300)
    plt.close()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ==============================================================================
# 1. SPair-71k Baselines (Zero-Shot)
# ==============================================================================
plt.figure(figsize=(8, 5))
labels = ['DINOv2 (ViT-B/14)', 'DINOv3 (ViT-B)', 'SAM (ViT-B)']
values = [43.89, 29.56, 17.58]
bars = plt.bar(labels, values, color=[colors[0], colors[1], colors[2]], width=0.6)
plt.title('Baseline Zero-Shot Performance on SPair-71k', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 50)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold')
save_plot("01_Baselines_SPair")

# ==============================================================================
# 2. Main Performance Enhancement (Baseline vs LoRA vs BitFit)
# ==============================================================================
plt.figure(figsize=(8, 5))
labels = ['DINOv2 Zero-Shot', 'BitFit + AW', 'LoRA + AW']
values = [43.89, 77.83, 80.53]
bars = plt.bar(labels, values, color=['gray', colors[1], colors[2]], width=0.6)
plt.title('Impact of Parameter-Efficient Fine-Tuning (SPair-71k)', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 100)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', fontweight='bold')
save_plot("02_PEFT_Main_Results")

# ==============================================================================
# 3. Ablation Study: Adaptive Window (AW)
# ==============================================================================
plt.figure(figsize=(9, 6))
methods = ['LoRA', 'BitFit']
no_aw = [77.63, 76.24]
with_aw = [80.53, 77.83]

x = np.arange(len(methods))
width = 0.35

plt.bar(x - width/2, no_aw, width, label='Hard-Argmax (No AW)', color='#d62728')
plt.bar(x + width/2, with_aw, width, label='Adaptive Soft-Argmax (With AW)', color='#2ca02c')

plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.title('Ablation Study: Adaptive Window Mechanism', fontsize=14, pad=15)
plt.xticks(x, methods, fontsize=12)
plt.legend()
plt.ylim(70, 85)

for i in range(len(x)):
    plt.text(x[i] - width/2, no_aw[i] + 0.5, f"{no_aw[i]:.2f}%", ha='center')
    plt.text(x[i] + width/2, with_aw[i] + 0.5, f"{with_aw[i]:.2f}%", ha='center', fontweight='bold')
save_plot("03_Ablation_AdaptiveWindow")

# ==============================================================================
# 4. Out-of-Distribution Generalization (PF-Pascal)
# ==============================================================================
plt.figure(figsize=(8, 5))
labels = ['DINOv2 Zero-Shot', 'DINOv3 Zero-Shot', 'BitFit (Finetuned)', 'LoRA (Finetuned)']
values = [26.70, 19.50, 42.66, 47.92]
bars = plt.bar(labels, values, color=['gray', 'darkgray', colors[1], colors[0]], width=0.6)
plt.title('OOD Generalization on PF-Pascal', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 60)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold')
save_plot("04_Generalization_PFPascal")

# ==============================================================================
# 5. Curriculum Learning Impact
# ==============================================================================
plt.figure(figsize=(6, 5))
labels = ['LoRA Standard', 'LoRA Curriculum']
values = [80.53, 80.50]  # Basato sui dati estratti precedentemente (nessun guadagno misurabile)
bars = plt.bar(labels, values, color=[colors[0], colors[4]], width=0.5)
plt.title('Impact of Curriculum Learning', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(75, 85)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha='center', fontweight='bold')
save_plot("05_Curriculum_Learning")

# ==============================================================================
# 6. Geometric Robustness (Rotation Degradation Curve)
# ==============================================================================
plt.figure(figsize=(9, 6))
angles = [0, 45, 90, 180]
# NOTA: Solo risultati puri (estratti dal run). Il 15° è rimosso.
lora_rot_pck = [80.53, 61.60, 42.60, 28.06] 
bitfit_rot_pck = [77.83, 56.03, 40.26, 28.57]
# Dati reali estratti dalle baseline su colab
dinov2_rot_pck = [43.89, 35.63, 27.55, 22.82]
dinov3_rot_pck = [29.56, 36.04, 29.98, 23.68]

plt.plot(angles, lora_rot_pck, marker='o', linewidth=2, label='LoRA', color=colors[0])
plt.plot(angles, bitfit_rot_pck, marker='s', linewidth=2, label='BitFit', color=colors[1])
plt.plot(angles, dinov2_rot_pck, marker='^', linewidth=2, linestyle='--', label='DINOv2 (Frozen)', color='gray')
plt.plot(angles, dinov3_rot_pck, marker='v', linewidth=2, linestyle='--', label='DINOv3 (Frozen)', color='darkgray')

plt.title('Geometric Robustness: Rotation Degradation Curve', fontsize=14, pad=15)
plt.xlabel('Rotation Angle (Degrees)', fontsize=12)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.xticks(angles)
plt.ylim(20, 90)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
for i, txt in enumerate(lora_rot_pck):
    plt.annotate(f"{txt:.1f}", (angles[i], lora_rot_pck[i]+2), ha='center', color=colors[0])
for i, txt in enumerate(bitfit_rot_pck):
    plt.annotate(f"{txt:.1f}", (angles[i], bitfit_rot_pck[i]-3), ha='center', color=colors[1])
save_plot("06_Robustness_Rotation")

# ==============================================================================
# 7. Temperature Calibration (Softmax Entropy)
# ==============================================================================
plt.figure(figsize=(8, 5))
temps = ['0.01 (Hard)', '0.05 (Calibrated)', '0.10 (Soft)', '0.50 (Uniform)']
pcks = [80.98, 81.87, 79.00, 26.32]

bars = plt.bar(temps, pcks, color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'], width=0.6)
plt.title('Temperature Calibration (Soft-Argmax)', fontsize=14, pad=15)
plt.xlabel('Temperature (T)')
plt.ylabel('PCK@0.10 (%)')
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', fontweight='bold')
plt.axhline(y=80.98, color='gray', linestyle='--', alpha=0.5) # Linea per indicare il baseline T=0.01
save_plot("07_Temperature_Calibration")

# ==============================================================================
# 8. Trainable Parameters Comparison
# ==============================================================================
plt.figure(figsize=(7, 5))
labels = ['LoRA (Rank=16)', 'BitFit']
params_m = [0.906, 0.082] # in milioni (0.9M per lora, 82K per bitfit)
percent = [1.036, 0.093]  # percentuale rispetto a 87M

bars = plt.bar(labels, params_m, color=['#9467bd', '#8c564b'], width=0.5)
plt.title('Parameter Efficiency Comparison', fontsize=14, pad=15)
plt.ylabel('Trainable Parameters (Millions)', fontsize=12)

for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{params_m[i]}M\n({percent[i]:.3f}%)", ha='center', fontweight='bold')
save_plot("08_Parameter_Efficiency")

# ==============================================================================
# 9. Layer-wise Representation Analysis (DINOv2)
# ==============================================================================
plt.figure(figsize=(7, 5))
labels = ['Layer 4 (Low-Level)', 'Layer 8 (Mid-Level)', 'Layer 11 (High-Level)']
values = [13.96, 20.15, 43.38]
bars = plt.bar(labels, values, color=['#aec7e8', '#7f7f7f', '#1f77b4'], width=0.5)
plt.title('DINOv2 Layer-wise Semantic Capacity', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 50)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold')
save_plot("09_LayerWise_Analysis")

print("[INFO] Plots generati con successo nella cartella 'reports'")
