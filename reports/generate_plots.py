import matplotlib.pyplot as plt
import numpy as np
import os


OUTPUT_DIR_DRIVE = r"G:\My Drive\AML\reports"
OUTPUT_DIR_LOCAL = r"G:\My Drive\Magistrale\2year2semester\AML\Project_AML\reports"
os.makedirs(OUTPUT_DIR_DRIVE, exist_ok=True)
os.makedirs(OUTPUT_DIR_LOCAL, exist_ok=True)

# Imposta uno stile carino e accademico
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_DRIVE, f"{name}.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR_LOCAL, f"{name}.png"), dpi=300)
    plt.close()

# Palette: Rosso Scuro, Blu Scuro, Viola Profondo, Navy (per baseline)
colors = ['#A93226', '#2471A3', '#7D3C98', '#2E4053', '#1B4F72', '#512E5F'] 
# 0: Red, 1: Blue, 2: Purple, 3: Navy Slate (Baseline), 4: Deep Blue, 5: Deep Purple

# ==============================================================================
# 1. SPair-71k Baselines (Zero-Shot)
# ==============================================================================
plt.figure(figsize=(8, 5))
labels = ['DINOv2 (ViT-B/14)', 'DINOv3 (ViT-B)', 'SAM (ViT-B)']
values = [43.88, 44.49, 6.41]

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
values = [43.88, 77.83, 80.53]

bars = plt.bar(labels, values, color=[colors[3], colors[1], colors[0]], width=0.6)
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
no_aw = [78.13, 75.40]
with_aw = [80.53, 77.83]


x = np.arange(len(methods))
width = 0.35

plt.bar(x - width/2, no_aw, width, label='Hard-Argmax (No AW)', color=colors[2]) 
plt.bar(x + width/2, with_aw, width, label='Adaptive Soft-Argmax (With AW)', color=colors[1]) 


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
# Dati SPair corrispondenti per il confronto
spair_values = [43.88, 44.49, 77.83, 80.53] 
values = [26.76, 26.76, 46.26, 47.92]


# Barre "fantasma" per mostrare la degradazione da SPair
plt.bar(labels, spair_values, color=[colors[3], colors[5], colors[1], colors[0]], alpha=0.15, edgecolor='black', linestyle='--', width=0.6, label='SPair-71k (In-Dist)')
bars = plt.bar(labels, values, color=[colors[3], colors[5], colors[1], colors[0]], width=0.6, label='PF-Pascal (OOD)')

plt.title('OOD Generalization: PF-Pascal vs SPair-71k', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 100)
plt.legend()

for i, bar in enumerate(bars):
    yval = bar.get_height()
    ratio = yval / spair_values[i]
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%\n({ratio:.2f}x)", ha='center', fontweight='bold', fontsize=10)
save_plot("04_Generalization_PFPascal")

# ==============================================================================
# 5. Curriculum Learning Impact
# ==============================================================================
plt.figure(figsize=(6, 5))
labels = ['LoRA Standard', 'LoRA Curriculum']
pure = [78.13, 77.13]
aw = [80.53, 79.60]


# Barre base
plt.bar(labels, pure, color=[colors[0], colors[2]], width=0.5, label='Standard Training')
# Barre AW (Ghost)
plt.bar(labels, aw, color=[colors[0], colors[2]], alpha=0.2, edgecolor='black', linestyle='--', width=0.5, label='With Adaptive Window')

plt.title('Impact of Curriculum Learning on Convergence', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(70, 90)
plt.legend(loc='upper right', frameon=True)

for i in range(len(labels)):
    # Valore pure (dentro la barra)
    plt.text(i, pure[i] - 3, f"{pure[i]:.2f}%", ha='center', color='white', fontweight='bold', fontsize=10)
    # Valore AW (sopra la barra)
    plt.text(i, aw[i] + 1, f"{aw[i]:.2f}%", ha='center', color=colors[i*2], fontweight='bold', fontsize=10)
save_plot("05_Curriculum_Learning")

# ==============================================================================
# 6. Geometric Robustness (Rotation Degradation Curve)
# ==============================================================================
plt.figure(figsize=(9, 6))
angles = [0, 45, 90, 180]
lora_rot_pck = [78.13, 61.60, 42.60, 28.06]
bitfit_rot_pck = [75.40, 56.03, 40.26, 28.57]
dinov2_rot_pck = [43.88, 35.63, 27.55, 22.82]
dinov3_rot_pck = [44.49, 36.04, 29.98, 23.68]


plt.plot(angles, lora_rot_pck, marker='o', linewidth=2, label='LoRA', color=colors[0])
plt.plot(angles, bitfit_rot_pck, marker='s', linewidth=2, label='BitFit', color=colors[1])
plt.plot(angles, dinov2_rot_pck, marker='^', linewidth=2, linestyle='--', label='DINOv2 (Frozen)', color=colors[3])
plt.plot(angles, dinov3_rot_pck, marker='v', linewidth=2, linestyle='--', label='DINOv3 (Frozen)', color=colors[5])


plt.title('Geometric Robustness: Rotation Degradation Curve', fontsize=14, pad=15)
plt.xlabel('Rotation Angle (Degrees)', fontsize=12)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.xticks(angles)
plt.ylim(20, 90)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
for i, txt in enumerate(lora_rot_pck):
    ratio = txt / lora_rot_pck[0]
    plt.annotate(f"{txt:.1f}\n({ratio:.2f}x)", (angles[i], lora_rot_pck[i]+2), ha='center', color=colors[0], fontweight='bold')
for i, txt in enumerate(bitfit_rot_pck):
    ratio = txt / bitfit_rot_pck[0]
    plt.annotate(f"{txt:.1f}\n({ratio:.2f}x)", (angles[i], bitfit_rot_pck[i]-5), ha='center', color=colors[1])
for i, txt in enumerate(dinov2_rot_pck):
    ratio = txt / dinov2_rot_pck[0]
    plt.annotate(f"{ratio:.2f}x", (angles[i], dinov2_rot_pck[i]-3), ha='center', color=colors[3], fontsize=9)
save_plot("06_Robustness_Rotation")

# ==============================================================================
# 7. Temperature Calibration (Softmax Entropy)
# ==============================================================================
plt.figure(figsize=(8, 5))
temps = ['0.01 (Hard)', '0.05 (Calibrated)', '0.10 (Soft)', '0.50 (Uniform)']
pcks = [80.98, 81.87, 79.00, 26.32]

bars = plt.bar(temps, pcks, color=[colors[3], colors[1], colors[2], colors[0]], width=0.6)
plt.title('Temperature Calibration (Soft-Argmax)', fontsize=14, pad=15)
plt.xlabel('Temperature (T)')
plt.ylabel('PCK@0.10 (%)')
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', fontweight='bold')
plt.axhline(y=80.98, color=colors[3], linestyle='--', alpha=0.5) # Linea per indicare il baseline T=0.01
save_plot("07_Temperature_Calibration")

# ==============================================================================
# 8. Trainable Parameters Comparison
# ==============================================================================
plt.figure(figsize=(7, 5))
labels = ['LoRA (Rank=16)', 'BitFit']
params_m = [0.906, 0.082] # in milioni (0.9M per lora, 82K per bitfit)
percent = [1.036, 0.093]  # percentuale rispetto a 87M

bars = plt.bar(labels, params_m, color=[colors[0], colors[1]], width=0.5)
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

bars = plt.bar(labels, values, color=[colors[4], colors[3], colors[1]], width=0.5)
plt.title('DINOv2 Layer-wise Semantic Capacity', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 50)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold')
save_plot("09_LayerWise_Analysis")

print("[INFO] Plots generati con successo nella cartella 'reports'")
