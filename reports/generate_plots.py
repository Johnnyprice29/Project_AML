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
plt.title('Baseline Zero-Shot Performance on SPair-71k', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 55)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold', fontsize=11)
save_plot("01_Baselines_SPair")

# ==============================================================================
# 2. Main Performance Enhancement (Baseline vs Probe vs LoRA)
# ==============================================================================
plt.figure(figsize=(9, 5))
labels = ['DINOv2 Zero-Shot', 'Linear Probe + AW', 'BitFit + AW', 'LoRA + AW']
values = [43.88, 66.47, 77.83, 80.53]
bars = plt.bar(labels, values, color=[colors[3], colors[5], colors[1], colors[0]], width=0.6)
plt.title('Impact of Adaptation Strategies on SPair-71k', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 100)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', fontweight='bold', fontsize=12)
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
    plt.text(x[i] - width/2, no_aw[i] + 0.5, f"{no_aw[i]:.2f}%", ha='center', fontweight='bold', fontsize=10)
    plt.text(x[i] + width/2, with_aw[i] + 0.5, f"{with_aw[i]:.2f}%", ha='center', fontweight='bold', fontsize=10, color='darkgreen')
save_plot("03_Ablation_AdaptiveWindow")

# ==============================================================================
# 4. Out-of-Distribution Generalization (PF-Pascal)
# ==============================================================================
plt.figure(figsize=(8, 5))
labels = ['DINOv2 Zero-Shot', 'DINOv3 Zero-Shot', 'BitFit (Finetuned)', 'LoRA (Finetuned)']
# Dati SPair corrispondenti per il confronto
spair_values = [43.88, 44.49, 77.83, 80.53] 
values = [26.76, 26.70, 46.26, 47.92]

# Barre "fantasma" per mostrare la degradazione da SPair
ghost_bars = plt.bar(labels, spair_values, color=[colors[3], colors[5], colors[1], colors[0]], alpha=0.15, edgecolor='black', linestyle='--', width=0.6, label='SPair-71k (In-Dist)')
bars = plt.bar(labels, values, color=[colors[3], colors[5], colors[1], colors[0]], width=0.6, label='PF-Pascal (OOD)')

plt.title('OOD Generalization: PF-Pascal vs SPair-71k', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 100)
plt.legend()

for i, bar in enumerate(bars):
    yval = bar.get_height()
    spair_val = spair_values[i]
    ratio = yval / spair_val
    # Etichetta SPair (sopra la barra fantasma)
    plt.text(bar.get_x() + bar.get_width()/2, spair_val + 1, f"{spair_val:.2f}%", ha='center', color='gray', fontsize=9, alpha=0.7)
    # Etichetta PF-Pascal (sopra la barra piena)
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%\n({ratio:.2f}x)", ha='center', fontweight='bold', fontsize=10)
save_plot("04_Generalization_PFPascal")

# ==============================================================================
# 5. Deep Ablation: Training Strategies & Mechanisms
# ==============================================================================
plt.figure(figsize=(12, 7))
models = ['Linear Probe (MLP Only)', 'LoRA (Backbone Adaptation)']

# Dati estratti dai file dei risultati (Mean PCK@0.10)
std_aw = [66.47, 80.53]
std_no_aw = [65.69, 78.13]
curr_aw = [60.89, 79.60]
curr_no_aw = [60.56, 77.13]

x = np.arange(len(models))
width = 0.2

plt.bar(x - 1.5*width, std_aw, width, label='Standard + AW', color=colors[1])
plt.bar(x - 0.5*width, std_no_aw, width, label='Standard No AW', color=colors[1], alpha=0.5, hatch='//')
plt.bar(x + 0.5*width, curr_aw, width, label='Curriculum + AW', color=colors[2])
plt.bar(x + 1.5*width, curr_no_aw, width, label='Curriculum No AW', color=colors[2], alpha=0.5, hatch='\\\\')

plt.title('Comprehensive Ablation: Curriculum Learning & Adaptive Window', fontsize=15, pad=20)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.xticks(x, models, fontsize=13)
plt.ylim(50, 90)
plt.legend(loc='upper left', ncol=2, frameon=True, shadow=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Aggiunta etichette sui valori (Sempre in grassetto per visibilità)
for i in range(len(models)):
    plt.text(i - 1.5*width, std_aw[i] + 1, f"{std_aw[i]:.2f}%", ha='center', fontweight='bold', fontsize=10)
    plt.text(i - 0.5*width, std_no_aw[i] + 1, f"{std_no_aw[i]:.2f}%", ha='center', fontweight='bold', fontsize=9)
    plt.text(i + 0.5*width, curr_aw[i] + 1, f"{curr_aw[i]:.2f}%", ha='center', fontweight='bold', fontsize=10)
    plt.text(i + 1.5*width, curr_no_aw[i] + 1, f"{curr_no_aw[i]:.2f}%", ha='center', fontweight='bold', fontsize=9)

# Linea di riferimento Baseline Zero-Shot
plt.axhline(y=43.88, color=colors[3], linestyle='--', alpha=0.6, label='DINOv2 Zero-Shot')
plt.text(1.3, 45, 'Zero-Shot Baseline (43.88%)', color=colors[3], fontweight='bold', fontsize=9)

save_plot("05_Training_Strategies_Ablation")

# ==============================================================================
# 6. Geometric Robustness (Rotation Degradation Curve)
# ==============================================================================
plt.figure(figsize=(9, 6))
angles = [0, 45, 90, 180]
# NOTA: Solo risultati puri (estratti dal run). Il 15° è rimosso.
lora_rot_pck = [78.13, 61.60, 42.60, 28.06]
bitfit_rot_pck = [75.40, 56.03, 40.26, 28.57]
# Dati reali estratti dalle baseline su colab
dinov2_rot_pck = [43.88, 35.63, 27.55, 22.82]
dinov3_rot_pck = [44.49, 36.10, 30.15, 24.12]

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
    plt.annotate(f"{txt:.2f}%\n({ratio:.2f}x)", (angles[i], lora_rot_pck[i]+2), ha='center', color=colors[0], fontweight='bold')
for i, txt in enumerate(bitfit_rot_pck):
    ratio = txt / bitfit_rot_pck[0]
    plt.annotate(f"{txt:.2f}%", (angles[i], bitfit_rot_pck[i]-5), ha='center', color=colors[1], fontweight='bold')
for i, txt in enumerate(dinov2_rot_pck):
    ratio = txt / dinov2_rot_pck[0]
    plt.annotate(f"{ratio:.2f}x", (angles[i], dinov2_rot_pck[i]-3), ha='center', color=colors[3], fontsize=9)
save_plot("06_Robustness_Rotation")

# ==============================================================================
# 7. Temperature Calibration (Softmax Entropy)
# ==============================================================================
plt.figure(figsize=(8, 5))
temps = ['0.01 (Hard)', '0.05 (Calibrated)', '0.10 (Soft)', '0.50 (Uniform)']
pcks = [80.53, 81.34, 78.45, 25.80]

bars = plt.bar(temps, pcks, color=[colors[3], colors[1], colors[2], colors[0]], width=0.6)
plt.title('Temperature Calibration (Soft-Argmax)', fontsize=14, pad=15)
plt.xlabel('Temperature (T)')
plt.ylabel('PCK@0.10 (%)')
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', fontweight='bold', fontsize=11)
plt.axhline(y=80.98, color=colors[3], linestyle='--', alpha=0.5) # Linea per indicare il baseline T=0.01
save_plot("07_Temperature_Calibration")

# ==============================================================================
# 8. Trainable Parameters Comparison
# ==============================================================================
plt.figure(figsize=(8, 5))
labels = ['LoRA (Rank=16)', 'Linear Probe', 'BitFit']
params_m = [0.906, 0.263, 0.082] 
percent = [1.036, 0.301, 0.093] 

bars = plt.bar(labels, params_m, color=[colors[0], colors[5], colors[1]], width=0.5)
plt.title('Parameter Efficiency Comparison', fontsize=14, pad=15)
plt.ylabel('Trainable Parameters (Millions)', fontsize=12)
plt.ylim(0, 1.2)

for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{params_m[i]}M\n({percent[i]:.3f}%)", ha='center', fontweight='bold', fontsize=10)
save_plot("08_Parameter_Efficiency")

# ==============================================================================
# 9. Layer-wise Representation Analysis (DINOv2)
# ==============================================================================
plt.figure(figsize=(7, 5))
labels = ['Layer 4 (Low-Level)', 'Layer 8 (Mid-Level)', 'Layer 11 (High-Level)']
values = [14.00, 20.29, 43.97]
bars = plt.bar(labels, values, color=[colors[4], colors[3], colors[1]], width=0.5)
plt.title('DINOv2 Layer-wise Semantic Capacity', fontsize=14, pad=15)
plt.ylabel('PCK@0.10 (%)', fontsize=12)
plt.ylim(0, 50)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold', fontsize=11)
save_plot("09_LayerWise_Analysis")

print("[INFO] Plots generati con successo nella cartella 'reports'")
