import matplotlib.pyplot as plt

# Dati estratti dai log
epochs = [1, 2, 3, 4, 5]
lora_val_pck = [80.43, 81.11, 81.74, 81.86, 81.63]
bitfit_val_pck = [69.96, 75.39, 79.71, 80.19, 79.95]

plt.figure(figsize=(10, 6))
plt.plot(epochs, lora_val_pck, 'o-', label='LoRA (Best: 81.86%)', linewidth=2, color='#1f77b4')
plt.plot(epochs, bitfit_val_pck, 's--', label='BitFit (Best: 80.19%)', linewidth=2, color='#ff7f0e')

plt.title('Performance Comparison: LoRA vs BitFit (Validation PCK@0.1)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation PCK @ 0.1 (%)', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.ylim(65, 85)

# Aggiungo annotazioni per i picchi
plt.annotate(f'Peak: 81.86%', (4, 81.86), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='#1f77b4')
plt.annotate(f'Peak: 80.19%', (4, 80.19), textcoords="offset points", xytext=(0,-15), ha='center', fontweight='bold', color='#ff7f0e')

plt.tight_layout()
plt.savefig('g:/My Drive/AML/Results/training_comparison.png', dpi=300)
print("Grafico salvato con successo in Results/training_comparison.png")
