import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURAZIONE PERCORSI
# ==========================================
base_folder = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(base_folder, "..", "docs", "plots")
path_cache_risultati = os.path.join(plots_dir, "risultati_accuratezza.json")

# Verifica preliminare se hai già calcolato i dati
if not os.path.exists(path_cache_risultati):
    print(f"[ERRORE] File dei dati non trovato in: {path_cache_risultati}")
    print("Esegui prima lo script principale 'accuratezza.py' per generare i dati!")
    exit(1)

# ==========================================
# CARICAMENTO DATI DAL JSON
# ==========================================
with open(path_cache_risultati, "r") as f:
    dati = json.load(f)

baseline_acc = dati["baseline_acc"]
loss_baseline = dati["loss_baseline"]
epoche_baseline = dati["epoche_baseline"]
percentuali = dati["percentuali"]
risultati = dati["risultati"]
loss_target_50 = dati["loss_target_50"]
epoche_target_50 = dati["epoche_target_50"]

perc_labels = [int(p * 100) for p in percentuali]
delta_outliers = [(baseline_acc - m) * 100 for m in risultati['Outliers']['media']]
delta_gauss = [(baseline_acc - m) * 100 for m in risultati['Gaussiano']['media']]

# ==========================================
# GENERAZIONE GRAFICI (Modificabili all'infinito)
# ==========================================
print("Generazione grafici in corso...")
sns.set_theme(style="whitegrid")

# GRAFICO 1: Performance e Delta Calo (Affiancati)
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Andamento Accuratezza
axes1[0].errorbar(perc_labels, risultati['Outliers']['media'], yerr=risultati['Outliers']['std'], fmt='-o', label='A: Outliers Estremi', color='#17becf', capsize=5, linewidth=2)
axes1[0].errorbar(perc_labels, risultati['Gaussiano']['media'], yerr=risultati['Gaussiano']['std'], fmt='-s', label='B: Rumore Gaussiano', color='#1f77b4', capsize=5, linewidth=2)
axes1[0].axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.4f})')
axes1[0].set_title('Accuratezza: Impatto degli Errori di Misura', fontsize=13, fontweight='bold')
axes1[0].set_xlabel('Percentuale di record corrotti (%)')
axes1[0].set_ylabel('Accuracy')
axes1[0].set_xticks(perc_labels)
axes1[0].legend()

# Subplot 2: Istogramma Delta Perdita
x = np.arange(len(perc_labels))
width = 0.35
axes1[1].bar(x - width/2, delta_outliers, width, label='Perdita Outliers', color='#17becf', alpha=0.85)
axes1[1].bar(x + width/2, delta_gauss, width, label='Perdita Gaussiano', color='#1f77b4', alpha=0.85)
axes1[1].set_title("Calo di Performance (Delta %)", fontsize=13, fontweight='bold')
axes1[1].set_xlabel('Percentuale di record corrotti (%)')
axes1[1].set_ylabel('Perdita (%)')
axes1[1].set_xticks(x)
axes1[1].set_xticklabels(perc_labels)
axes1[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "accuratezza_performance.png"), dpi=300)

# GRAFICO 2: Loss Curve
if loss_target_50 is not None and len(loss_target_50) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(loss_baseline, label=f'Baseline (Sana - {epoche_baseline} epoche)', color='gray', linewidth=2.5)
    ax2.plot(loss_target_50, label=f'Outliers 50% (Arresto a {epoche_target_50} epoche)', color='#17becf', linewidth=2.5, linestyle='--')
    ax2.set_title("Analisi Runtime: Loss Curve (Accuratezza)", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.savefig(os.path.join(plots_dir, "accuratezza_loss_curve.png"), dpi=300)

plt.show()
print("[OK] Grafici aggiornati e visualizzati!")