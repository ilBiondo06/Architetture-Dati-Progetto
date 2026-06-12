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
path_cache_risultati = os.path.join(plots_dir, "risultati_completezza.json")

# Verifica preliminare dell'esistenza dei dati
if not os.path.exists(path_cache_risultati):
    print(f"[ERRORE] File dei dati non trovato in: {path_cache_risultati}")
    print("Devi prima eseguire lo script principale 'completezza.py' per generare i risultati!")
    exit(1)

# ==========================================
# CARICAMENTO DATI DAL JSON
# ==========================================
print(f"Caricamento dati da {path_cache_risultati}...")
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
delta_missing = [(baseline_acc - m) * 100 for m in risultati['Systemic_Missing']['media']]
delta_lag = [(baseline_acc - m) * 100 for m in risultati['Lag_25']['media']]

# ==========================================
# GENERAZIONE GRAFICI
# ==========================================
print("Generazione grafici in corso...")
sns.set_theme(style="whitegrid")

# --- FIGURA 1: PERFORMANCE E DELTA ---
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

axes1[0].errorbar(perc_labels, risultati['Systemic_Missing']['media'], yerr=risultati['Systemic_Missing']['std'], fmt='-o', label='A: Completezza (Blackout Elo)', color='#d62728', capsize=5, linewidth=2)
axes1[0].errorbar(perc_labels, risultati['Lag_25']['media'], yerr=risultati['Lag_25']['std'], fmt='-s', label='B: Tempestività (Lag 25 Partite)', color='#ff7f0e', capsize=5, linewidth=2)
axes1[0].axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.4f})')
axes1[0].set_title('Completezza e Tempestività: Impatto sull\'Accuracy', fontsize=13, fontweight='bold')
axes1[0].set_xlabel('Percentuale di record corrotti (%)')
axes1[0].set_ylabel('Accuracy')
axes1[0].set_xticks(perc_labels)
axes1[0].legend()

x = np.arange(len(perc_labels))
width = 0.35
axes1[1].bar(x - width/2, delta_missing, width, label='Perdita Blackout', color='#d62728', alpha=0.85)
axes1[1].bar(x + width/2, delta_lag, width, label='Perdita Lag-25', color='#ff7f0e', alpha=0.85)
axes1[1].set_title("Calo di Performance (Delta %)", fontsize=13, fontweight='bold')
axes1[1].set_xlabel('Percentuale di record corrotti (%)')
axes1[1].set_ylabel('Perdita (%)')
axes1[1].set_xticks(x)
axes1[1].set_xticklabels(perc_labels)
axes1[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "completezza_tempestivita_performance.png"), dpi=300)

# --- FIGURA 2: LOSS CURVE RUNTIME ---
if loss_target_50 is not None and len(loss_target_50) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(loss_baseline, label=f'Baseline (Sana - {epoche_baseline} epoche)', color='gray', linewidth=2.5)
    ax2.plot(loss_target_50, label=f'Blackout Sistemico 50% (Arresto a {epoche_target_50} epoche)', color='#d62728', linewidth=2.5, linestyle='--')
    ax2.set_title("Analisi Runtime: Loss Curve (Completezza)", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.savefig(os.path.join(plots_dir, "completezza_loss_curve.png"), dpi=300)

plt.show()
print("[OK] Grafici aggiornati e salvati!")