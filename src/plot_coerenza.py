import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURAZIONE PERCORSI
# ==========================================
base_folder = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(base_folder, "..", "results", "plots")
path_cache_risultati = os.path.join(plots_dir, "risultati_coerenza.json")

# Verifica preliminare dell'esistenza dei dati calcolati
if not os.path.exists(path_cache_risultati):
    print(f"[ERRORE] File dei dati non trovato in: {path_cache_risultati}")
    print("Devi prima eseguire lo script principale 'coerenza.py' per generare i risultati!")
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

# Ricalcolo delle etichette e dei delta per il plotting
perc_labels = [int(p * 100) for p in percentuali]
delta_target = [(baseline_acc - m) * 100 for m in risultati['Target_Flipping']['media']]
delta_elo = [(baseline_acc - m) * 100 for m in risultati['Elo_Ranking']['media']]

# ==========================================
# GENERAZIONE E SALVATAGGIO DEI GRAFICI
# ==========================================
print("Generazione e salvataggio dei grafici in corso...")
sns.set_theme(style="whitegrid")

# --- FIGURA 1: ACCURATEZZA ASSOLUTA VS DELTA PERDITA (Affiancati) ---
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

# Subplot Sinistro: Andamento dell'Accuratezza
axes1[0].errorbar(perc_labels, risultati['Target_Flipping']['media'], yerr=risultati['Target_Flipping']['std'], fmt='-o', label='Target Flipping', color='#d62728', capsize=5, linewidth=2)
axes1[0].errorbar(perc_labels, risultati['Elo_Ranking']['media'], yerr=risultati['Elo_Ranking']['std'], fmt='-s', label='Incoerenza Elo-Ranking', color='#9467bd', capsize=5, linewidth=2)
axes1[0].axhline(y=baseline_acc, color='#1f77b4', linestyle='--', linewidth=2, label=f'Baseline Intatta ({baseline_acc:.4f})')
axes1[0].set_title('Accuratezza vs Percentuale di Errore', fontsize=13, fontweight='bold')
axes1[0].set_xlabel('Record incoerenti (%)', fontsize=11)
axes1[0].set_ylabel('Accuratezza Media', fontsize=11)
axes1[0].set_xticks(perc_labels)
axes1[0].legend(loc='lower left')

# Subplot Destro: Istogramma del Calo di Performance (Delta %)
x = np.arange(len(perc_labels))
width = 0.35
axes1[1].bar(x - width/2, delta_target, width, label='Perdita Target Flipping', color='#d62728', alpha=0.85)
axes1[1].bar(x + width/2, delta_elo, width, label='Perdita Incoerenza Elo-Rank', color='#9467bd', alpha=0.85)
axes1[1].set_title("Calo di Performance (Delta %)", fontsize=13, fontweight='bold')
axes1[1].set_xlabel('Record incoerenti (%)', fontsize=11)
axes1[1].set_ylabel('Perdita di Accuratezza rispetto alla Baseline (Delta %)', fontsize=11)
axes1[1].set_xticks(x)
axes1[1].set_xticklabels(perc_labels)
axes1[1].legend(loc='upper left')

# Aggiunta dei testi con la percentuale esatta sopra le ultime colonne (50%)
axes1[1].text(x[-1] - width/2, delta_target[-1] + 0.3, f"{delta_target[-1]:.1f}%", ha='center', fontsize=10, fontweight='bold')
axes1[1].text(x[-1] + width/2, delta_elo[-1] + 0.3, f"{delta_elo[-1]:.1f}%", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
path_fig1 = os.path.join(plots_dir, "coerenza_accuratezza_vs_delta.png")
plt.savefig(path_fig1, dpi=300)
print(f"[OK] Grafico delle performance salvato in: {path_fig1}")

# --- FIGURA 2: LOSS CURVE RUNTIME ---
if loss_target_50 is not None and len(loss_target_50) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(loss_baseline, label=f'Baseline: Convergenza in {epoche_baseline} epoche', color='#2ca02c', linewidth=2.5)
    ax2.plot(loss_target_50, label=f'Target Flipping 50%: Convergenza in {epoche_target_50} epoche', color='#d62728', linewidth=2.5, linestyle='--')
    ax2.set_title("Analisi di Runtime: Come cambia l'apprendimento (Loss Curve)", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoche di addestramento', fontsize=12)
    ax2.set_ylabel('Error Loss', fontsize=12)
    ax2.legend()
    
    plt.tight_layout()
    path_fig2 = os.path.join(plots_dir, "coerenza_loss_curve.png")
    plt.savefig(path_fig2, dpi=300)
    print(f"[OK] Grafico delle Loss Curve salvato in: {path_fig2}")

# Mostra i grafici a schermo
plt.show()
print("[FINE] Tutti i grafici sono stati visualizzati e aggiornati!")