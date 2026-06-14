import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURAZIONE PERCORSI
# ==========================================
base_folder = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_folder, "..", "results")
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True) # Crea la cartella se non esiste

path_json = os.path.join(results_dir, "risultati_coerenza.json")

if not os.path.exists(path_json):
    print(f"[ERRORE] File dei dati non trovato in: {path_json}")
    exit(1)

# ==========================================
# CARICAMENTO DATI DAL JSON
# ==========================================
print(f"Generazione grafici in corso da: {path_json}...")
with open(path_json, "r") as f:
    dati = json.load(f)

baseline_acc = dati["baseline_acc"]
percentuali = dati["percentuali"]
perc_labels = [int(p * 100) for p in percentuali]
risultati = dati["risultati"]

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# GRAFICO 1: ACCURATEZZA CLASSICA E DELTA
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Accuracy Assoluta
axes1[0].errorbar(perc_labels, risultati['Target_Flipping']['media'], yerr=risultati['Target_Flipping']['std'], fmt='-o', label='Target Flipping (Critico)', color='#d62728', capsize=5, linewidth=2.5)
axes1[0].errorbar(perc_labels, risultati['Elo_Ranking']['media'], yerr=risultati['Elo_Ranking']['std'], fmt='-s', label='Elo vs Ranking (Dominio)', color='#1f77b4', capsize=5, linewidth=2.5)
axes1[0].axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.4f})')
axes1[0].set_title('Coerenza: Impatto sull\'Accuracy', fontsize=14, fontweight='bold')
axes1[0].set_xlabel('Percentuale di record incoerenti (%)'); axes1[0].set_ylabel('Accuracy')
axes1[0].set_xticks(perc_labels); axes1[0].legend()

# Subplot 2: Istogramma Delta Perdita
delta_target = [(baseline_acc - m) * 100 for m in risultati['Target_Flipping']['media']]
delta_elo = [(baseline_acc - m) * 100 for m in risultati['Elo_Ranking']['media']]
x = np.arange(len(perc_labels)); width = 0.35

axes1[1].bar(x - width/2, delta_target, width, label='Perdita Target Flipping', color='#d62728', alpha=0.85)
axes1[1].bar(x + width/2, delta_elo, width, label='Perdita Elo vs Rank', color='#1f77b4', alpha=0.85)
axes1[1].set_title("Calo di Performance Netta (Delta %)", fontsize=14, fontweight='bold')
axes1[1].set_xlabel('Percentuale di record incoerenti (%)'); axes1[1].set_ylabel('Perdita (%)')
axes1[1].set_xticks(x); axes1[1].set_xticklabels(perc_labels); axes1[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "coerenza_01_performance_assoluta.png"), dpi=300)

# -------------------------------
# GRAFICO 2: GAP TRAIN VS TEST
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(perc_labels, risultati['Target_Flipping']['media_train'], label='Accuracy sul TRAIN', color='#ff7f0e', marker='o', linewidth=2.5)
plt.plot(perc_labels, risultati['Target_Flipping']['media'], label='Accuracy sul TEST (Realtà)', color='#d62728', marker='s', linewidth=2.5)
plt.fill_between(perc_labels, risultati['Target_Flipping']['media'], risultati['Target_Flipping']['media_train'], color='gray', alpha=0.2, label='Generalization Gap')

plt.title('La Forbice dell\'Overfitting: Train vs Test (Target Flipping)', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record incoerenti (%)'); plt.ylabel('Accuratezza')
plt.legend()
plt.savefig(os.path.join(plots_dir, "coerenza_02_train_test_gap.png"), dpi=300)

# ----------------------------------
# GRAFICO 3: CONFIDENZA PREDITTIVA 
# ----------------------------------
plt.figure(figsize=(10, 5))
plt.plot(perc_labels, [c * 100 for c in risultati['Target_Flipping']['confidenza']], label='Target Flipping (Confidenza)', color='purple', marker='^', linewidth=2.5)
plt.axhline(y=75.0, color='gray', linestyle='--', label='Soglia Teorica di Rischio (75%)')

plt.title('Il Crollo delle Certezze: Confidenza Predittiva', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record incoerenti (%)'); plt.ylabel('Probabilità media delle previsioni (%)')
plt.legend()
plt.savefig(os.path.join(plots_dir, "coerenza_03_confidenza.png"), dpi=300)

# ---------------------------------
# GRAFICO 4: COSTO COMPUTAZIONALE
# ---------------------------------
plt.figure(figsize=(9, 5))
plt.bar(perc_labels, risultati['Target_Flipping']['tempi'], color='orange', edgecolor='black', alpha=0.8)

plt.title('Impatto Computazionale: Tempi di Addestramento', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record incoerenti (%)'); plt.ylabel('Tempo impiegato (secondi)')
plt.savefig(os.path.join(plots_dir, "coerenza_04_tempi_calcolo.png"), dpi=300)

# ---------------------------------------------
# GRAFICO 5: EVOLUZIONE MATRICE DI CONFUSIONE 
# ---------------------------------------------
cm_dict = risultati['Target_Flipping'].get('cm_per_step')

if cm_dict is not None and len(cm_dict) > 0:
    fig5, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    # Ordiniamo le percentuali per essere sicuri della sequenza
    sorted_keys = sorted(cm_dict.keys(), key=float)
    
    for i, p_str in enumerate(sorted_keys):
        cm = np.array(cm_dict[p_str])
        perc_int = int(float(p_str) * 100)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                    xticklabels=['Sconfitta', 'Vittoria'],
                    yticklabels=['Sconfitta', 'Vittoria'], ax=axes[i])
        
        axes[i].set_title(f"Degrado: {perc_int}%", fontweight='bold')
        axes[i].set_xlabel('Predetto'); axes[i].set_ylabel('Vero')
    
    # Nasconde gli spazi vuoti se ci sono meno di 6 percentuali
    for j in range(i + 1, len(axes)):
        fig5.delaxes(axes[j])
        
    plt.suptitle('Evoluzione del Caos: Matrici di Confusione (Target Flipping)', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "coerenza_05_matrici_grid.png"), dpi=300, bbox_inches='tight')