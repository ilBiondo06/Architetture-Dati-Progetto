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
os.makedirs(plots_dir, exist_ok=True) 

path_json = os.path.join(results_dir, "risultati_accuratezza.json")

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

# ----------------------------------------
# GRAFICO 1: ACCURATEZZA CLASSICA E DELTA
# ----------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

axes1[0].errorbar(perc_labels, risultati['Outliers']['media'], yerr=risultati['Outliers']['std'], fmt='-o', label='Outliers (Altezza/Età)', color='#17becf', capsize=5, linewidth=2.5)
axes1[0].errorbar(perc_labels, risultati['Rumore_Gaussiano']['media'], yerr=risultati['Rumore_Gaussiano']['std'], fmt='-s', label='Rumore Gaussiano (Elo)', color='#1f77b4', capsize=5, linewidth=2.5)
axes1[0].axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.4f})')
axes1[0].set_title('Accuratezza: Impatto su Performance Assoluta', fontsize=14, fontweight='bold')
axes1[0].set_xlabel('Percentuale di record corrotti (%)'); axes1[0].set_ylabel('Accuracy')
axes1[0].set_xticks(perc_labels); axes1[0].legend()

delta_outliers = [(baseline_acc - m) * 100 for m in risultati['Outliers']['media']]
delta_gauss = [(baseline_acc - m) * 100 for m in risultati['Rumore_Gaussiano']['media']]
x = np.arange(len(perc_labels)); width = 0.35

axes1[1].bar(x - width/2, delta_outliers, width, label='Perdita Outliers', color='#17becf', alpha=0.85)
axes1[1].bar(x + width/2, delta_gauss, width, label='Perdita Rumore Gaussiano', color='#1f77b4', alpha=0.85)
axes1[1].set_title("Calo di Performance Netta (Delta %)", fontsize=14, fontweight='bold')
axes1[1].set_xlabel('Percentuale di record corrotti (%)'); axes1[1].set_ylabel('Perdita (%)')
axes1[1].set_xticks(x); axes1[1].set_xticklabels(perc_labels); axes1[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "accuratezza_01_performance_assoluta.png"), dpi=300)

# ------------------------------
# GRAFICO 2: GAP TRAIN VS TEST 
# ------------------------------
plt.figure(figsize=(10, 6))
plt.plot(perc_labels, risultati['Rumore_Gaussiano']['media_train'], label='Accuracy sul TRAIN', color='#ff7f0e', marker='o', linewidth=2.5)
plt.plot(perc_labels, risultati['Rumore_Gaussiano']['media'], label='Accuracy sul TEST (Realtà)', color='#1f77b4', marker='s', linewidth=2.5)
plt.fill_between(perc_labels, risultati['Rumore_Gaussiano']['media'], risultati['Rumore_Gaussiano']['media_train'], color='gray', alpha=0.2, label='Generalization Gap')

plt.title('La Forbice dell\'Overfitting: Train vs Test (Rumore Gaussiano)', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record corrotti (%)'); plt.ylabel('Accuratezza')
plt.legend()
plt.savefig(os.path.join(plots_dir, "accuratezza_02_train_test_gap.png"), dpi=300)

# ---------------------------------
# GRAFICO 3: CONFIDENZA PREDITTIVA
# ---------------------------------
plt.figure(figsize=(10, 5))
plt.plot(perc_labels, [c * 100 for c in risultati['Rumore_Gaussiano']['confidenza']], label='Confidenza (Rumore Gaussiano)', color='purple', marker='^', linewidth=2.5)
plt.axhline(y=75.0, color='gray', linestyle='--', label='Soglia Teorica di Rischio (75%)')

plt.title('Il Crollo delle Certezze: Confidenza Predittiva', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record corrotti (%)'); plt.ylabel('Probabilità media delle previsioni (%)')
plt.legend()
plt.savefig(os.path.join(plots_dir, "accuratezza_03_confidenza.png"), dpi=300)

# ---------------------------------
# GRAFICO 4: COSTO COMPUTAZIONALE
# ---------------------------------
plt.figure(figsize=(9, 5))
plt.bar(perc_labels, risultati['Rumore_Gaussiano']['tempi'], color='orange', edgecolor='black', alpha=0.8)

plt.title('Impatto Computazionale: Tempi di Addestramento', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record corrotti (%)'); plt.ylabel('Tempo impiegato (secondi)')
plt.savefig(os.path.join(plots_dir, "accuratezza_04_tempi_calcolo.png"), dpi=300)

# --------------------------------------------
# GRAFICO 5: EVOLUZIONE MATRICE DI CONFUSIONE 
# --------------------------------------------
cm_dict = risultati['Rumore_Gaussiano'].get('cm_per_step')

if cm_dict is not None and len(cm_dict) > 0:
    fig5, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    sorted_keys = sorted(cm_dict.keys(), key=float)
    
    for i, p_str in enumerate(sorted_keys):
        cm = np.array(cm_dict[p_str])
        perc_int = int(float(p_str) * 100)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Sconfitta', 'Vittoria'],
                    yticklabels=['Sconfitta', 'Vittoria'], ax=axes[i])
        
        axes[i].set_title(f"Degrado: {perc_int}%", fontweight='bold')
        axes[i].set_xlabel('Predetto'); axes[i].set_ylabel('Vero')
    
    for j in range(i + 1, len(axes)):
        fig5.delaxes(axes[j])
        
    plt.suptitle('Evoluzione del Caos: Matrici di Confusione (Rumore Gaussiano)', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "accuratezza_05_matrici_grid.png"), dpi=300, bbox_inches='tight')
