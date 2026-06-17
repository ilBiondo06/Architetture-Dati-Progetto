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
src_dir = os.path.join(base_folder, "..", "src")
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True) 

path_json = os.path.join(results_dir, "risultati_completezza.json")
path_json_base = os.path.join(results_dir, "baseline_metrics.json")

if not os.path.exists(path_json):
    print(f"[ERRORE] File dei dati non trovato in: {path_json}")
    exit(1)

if not os.path.exists(path_json_base):
    print(f"[ERRORE] File della baseline non trovato in: {path_json_base}")
    exit(1)

# ==========================================
# CARICAMENTO DATI DAL JSON
# ==========================================
print(f"Generazione grafici in corso da: {path_json} e {path_json_base}...")

with open(path_json, "r") as f:
    dati = json.load(f)

with open(path_json_base, "r") as f:
    dati_base = json.load(f)

baseline_acc = dati_base["accuracy"]
cm_baseline = dati_base["cm"] # Matrice di confusione base (0%)
baseline_conf = dati_base["confidenza"] * 100
percentuali = dati["percentuali"]
perc_labels = [int(p * 100) for p in percentuali]
risultati = dati["risultati"]
baseline_time = dati_base["time"]

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# GRAFICO 1: ACCURATEZZA CLASSICA E DELTA
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

axes1[0].errorbar(perc_labels, risultati['Systemic_Missing']['media'], yerr=risultati['Systemic_Missing']['std'], fmt='-o', label='Blackout (Completezza)', color='#d62728', capsize=5, linewidth=2.5)
axes1[0].errorbar(perc_labels, risultati['Lag_10']['media'], yerr=risultati['Lag_10']['std'], fmt='-s', label='Lag-10 (Tempestività)', color='#ff7f0e', capsize=5, linewidth=2.5)
axes1[0].axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.4f})')
axes1[0].set_title('Completezza e Tempestività', fontsize=14, fontweight='bold')
axes1[0].set_xlabel('Percentuale di record corrotti (%)')
axes1[0].set_ylabel('Accuracy')
axes1[0].set_xticks(perc_labels)
axes1[0].legend()

delta_missing = [(baseline_acc - m) * 100 for m in risultati['Systemic_Missing']['media']]
delta_lag = [(baseline_acc - m) * 100 for m in risultati['Lag_10']['media']]
x = np.arange(len(perc_labels))
width = 0.35

axes1[1].bar(x - width/2, delta_missing, width, label='Perdita Blackout', color='#d62728', alpha=0.85)
axes1[1].bar(x + width/2, delta_lag, width, label='Perdita Lag-10', color='#ff7f0e', alpha=0.85)
axes1[1].set_title("Calo di Performance Netta (Delta %)", fontsize=14, fontweight='bold')
axes1[1].set_xlabel('Percentuale di record corrotti (%)')
axes1[1].set_ylabel('Perdita (%)')
axes1[1].set_xticks(x)
axes1[1].set_xticklabels(perc_labels)
axes1[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "completezza_01_performance_assoluta.png"), dpi=300)

# ---------------------------------------------------------
# GRAFICO 2: GAP TRAIN VS TEST 
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

# Blackout (Systemic Missing)
plt.plot(perc_labels, risultati['Systemic_Missing']['media_train'], label='Train (Blackout)', color='#ff9896', marker='o', linewidth=2.5, linestyle='--')
plt.plot(perc_labels, risultati['Systemic_Missing']['media'], label='Test (Blackout)', color='#d62728', marker='s', linewidth=2.5)
plt.fill_between(perc_labels, risultati['Systemic_Missing']['media'], risultati['Systemic_Missing']['media_train'], color='#d62728', alpha=0.15)

# Lag-10
plt.plot(perc_labels, risultati['Lag_10']['media_train'], label='Train (Lag-10)', color='#ffbb78', marker='^', linewidth=2.5, linestyle='--')
plt.plot(perc_labels, risultati['Lag_10']['media'], label='Test (Lag-10)', color='#ff7f0e', marker='D', linewidth=2.5)
plt.fill_between(perc_labels, risultati['Lag_10']['media'], risultati['Lag_10']['media_train'], color='#ff7f0e', alpha=0.15)

plt.title('La Forbice dell\'Overfitting: Train vs Test per Esperimento', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record corrotti (%)')
plt.ylabel('Accuratezza')
plt.legend()
plt.savefig(os.path.join(plots_dir, "completezza_02_train_test_gap.png"), dpi=300)

# ---------------------------------------------------------
# GRAFICO 3: CONFIDENZA PREDITTIVA
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(perc_labels, [c * 100 for c in risultati['Systemic_Missing']['confidenza']], label='Blackout', color='#d62728', marker='o', linewidth=2.5)
plt.plot(perc_labels, [c * 100 for c in risultati['Lag_10']['confidenza']], label='Lag-10', color='#ff7f0e', marker='s', linewidth=2.5)
plt.axhline(y=baseline_conf, color='gray', linestyle='--', label=f'Baseline ({baseline_conf:.2f}%)')

plt.title('Il Crollo delle Certezze: Confidenza Predittiva', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record corrotti (%)')
plt.ylabel('Probabilità media delle previsioni (%)')
plt.legend()
plt.savefig(os.path.join(plots_dir, "completezza_03_confidenza.png"), dpi=300)

# ---------------------------------------------------------
# GRAFICO 4: COSTO COMPUTAZIONALE
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(perc_labels, risultati['Systemic_Missing']['tempi'], label='Blackout', color='#d62728', marker='o', linewidth=2.5)
plt.plot(perc_labels, risultati['Lag_10']['tempi'], label='Lag-10', color='#ff7f0e', marker='s', linewidth=2.5)
plt.axhline(y=baseline_time, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_time:.1f} s)')
plt.title('Impatto Computazionale: Tempi di Addestramento', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record corrotti (%)')
plt.ylabel('Tempo impiegato (secondi)')
plt.legend()
plt.savefig(os.path.join(plots_dir, "completezza_04_tempi_calcolo.png"), dpi=300)

# -------------------------------------------------------------
# GRAFICO 5 e 6: EVOLUZIONE MATRICE DI CONFUSIONE (DOPPIO)
# -------------------------------------------------------------
def plot_matrici_confusione(exp_key, exp_title, colormap, filename, base_cm):
    cm_dict = risultati[exp_key].get('cm_per_step')
    if cm_dict is not None and len(cm_dict) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        
        # 1. Matrice Base (dati presi da baseline_metrics.json)
        sns.heatmap(np.array(base_cm), annot=True, fmt='d', cmap=colormap, cbar=False,
                    xticklabels=['Sconfitta', 'Vittoria'],
                    yticklabels=['Sconfitta', 'Vittoria'], ax=axes[0])
        axes[0].set_title("Degrado: 0% (Baseline)", fontweight='bold')
        axes[0].set_xlabel('Predetto')
        axes[0].set_ylabel('Vero')

        # 2. Matrici di degrado
        sorted_keys = sorted(cm_dict.keys(), key=float)
        for i, p_str in enumerate(sorted_keys):
            cm = np.array(cm_dict[p_str])
            perc_int = int(float(p_str) * 100)
            
            ax_idx = i + 1 
            
            sns.heatmap(cm, annot=True, fmt='d', cmap=colormap, cbar=False,
                        xticklabels=['Sconfitta', 'Vittoria'],
                        yticklabels=['Sconfitta', 'Vittoria'], ax=axes[ax_idx])
            
            axes[ax_idx].set_title(f"Degrado: {perc_int}%", fontweight='bold')
            axes[ax_idx].set_xlabel('Predetto')
            axes[ax_idx].set_ylabel('Vero')
            
        plt.suptitle(f'Evoluzione dell\'Inganno: Matrici di Confusione ({exp_title})', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')

# Genera la griglia per Systemic Missing (in rosso)
plot_matrici_confusione('Systemic_Missing', 'Blackout', 'Reds', "completezza_05_matrici_missing.png", cm_baseline)

# Genera la griglia per Lag-10 (in arancione)
plot_matrici_confusione('Lag_10', 'Lag-10', 'Oranges', "completezza_06_matrici_lag.png", cm_baseline)

print(f"Tutti i grafici sono stati generati e salvati in: {plots_dir}")