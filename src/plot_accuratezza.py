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
os.makedirs(plots_dir, exist_ok=True) # Crea la cartella se non esiste

path_json = os.path.join(results_dir, "risultati_accuratezza.json")
path_json_base = os.path.join(results_dir, "baseline_metrics.json") # Assicurati che punti a results_dir

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

# PRELIEVO METRICHE MEDIATE DALLA BASELINE A 10 RUN
baseline_acc = dati_base["accuracy"]
baseline_conf = dati_base["confidenza"] * 100
baseline_time = dati_base["time"]
cm_baseline = dati_base["cm"] # Matrice di confusione base (0%)

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
axes1[0].set_xlabel('Percentuale di record corrotti (%)')
axes1[0].set_ylabel('Accuracy')
axes1[0].set_xticks(perc_labels)
axes1[0].legend()

delta_outliers = [(baseline_acc - m) * 100 for m in risultati['Outliers']['media']]
delta_gauss = [(baseline_acc - m) * 100 for m in risultati['Rumore_Gaussiano']['media']]
x = np.arange(len(perc_labels))
width = 0.35

axes1[1].bar(x - width/2, delta_outliers, width, label='Perdita Outliers', color='#17becf', alpha=0.85)
axes1[1].bar(x + width/2, delta_gauss, width, label='Perdita Rumore Gaussiano', color='#1f77b4', alpha=0.85)
axes1[1].set_title("Calo di Performance Netta (Delta %)", fontsize=14, fontweight='bold')
axes1[1].set_xlabel('Percentuale di record corrotti (%)')
axes1[1].set_ylabel('Perdita (%)')
axes1[1].set_xticks(x)
axes1[1].set_xticklabels(perc_labels)
axes1[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "accuratezza_01_performance_assoluta.png"), dpi=300)

# ------------------------------
# GRAFICO 2: GAP TRAIN VS TEST 
# ------------------------------
plt.figure(figsize=(12, 7))

# Rumore Gaussiano
plt.plot(perc_labels, risultati['Rumore_Gaussiano']['media_train'], label='Train (Rumore Gaussiano)', color='#ff7f0e', marker='o', linewidth=2.5, linestyle='--')
plt.plot(perc_labels, risultati['Rumore_Gaussiano']['media'], label='Test (Rumore Gaussiano)', color='#1f77b4', marker='s', linewidth=2.5)
plt.fill_between(perc_labels, risultati['Rumore_Gaussiano']['media'], risultati['Rumore_Gaussiano']['media_train'], color='#1f77b4', alpha=0.15)

# Outliers
plt.plot(perc_labels, risultati['Outliers']['media_train'], label='Train (Outliers)', color='#2ca02c', marker='^', linewidth=2.5, linestyle='--')
plt.plot(perc_labels, risultati['Outliers']['media'], label='Test (Outliers)', color='#17becf', marker='D', linewidth=2.5)
plt.fill_between(perc_labels, risultati['Outliers']['media'], risultati['Outliers']['media_train'], color='#17becf', alpha=0.15)

plt.title('La Forbice dell\'Overfitting: Train vs Test per Esperimento', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record corrotti (%)')
plt.ylabel('Accuratezza')
plt.legend()
plt.savefig(os.path.join(plots_dir, "accuratezza_02_train_test_gap.png"), dpi=300)

# ---------------------------------
# GRAFICO 3: CONFIDENZA PREDITTIVA
# ---------------------------------
plt.figure(figsize=(10, 5))
plt.plot(perc_labels, [c * 100 for c in risultati['Rumore_Gaussiano']['confidenza']], label='Rumore Gaussiano', color='purple', marker='o', linewidth=2.5)
plt.plot(perc_labels, [c * 100 for c in risultati['Outliers']['confidenza']], label='Outliers', color='teal', marker='s', linewidth=2.5)

# LINEA DELLA BASELINE AL POSTO DEL 70% FISSO
plt.axhline(y=baseline_conf, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_conf:.1f}%)')

plt.title('Il Crollo delle Certezze: Confidenza Predittiva', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record corrotti (%)')
plt.ylabel('Probabilità media delle previsioni (%)')
plt.legend()
plt.savefig(os.path.join(plots_dir, "accuratezza_03_confidenza.png"), dpi=300)

# ---------------------------------
# GRAFICO 4: COSTO COMPUTAZIONALE
# ---------------------------------
plt.figure(figsize=(10, 5))
plt.plot(perc_labels, risultati['Rumore_Gaussiano']['tempi'], label='Rumore Gaussiano', color='orange', marker='o', linewidth=2.5)
plt.plot(perc_labels, risultati['Outliers']['tempi'], label='Outliers', color='blue', marker='s', linewidth=2.5)

# LINEA DELLA BASELINE DEI TEMPI
plt.axhline(y=baseline_time, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_time:.1f} s)')

plt.title('Impatto Computazionale: Tempi di Addestramento', fontsize=14, fontweight='bold')
plt.xlabel('Percentuale di record corrotti (%)')
plt.ylabel('Tempo impiegato (secondi)')
plt.legend()
plt.savefig(os.path.join(plots_dir, "accuratezza_04_tempi_calcolo.png"), dpi=300)

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
            
        plt.suptitle(f'Evoluzione del Caos: Matrici di Confusione ({exp_title})', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')

# Genera la griglia per Outliers (ho scelto verde per staccarlo dagli altri)
plot_matrici_confusione('Outliers', 'Outliers', 'Greens', "accuratezza_05_matrici_outliers.png", cm_baseline)

# Genera la griglia per Rumore Gaussiano (in blu)
plot_matrici_confusione('Rumore_Gaussiano', 'Rumore Gaussiano', 'Blues', "accuratezza_06_matrici_gaussiano.png", cm_baseline)

print(f"Tutti i grafici di Accuratezza sono stati generati e salvati in: {plots_dir}")