import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from baseline_data_quality import get_train_test_split, get_or_compute_baseline, train_and_evaluate_mlp

# ==========================================
# FUNZIONI DI DEGRADO (ACCURATEZZA)
# ==========================================

def eta_altezza_outliers(X, percentage):
    X_dirty = X.copy()
    n_changes = int(len(X_dirty) * percentage)
    
    idx = np.random.choice(X_dirty.index, n_changes, replace=False)
    
    # creiamo array di valori assurdi casuali
    # moltiplicati per 1 o -1 per avere sia giganti che altezze negative
    sbalzo_altezza = np.random.uniform(100.0, 200.0, size=n_changes) * np.random.choice([1, -1], size=n_changes)
    sbalzo_eta = np.random.uniform(50.0, 100.0, size=n_changes) * np.random.choice([1, -1], size=n_changes)
    
    X_dirty.loc[idx, 'diff_ht'] = sbalzo_altezza
    X_dirty.loc[idx, 'diff_age'] = sbalzo_eta
    
    return X_dirty

def rumore_gaussiano(X, percentage):
    """
    ESPERIMENTO B: Rumore Gaussiano
    Applica una distorsione statistica (Gaussian Noise) alla colonna DIFF_ELO.
    """
    X_dirty = X.copy()
    n_changes = int(len(X_dirty) * percentage)
    
    idx = np.random.choice(X_dirty.index, n_changes, replace=False)
    if 'DIFF_ELO' in X_dirty.columns:
        std_dev = X_dirty['DIFF_ELO'].std()
        noise = np.random.normal(0, std_dev * 2, n_changes)
        X_dirty.loc[idx, 'DIFF_ELO'] += noise
        
    return X_dirty

# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================

if __name__ == "__main__":
    print("--- INIZIO PIPELINE: ESPERIMENTI DI ACCURATEZZA---")
    
    # 1. Configurazione Percorsi
    base_folder = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(base_folder, "..", "data", "clean")
    dirty_dir = os.path.join(base_folder, "..", "data", "dirty")
    results_dir = os.path.join(base_folder, "..", "results") 
    
    os.makedirs(dirty_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    dataset_path = os.path.join(clean_dir, "dataset_ml_ready.csv") 
    
    # 2. Caricamento Baseline
    baseline_acc = get_or_compute_baseline(dataset_path)
    
    X_train_clean, y_train_clean, X_test_clean, y_test_clean = get_train_test_split(dataset_path)
    
    percentuali = [0.10, 0.20, 0.40, 0.60, 0.80]  
    N_RUNS = 10  
    
    # 3. Struttura dati arricchita
    risultati = {
        'Outliers': {'media': [], 'std': [], 'media_train': [], 'confidenza': [], 'tempi': [], 'cm_per_step': {}},
        'Rumore_Gaussiano': {'media': [], 'std': [], 'media_train': [], 'confidenza': [], 'tempi': [], 'cm_per_step': {}}
    }
    
    # -------------------------------------
    # ESPERIMENTO A: OUTLIERS ETÀ-ALTEZZA
    # -------------------------------------
    print(f"--- AVVIO ESPERIMENTO A: OUTLIERS ETÀ-ALTEZZA | {N_RUNS} RUNS ---")
    for p in percentuali:
        r_acc_test, r_acc_train, r_conf, r_time = [], [], [], []
        
        for run in tqdm(range(N_RUNS), desc=f"Degrado {p*100:0.0f}%", colour='cyan'):
            X_stale = eta_altezza_outliers(X_train_clean, percentage=p)
            
            # Salvataggio dati sporchi per PyDeequ (solo run 0)
            if run == 0:
                df_to_save = X_stale.copy()
                df_to_save['target'] = y_train_clean
                df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_accuratezza_outliers_{int(p*100)}pct.csv"), index=False)
            
            # Addestramento e metriche
            metrics = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
            
            r_acc_test.append(metrics['acc_test'])
            r_acc_train.append(metrics['acc_train'])
            r_conf.append(metrics['confidenza'])
            r_time.append(metrics['time'])
            
            if run == 0:
                risultati['Outliers']['cm_per_step'][str(p)] = metrics['cm']
                
        # Aggregazione Statistica
        risultati['Outliers']['media'].append(float(np.mean(r_acc_test)))
        risultati['Outliers']['std'].append(float(np.std(r_acc_test)))
        risultati['Outliers']['media_train'].append(float(np.mean(r_acc_train)))
        risultati['Outliers']['confidenza'].append(float(np.mean(r_conf)))
        risultati['Outliers']['tempi'].append(float(np.mean(r_time)))

    # ----------------------------------
    # ESPERIMENTO B: RUMORE GAUSSIANO 
    # ---------------------------------
    print(f"\n--- AVVIO ESPERIMENTO B: RUMORE GAUSSIANO | {N_RUNS} RUNS ---")
    for p in percentuali:
        r_acc_test, r_acc_train, r_conf, r_time = [], [], [], []
        
        for run in tqdm(range(N_RUNS), desc=f"Degrado {p*100:0.0f}%", colour='blue'):
            X_stale = rumore_gaussiano(X_train_clean, percentage=p)
            
            if run == 0:
                df_to_save = X_stale.copy()
                df_to_save['target'] = y_train_clean
                df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_accuratezza_gaussiano_{int(p*100)}pct.csv"), index=False)
            
            metrics = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
            
            r_acc_test.append(metrics['acc_test'])
            r_acc_train.append(metrics['acc_train'])
            r_conf.append(metrics['confidenza'])
            r_time.append(metrics['time'])
            
            if run == 0:
                risultati['Rumore_Gaussiano']['cm_per_step'][str(p)] = metrics['cm']
                
        # Aggregazione Statistica
        risultati['Rumore_Gaussiano']['media'].append(float(np.mean(r_acc_test)))
        risultati['Rumore_Gaussiano']['std'].append(float(np.std(r_acc_test)))
        risultati['Rumore_Gaussiano']['media_train'].append(float(np.mean(r_acc_train)))
        risultati['Rumore_Gaussiano']['confidenza'].append(float(np.mean(r_conf)))
        risultati['Rumore_Gaussiano']['tempi'].append(float(np.mean(r_time)))

    # ==========================================
    # SALVATAGGIO JSON
    # ==========================================
    dati_da_salvare = {
        "baseline_acc": float(baseline_acc),
        "percentuali": [float(p) for p in percentuali],
        "risultati": risultati
    }
    
    path_json = os.path.join(results_dir, "risultati_accuratezza.json")
    with open(path_json, "w") as f:
        json.dump(dati_da_salvare, f, indent=4)
        
    print(f"\nEsperimenti Accuratezza terminati. Dati salvati in: {path_json}")