import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from baseline_data_quality import train_and_evaluate_mlp, get_train_test_split, get_or_compute_baseline

# ==========================================
# FUNZIONI DI DEGRADO (COMPLETEZZA E TEMPESTIVITÀ)
# ==========================================

def missing_values_zero_systemic(X, percentage):
    """
    ESPERIMENTO A: Completezza (Blackout Sistemico)
    Simula la rottura del sistema che calcola l'Elo azzerando
    tutte le colonne correlate simultaneamente.
    """
    X_dirty = X.copy()
    n_changes = int(len(X_dirty) * percentage)
    
    elo_cols = [col for col in X_dirty.columns if 'elo' in col.lower()]
    idx = np.random.choice(X_dirty.index, n_changes, replace=False)
    
    for col in elo_cols:
        X_dirty.loc[idx, col] = 0.0
        
    return X_dirty


def temporaly_lag(X, percentage, lag=5):
    """
    ESPERIMENTO B: Lag temporaneo (Data Staleness)
    Raggruppa lo storico per singolo giocatore (p1_id) e usa le sue 
    metriche Elo vecchie di 'lag' partite.
    """
    X_dirty = X.copy()
    n_changes = int(len(X_dirty) * percentage)
    
    elo_cols = [col for col in X_dirty.columns if 'elo' in col.lower()]
    idx = np.random.choice(X_dirty.index, n_changes, replace=False)
    
    for col in elo_cols:
        stale_values = X_dirty.groupby('p1_id')[col].shift(lag).bfill()
        X_dirty.loc[idx, col] = stale_values.loc[idx]
        
    return X_dirty


# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================

if __name__ == "__main__":
    print("--- INIZIO PIPELINE: ESPERIMENTI COMPLETEZZA E TEMPESTIVITÀ ---")
    
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
        'Systemic_Missing': {'media': [], 'std': [], 'media_train': [], 'confidenza': [], 'tempi': [], 'cm_per_step': {}},
        'Lag_10': {'media': [], 'std': [], 'media_train': [], 'confidenza': [], 'tempi': [], 'cm_per_step': {}}
    }

    # -----------------------------------------------------------------
    # ESPERIMENTO A: BLACKOUT ELO (COMPLETEZZA)
    # -----------------------------------------------------------------
    print(f"--- AVVIO ESPERIMENTO A: BLACKOUT SISTEMICO | {N_RUNS} RUNS ---")
    for p in percentuali:
        r_acc_test, r_acc_train, r_conf, r_time = [], [], [], []
        
        for run in tqdm(range(N_RUNS), desc=f"Degrado {p*100:0.0f}%", colour='red'):
            X_stale = missing_values_zero_systemic(X_train_clean, percentage=p)
            
            # Salvataggio dati sporchi per PyDeequ (solo run 0)
            if run == 0:
                df_to_save = X_stale.copy()
                df_to_save['target'] = y_train_clean
                df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_completezza_blackout_{int(p*100)}pct.csv"), index=False)
            
            # Addestramento e metriche
            metrics = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
            
            r_acc_test.append(metrics['acc_test'])
            r_acc_train.append(metrics['acc_train'])
            r_conf.append(metrics['confidenza'])
            r_time.append(metrics['time'])
            
            if run == 0:
                risultati['Systemic_Missing']['cm_per_step'][str(p)] = metrics['cm']
                
        # Aggregazione Statistica
        risultati['Systemic_Missing']['media'].append(float(np.mean(r_acc_test)))
        risultati['Systemic_Missing']['std'].append(float(np.std(r_acc_test)))
        risultati['Systemic_Missing']['media_train'].append(float(np.mean(r_acc_train)))
        risultati['Systemic_Missing']['confidenza'].append(float(np.mean(r_conf)))
        risultati['Systemic_Missing']['tempi'].append(float(np.mean(r_time)))

    # -----------------------------------------------------------------
    # ESPERIMENTO B: TEMPORALY LAG (TEMPESTIVITÀ)
    # -----------------------------------------------------------------
    print(f"\n--- AVVIO ESPERIMENTO B: DATI OBSOLETI (LAG-10) | {N_RUNS} RUNS ---")
    for p in percentuali:
        r_acc_test, r_acc_train, r_conf, r_time = [], [], [], []
        
        for run in tqdm(range(N_RUNS), desc=f"Degrado {p*100:0.0f}%", colour='yellow'):
            X_stale = temporaly_lag(X_train_clean, percentage=p, lag=10)
            
            if run == 0:
                df_to_save = X_stale.copy()
                df_to_save['target'] = y_train_clean
                df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_tempestivita_lag10_{int(p*100)}pct.csv"), index=False)
            
            metrics = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
            
            r_acc_test.append(metrics['acc_test'])
            r_acc_train.append(metrics['acc_train'])
            r_conf.append(metrics['confidenza'])
            r_time.append(metrics['time'])
            
            if run == 0:
                risultati['Lag_10']['cm_per_step'][str(p)] = metrics['cm']
                
        # Aggregazione Statistica
        risultati['Lag_10']['media'].append(float(np.mean(r_acc_test)))
        risultati['Lag_10']['std'].append(float(np.std(r_acc_test)))
        risultati['Lag_10']['media_train'].append(float(np.mean(r_acc_train)))
        risultati['Lag_10']['confidenza'].append(float(np.mean(r_conf)))
        risultati['Lag_10']['tempi'].append(float(np.mean(r_time)))

    # ==========================================
    # SALVATAGGIO JSON
    # ==========================================
    dati_da_salvare = {
        "baseline_acc": float(baseline_acc),
        "percentuali": [float(p) for p in percentuali],
        "risultati": risultati
    }
    
    path_json = os.path.join(results_dir, "risultati_completezza.json")
    with open(path_json, "w") as f:
        json.dump(dati_da_salvare, f, indent=4)
        
    print(f"\nEsperimenti Completezza terminati. Dati salvati in: {path_json}")