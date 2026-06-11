import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from baseline_data_quality import train_and_evaluate_mlp, get_train_test_split, get_or_compute_baseline, load_baseline_cache

# ==========================================
# FUNZIONI DI DEGRADO (COMPLETEZZA E TEMPESTIVITÀ)
# ==========================================

def missing_values_zero_systemic(X, percentage):
    """
    ESPERIMENTO A: Completezza (Il Vero Blackout Sistemico)
    Simula la rottura del sistema che calcola l'Elo. 
    TUTTE le colonne relative all'Elo vengono azzerate per i record colpiti,
    impedendo alla rete neurale di usare la collinearità.
    """
    X_dirty = X.copy()
    n_changes = int(len(X_dirty) * percentage)
    
    # Selezioniamo TUTTE le colonne che contengono 'elo'
    elo_cols = [col for col in X_dirty.columns if 'elo' in col.lower()]
    
    # Scegliamo le stesse identiche righe per tutte le colonne (Systemic Failure)
    idx = np.random.choice(X_dirty.index, n_changes, replace=False)
    
    for col in elo_cols:
        X_dirty.loc[idx, col] = 0.0
        
    return X_dirty


def sporca_elo_last(X, percentage, lag=25):
    """
    ESPERIMENTO B: Tempestività (Data Staleness)
    [RECUPERATO DAL TUO CODICE ORIGINALE]
    Simula un ritardo nell'aggiornamento dell'Elo.
    Sostituisce il valore attuale con quello di 'lag' posizioni precedenti.
    """
    X_dirty = X.copy()
    n_rows = len(X_dirty)
    n_changes = int(n_rows * percentage)
    
    # Selezioniamo le colonne che contengono 'elo'
    elo_cols = [col for col in X_dirty.columns if 'elo' in col.lower()]
    
    for col in elo_cols:
        # Spostiamo i dati indietro del valore di lag e riempiamo i primi buchi in alto
        # (Usiamo bfill() per evitare i warning del vecchio method='bfill')
        stale_values = X_dirty[col].shift(lag).bfill()
        
        idx = np.random.choice(X_dirty.index, n_changes, replace=False)
        X_dirty.loc[idx, col] = stale_values.loc[idx]
        
    return X_dirty


# ==========================================
# ESECUZIONE PRINCIPALE E GRAFICI
# ==========================================

if __name__ == "__main__":
    print("--- INIZIO PIPELINE: ESPERIMENTI DI COMPLETEZZA E TEMPESTIVITÀ ---")
    
    base_folder = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(base_folder, "..", "data", "clean")
    dirty_dir = os.path.join(base_folder, "..", "data", "dirty")
    plots_dir = os.path.join(base_folder, "..", "docs", "plots")
    
    os.makedirs(dirty_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    dataset_path = os.path.join(clean_dir, "dataset_ml_ready.csv") 
    
    try:
        baseline_acc = get_or_compute_baseline(dataset_path)
        baseline_cache = load_baseline_cache()
        if baseline_cache:
            loss_baseline = baseline_cache["loss_curve"]
            epoche_baseline = baseline_cache["n_iter"]
            print(f"Baseline caricata da JSON: {baseline_acc:.4f} (Convergenza in {epoche_baseline} epoche).\n")
        else:
            loss_baseline, epoche_baseline = [], 0
            
        X_train_clean, y_train_clean, X_test_clean, y_test_clean = get_train_test_split(dataset_path)
        
        percentuali = [0.05, 0.10, 0.20, 0.30, 0.50]  
        N_RUNS = 10  
        risultati = {'Systemic_Missing': {'media': [], 'std': []}, 'Lag_25': {'media': [], 'std': []}}
        
        loss_target_50 = None
        epoche_target_50 = 0

        # -----------------------------------------------------------------
        # ESPERIMENTO A: BLACKOUT SISTEMICO (COMPLETEZZA)
        # -----------------------------------------------------------------
        print(f"--- AVVIO ESPERIMENTO A: BLACKOUT ELO (COMPLETEZZA) | {N_RUNS} RUNS ---")
        for p in percentuali:
            run_accuracies = [] 
            for run in tqdm(range(N_RUNS), desc=f"Blackout {p*100:0.0f}%", colour='red'):
                X_stale = missing_values_zero_systemic(X_train_clean, percentage=p)
                
                if run == 0:
                    df_to_save = X_stale.copy()
                    df_to_save['target'] = y_train_clean
                    df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_completezza_blackout_{int(p*100)}pct.csv"), index=False)
                    
                    if p == 0.50:
                        acc, mlp_dirty = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean, return_model=True)
                        loss_target_50 = mlp_dirty.loss_curve_
                        epoche_target_50 = mlp_dirty.n_iter_
                    else:
                        acc = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
                else:
                    acc = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
                run_accuracies.append(acc)
            
            m, s = np.mean(run_accuracies), np.std(run_accuracies)
            risultati['Systemic_Missing']['media'].append(m); risultati['Systemic_Missing']['std'].append(s)
            print(f"  -> Accuracy: {m:.4f} ± {s:.4f} | Delta: -{(baseline_acc - m)*100:.2f}%\n")

        # -----------------------------------------------------------------
        # ESPERIMENTO B: DATA STALENESS LAG-25 (TEMPESTIVITÀ)
        # -----------------------------------------------------------------
        print(f"--- AVVIO ESPERIMENTO B: DATI VECCHI LAG-25 (TEMPESTIVITÀ) | {N_RUNS} RUNS ---")
        for p in percentuali:
            run_accuracies = [] 
            for run in tqdm(range(N_RUNS), desc=f"Lag-25 {p*100:0.0f}%", colour='yellow'):
                X_stale = sporca_elo_last(X_train_clean, percentage=p, lag=25)
                
                if run == 0:
                    df_to_save = X_stale.copy()
                    df_to_save['target'] = y_train_clean
                    df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_tempestivita_lag25_{int(p*100)}pct.csv"), index=False)
                
                acc = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
                run_accuracies.append(acc)
            
            m, s = np.mean(run_accuracies), np.std(run_accuracies)
            risultati['Lag_25']['media'].append(m); risultati['Lag_25']['std'].append(s)
            print(f"  -> Accuracy: {m:.4f} ± {s:.4f} | Delta: -{(baseline_acc - m)*100:.2f}%\n")

        # ================= SALVATAGGIO RISULTATI =================
        import json
        
        # Pulizia dei dati NumPy per renderli salvabili in JSON
        risultati_standardizzati = {
            'Systemic_Missing': {
                'media': [float(m) for m in risultati['Systemic_Missing']['media']],
                'std': [float(s) for s in risultati['Systemic_Missing']['std']]
            },
            'Lag_25': {
                'media': [float(m) for m in risultati['Lag_25']['media']],
                'std': [float(s) for s in risultati['Lag_25']['std']]
            }
        }
        
        dati_da_salvare = {
            "baseline_acc": float(baseline_acc),
            "loss_baseline": [float(x) for x in loss_baseline] if hasattr(loss_baseline, "__iter__") else [],
            "epoche_baseline": int(epoche_baseline),
            "percentuali": [float(p) for p in percentuali],
            "risultati": risultati_standardizzati,
            "loss_target_50": [float(x) for x in loss_target_50] if loss_target_50 is not None else None,
            "epoche_target_50": int(epoche_target_50)
        }
        
        path_cache_risultati = os.path.join(plots_dir, "risultati_completezza.json")
        with open(path_cache_risultati, "w") as f:
            json.dump(dati_da_salvare, f, indent=4)
        print(f"\n[OK] Risultati salvati correttamente in: {path_cache_risultati}")
        print("Usa 'plot_completezza.py' per visualizzare e modificare i grafici.")

    except Exception as e:
        print(f"\nErrore: {e}")