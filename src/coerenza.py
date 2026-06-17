import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Importiamo le funzioni dal file principale (assicurati che sia aggiornato con il dict di ritorno!)
from baseline_data_quality import get_train_test_split, get_or_compute_baseline, train_and_evaluate_mlp

# ==========================================
# FUNZIONI DI DEGRADO (COERENZA)
# ==========================================

def sporca_coerenza_target(y, percentage):
    """
    Esperimento Target Flipping
    Inverte l'etichetta del vincitore.
    """
    y_dirty = y.copy()
    n_changes = int(len(y_dirty) * percentage)
    idx = np.random.choice(y_dirty.index, n_changes, replace=False)
    y_dirty.loc[idx] = 1 - y_dirty.loc[idx] 
    return y_dirty

def sporca_coerenza_elo_ranking(X, percentage):
    """
    Esperimento Feature Contradiction
    Inverte il segno del Ranking ATP creando contraddizione logica.
    """
    X_dirty = X.copy()
    n_changes = int(len(X_dirty) * percentage)
    idx = np.random.choice(X_dirty.index, n_changes, replace=False)
    X_dirty.loc[idx, 'ATP_RANK_DIFF'] = -X_dirty.loc[idx, 'ATP_RANK_DIFF']
    return X_dirty


# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================

if __name__ == "__main__":
    print("--- INIZIO PIPELINE: ESPERIMENTI DI COERENZA ---")
    
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
        'Target_Flipping': {'media': [], 'std': [], 'media_train': [], 'confidenza': [], 'tempi': [], 'cm_per_step': {}, 'p1_win_pct': [], 'p2_win_pct': []},
        'Elo_Ranking': {'media': [], 'std': [], 'media_train': [], 'confidenza': [], 'tempi': [], 'cm_per_step': {}, 'p1_win_pct': [], 'p2_win_pct': []}
    }
    
    # --------------------------------
    # ESPERIMENTO A: TARGET FLIPPING
    # --------------------------------
    print(f"--- AVVIO ESPERIMENTO A: TARGET FLIPPING | {N_RUNS} RUNS ---")
    for p in percentuali:
        r_acc_test, r_acc_train, r_conf, r_time = [], [], [], []
        
        for run in tqdm(range(N_RUNS), desc=f"Degrado {p*100:0.0f}%", colour='red'):
            y_stale = sporca_coerenza_target(y_train_clean, percentage=p)
            
            # Salvataggio dei dati sporchi (Solo prima run) per PyDeequ
            if run == 0:
                df_to_save = X_train_clean.copy()
                df_to_save['target'] = y_stale
                df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_coerenza_targetflip_{int(p*100)}pct.csv"), index=False)
            
            # Addestramento ed estrazione dizionario metriche
            metrics = train_and_evaluate_mlp(X_train_clean, y_stale, X_test_clean, y_test_clean)
            
            r_acc_test.append(metrics['acc_test'])
            r_acc_train.append(metrics['acc_train'])
            r_conf.append(metrics['confidenza'])
            r_time.append(metrics['time'])
            
            # Salviamo la Matrice di confusione solo della prima run di questo step
            if run == 0:
                risultati['Target_Flipping']['cm_per_step'][str(p)] = metrics['cm']
                
        # Aggregazione Statistica
        risultati['Target_Flipping']['media'].append(float(np.mean(r_acc_test)))
        risultati['Target_Flipping']['std'].append(float(np.std(r_acc_test)))
        risultati['Target_Flipping']['media_train'].append(float(np.mean(r_acc_train)))
        risultati['Target_Flipping']['confidenza'].append(float(np.mean(r_conf)))
        risultati['Target_Flipping']['tempi'].append(float(np.mean(r_time)))
        risultati['Target_Flipping']['p1_win_pct'].append(float((y_stale == 1).mean() * 100))
        risultati['Target_Flipping']['p2_win_pct'].append(float((y_stale == 0).mean() * 100))

    # -----------------------------------------
    # ESPERIMENTO B: CONTRADDIZIONE DOMINIO 
    # -----------------------------------------
    print(f"\n--- AVVIO ESPERIMENTO B: ELO VS RANKING | {N_RUNS} RUNS ---")
    for p in percentuali:
        r_acc_test, r_acc_train, r_conf, r_time = [], [], [], []
        
        for run in tqdm(range(N_RUNS), desc=f"Degrado {p*100:0.0f}%", colour='blue'):
            X_stale = sporca_coerenza_elo_ranking(X_train_clean, percentage=p)
            
            if run == 0:
                df_to_save = X_stale.copy()
                df_to_save['target'] = y_train_clean
                df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_coerenza_elorank_{int(p*100)}pct.csv"), index=False)
            
            metrics = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
            
            r_acc_test.append(metrics['acc_test'])
            r_acc_train.append(metrics['acc_train'])
            r_conf.append(metrics['confidenza'])
            r_time.append(metrics['time'])
            
            if run == 0:
                risultati['Elo_Ranking']['cm_per_step'][str(p)] = metrics['cm']
                
        # Aggregazione Statistica
        risultati['Elo_Ranking']['media'].append(float(np.mean(r_acc_test)))
        risultati['Elo_Ranking']['std'].append(float(np.std(r_acc_test)))
        risultati['Elo_Ranking']['media_train'].append(float(np.mean(r_acc_train)))
        risultati['Elo_Ranking']['confidenza'].append(float(np.mean(r_conf)))
        risultati['Elo_Ranking']['tempi'].append(float(np.mean(r_time)))
        risultati['Elo_Ranking']['p1_win_pct'].append(float((y_train_clean == 1).mean() * 100))
        risultati['Elo_Ranking']['p2_win_pct'].append(float((y_train_clean == 0).mean() * 100))

    # ==========================================
    # SALVATAGGIO JSON
    # ==========================================
    dati_da_salvare = {
        "baseline_acc": float(baseline_acc),
        "percentuali": [float(p) for p in percentuali],
        "risultati": risultati
    }
    
    path_json = os.path.join(results_dir, "risultati_coerenza.json")
    with open(path_json, "w") as f:
        json.dump(dati_da_salvare, f, indent=4)
        
    print(f"\nEsperimenti Coerenza terminati. Dati salvati in: {path_json}")