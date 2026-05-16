import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from baseline_data_quality import train_and_evaluate_mlp, get_train_test_split, get_or_compute_baseline

def introduci_missing_values(X, percentage):
    """
    Simula la mancanza di dati inserendo NaN casuali.
    - percentage: percentuale di celle totali da svuotare.
    """
    X_dirty = X.copy()
    mask = np.random.rand(*X_dirty.shape) < percentage
    X_dirty[mask] = np.nan
    
    print(f"Svuotato il {percentage*100}% delle celle totali del dataset.")
    return X_dirty

def sporca_elo_last(X, percentage, lag=25):
    """
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
        stale_values = X_dirty[col].shift(lag).bfill()
        
        # Scegliamo gli indici casuali da sporcare
        idx = np.random.choice(X_dirty.index, n_changes, replace=False)
        
        # Sostituiamo i valori reali con quelli obsoleti
        X_dirty.loc[idx, col] = stale_values.loc[idx]
        
    return X_dirty

if __name__ == "__main__":
    print("--- INIZIO PIPELINE DATA QUALITY: ESPERIMENTI ---")
    
    # Calcolo dinamico del path per il dataset (stessa logica robusta usata in baseline)
    base_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_folder, "..", "data", "dataset_ml_ready.csv") 
    
    try:
        baseline_acc = get_or_compute_baseline(dataset_path)
        print(f"Accuratezza Baseline di riferimento: {baseline_acc:.4f}\n")
        
        X_train_clean, y_train_clean, X_test_clean, y_test_clean = get_train_test_split(dataset_path)
        
        percentuali = [0.05, 0.10, 0.20, 0.30, 0.50]  # Percentuali da testare 
        N_RUNS = 10  # Numero di esperimenti 
        
        risultati_media = []
        risultati_std = [] # Deviazione standard 
        
        print(f"--- AVVIO LOOP DI DEGRADO (TIMELINESS - ELO LAG 25) | {N_RUNS} RUNS PER STEP ---")
        
        for p in percentuali:
            if p == 0.0:
                media_acc = baseline_acc
                std_acc = 0.0
            else:
                run_accuracies = [] # Qui salviamo i risultati delle singole run
                
                for run in tqdm(range(N_RUNS), desc=f"Progresso esperimento con degrado {p*100:0.0f}%", colour='green'):
                    
                    X_stale = sporca_elo_last(X_train_clean, percentage=p, lag=10)
                    acc = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
                    run_accuracies.append(acc)
                
                # Statistiche finali per questa percentuale
                media_acc = np.mean(run_accuracies)
                std_acc = np.std(run_accuracies)
                
                diff = baseline_acc - media_acc
                print(f"  -> Accuracy Media: {media_acc:.4f} ± {std_acc:.4f} | Perdita: {(diff*100):.2f}%")
            
            # Salviamo per il report finale
            risultati_media.append(media_acc)
            risultati_std.append(std_acc)
            
        # 4. Tabella riassuntiva dei log
        print("\n" + "="*60)
        print("REPORT FINALE ESPERIMENTO (MEDIA SU 10 RUNS): TIMELINESS")
        print("="*60)
        for i, p in enumerate(percentuali):
            taglio_performance = (baseline_acc - risultati_media[i]) * 100
            print(f"Degrado {p*100:02.0f}% | Test Accuracy: {risultati_media[i]:.4f} ± {risultati_std[i]:.4f} | Delta: -{taglio_performance:.2f}%")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\nERRORE: file {dataset_path} non trovato")
    except Exception as e:
        print(f"\nErrore durante l'esecuzione: {e}")