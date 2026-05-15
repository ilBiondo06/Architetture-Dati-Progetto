import numpy as np
import pandas as pd

from baseline_data_quality import train_and_evaluate_mlp, get_train_test_split



def introduci_missing_values(X, percentage):
    """
    Simula la mancanza di dati inserendo NaN casuali.
    - percentage: percentuale di celle totali da svuotare.
    """
    X_dirty = X.copy()
    # Creiamo una maschera booleana casuale
    mask = np.random.rand(*X_dirty.shape) < percentage
    X_dirty[mask] = np.nan
    
    print(f"Svuotato il {percentage*100}% delle celle totali del dataset.")
    return X_dirty

def sporca_elo_last(X, percentage, lag=10):
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
        # Creiamo una serie con i valori traslati (quelli di 10 righe fa)
        # shift(lag) sposta i dati, bfill() riempie i primi 10 buchi con il primo valore disponibile
        stale_values = X_dirty[col].shift(lag).bfill()
        
        # Scegliamo gli indici da sporcare
        idx = np.random.choice(X_dirty.index, n_changes, replace=False)
        
        # Sostituiamo i valori attuali con quelli vecchi
        X_dirty.loc[idx, col] = stale_values.loc[idx]
        
    print(f"Simulato ritardo di {lag} partite per l'Elo ({percentage*100}% dei record).")
    return X_dirty

if __name__ == "__main__":
    print("--- INIZIO PIPELINE DATA QUALITY ---")
    dataset_path = "progetto/dataset_ml_ready.csv" 
    
    try:
        # 1. Ottieni i dati puliti già divisi e filtrati per FEATURES
        X_train_clean, y_train_clean, X_test_clean, y_test_clean = get_train_test_split(dataset_path)
        
        
        # 3. VALUTAZIONE BASELINE (Dati Puliti)
        print("\n--- TEST 1: Modello su Dati Puliti ---")
        acc_clean = train_and_evaluate_mlp(X_train_clean, y_train_clean, X_test_clean, y_test_clean)
        print(f"Accuracy Baseline: {acc_clean:.4f}")

        """
        # --- VALORI MANCANTI (Completeness) ---
        print("\n--- TEST 4: Modello con Valori Mancanti (5%) ---")
        # Introduciamo i NaN
        X_missing = introduci_missing_values(X_train_clean, 0.05)
        # IMPORTANTE: Il MLP non accetta NaN, quindi li riempiamo con la media
        X_train_dirty = X_missing.fillna(X_train_clean.mean())
        """
        # ---  ELO NON AGGIORNATO (Lag 10) ---
        print("\n--- TEST 5: Modello con Elo vecchio di 10 partite ---")
        X_stale = sporca_elo_last(X_train_clean, 0.1, lag=10)
        acc_stale = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
        
        print(f"Accuracy (Elo Stale): {acc_stale:.4f}")
        print(f"Perdita: {((acc_clean - acc_stale)*100):.2f}%")


        # 4. VALUTAZIONE ROBUSTEZZA (Dati Sporchi)
        print("\n--- TEST 2: Modello su Dati Sporchi ---")
        acc_dirty = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
        print(f"Accuracy con Accuracy Error (5%): {acc_dirty:.4f}")
        
        # 5. CONFRONTO
        diff = acc_clean - acc_dirty
        print(f"\nIl modello ha perso lo {diff*100:.2f}% di accuratezza a causa del rumore.")

    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")