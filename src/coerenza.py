import numpy as np
import pandas as pd

from baseline_data_quality import train_and_evaluate_mlp, get_train_test_split


def sporca_elo(X, percentage, strategy):
    """
    Introduce errori nell'Elo dei giocatori.
    Strategie supportate: 
    - 'zero': azzera il valore
    - 'mean': usa la media globale della colonna
    - 'stale_50': usa la media mobile delle 50 righe precedenti (simula dati vecchi)
    """
    X_dirty = X.copy()
    n_rows = len(X_dirty)
    n_changes = int(n_rows * percentage)
    
    elo_cols = [col for col in X_dirty.columns if 'elo' in col.lower()]
    
    for col in elo_cols:
        idx = np.random.choice(X_dirty.index, n_changes, replace=False)
        
        if strategy == 'zero':
            X_dirty.loc[idx, col] = 0.0
            
        elif strategy == 'mean':
            X_dirty.loc[idx, col] = X_dirty[col].mean()
            
        elif strategy == 'stale_50':
            # 1. Creiamo una colonna temporanea con la media mobile
            # min_periods=1 assicura che calcoli anche per le prime 49 righe
            rolling_mean = X_dirty[col].rolling(window=50, min_periods=1).mean()
            # 2. Sostituiamo solo gli indici selezionati con il valore "stagnante"
            X_dirty.loc[idx, col] = rolling_mean.loc[idx]
            
    print(f"Modificato Elo in {elo_cols} con strategia '{strategy}' ({percentage*100}%).")
    return X_dirty



def sporca_cronologia_date(X, percentage):
    """
    Simula errori di inserimento data scambiando i valori temporali 
    con date pescate casualmente dal dataset (out of order).
    """
    X_dirty = X.copy()
    # Cerchiamo colonne
    date_cols = [col for col in X_dirty.columns if any(x in col.lower() for x in ['tourney_date'])]
    
    if not date_cols:
        print("Nessuna colonna temporale trovata.")
        return X_dirty

    n_rows = len(X_dirty)
    n_changes = int(n_rows * percentage)

    for col in date_cols:
        idx_to_change = np.random.choice(X_dirty.index, n_changes, replace=False)
        # Peschiamo valori casuali dalla stessa colonna per creare l'anacronismo
        random_values = np.random.choice(X_dirty[col], size=n_changes, replace=True)
        X_dirty.loc[idx_to_change, col] = random_values
        
    print(f"Cronologia alterata per {date_cols} ({percentage*100}% dei record).")
    return X_dirty


if __name__ == "__main__":
    print("--- INIZIO PIPELINE DATA QUALITY ---")
    dataset_path = "progetto/dataset_ml_ready.csv" 
    
    try:
        # 1. Ottieni i dati puliti già divisi e filtrati per FEATURES
        X_train_clean, y_train_clean, X_test_clean, y_test_clean = get_train_test_split(dataset_path)
        
        # 2. CREAZIONE DEL SET "SPORCO"
        # Non aggiungiamo colonne, modifichiamo i valori dentro X_train_dirty
        # che corrispondono alle feature usate dal modello (diff_ht, diff_age, ecc.)
        print("Generazione dati sporchi...")
        X_train_dirty = sporca_elo(X_train_clean, 0.1, 'stale_50')
      # X_train_dirty = sporca_cronologia_date(X_train_clean, 0.)

        # 3. VALUTAZIONE BASELINE (Dati Puliti)
        print("\n--- TEST 1: Modello su Dati Puliti ---")
        acc_clean = train_and_evaluate_mlp(X_train_clean, y_train_clean, X_test_clean, y_test_clean)
        print(f"Accuracy Baseline: {acc_clean:.4f}")

        # 4. VALUTAZIONE ROBUSTEZZA (Dati Sporchi)
        print("\n--- TEST 2: Modello su Dati Sporchi ---")
        acc_dirty = train_and_evaluate_mlp(X_train_dirty, y_train_clean, X_test_clean, y_test_clean)
        print(f"Accuracy con Accuracy Error (5%): {acc_dirty:.4f}")
        
        # 5. CONFRONTO
        diff = acc_clean - acc_dirty
        print(f"\nIl modello ha perso lo {diff*100:.2f}% di accuratezza a causa del rumore.")

    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")