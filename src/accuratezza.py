import pandas as pd
import numpy as np

from baseline_data_quality import train_and_evaluate_mlp, get_train_test_split

def eta_altezza(X, percentage):
    """
    Introduce errori di accuratezza (outliers) nelle differenze.
    - diff_ht: inserisce differenze estreme (es. +- 150cm)
    - diff_age: inserisce differenze impossibili (es. +- 80 anni)
    """
    X_dirty = X.copy()
    n_rows = len(X_dirty)
    n_changes = int(n_rows * percentage)
    
    # Selezioniamo gli indici casuali
    idx = np.random.choice(X_dirty.index, n_changes, replace=False)
    
    # Modifichiamo diff_ht (differenza altezza)
    X_dirty.loc[idx, 'diff_ht'] = np.random.choice([150.0, -150.0], size=n_changes)
    
    # Modifichiamo diff_age (differenza età)
    X_dirty.loc[idx, 'diff_age'] = np.random.choice([80.0, -80.0], size=n_changes)
    
    print(f"Modificati {n_changes} record per diff_ht e diff_age ({percentage*100}%).")
    return X_dirty

def mano_dominante(X, percentage):
    """
    Inverte la mano dominante (0 -> 1, 1 -> 0) per p1_is_left e p2_is_left.
    """
    X_dirty = X.copy()
    n_rows = len(X_dirty)
    n_changes = int(n_rows * percentage)
    
    for col in ['p1_is_left', 'p2_is_left']:
        idx = np.random.choice(X_dirty.index, n_changes, replace=False)
        # Operazione XOR o semplice inversione: se è 1 diventa 0, se è 0 diventa 1
        X_dirty.loc[idx, col] = 1 - X_dirty.loc[idx, col]
    
    print(f"Invertita la mano dominante nel {percentage*100}% dei record.")
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
        X_train_dirty = eta_altezza(X_train_clean, 0.05)
        X_train_dirty = mano_dominante(X_train_dirty, 0.05)

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