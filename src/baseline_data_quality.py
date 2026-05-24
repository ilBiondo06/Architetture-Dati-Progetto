import json
import os

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

_BASELINE_ACCURACY = None

# COSTANTI E FEATURES, usate per l'allenamento e la valutazione del modello
FEATURES = [
    'best_of', 'round_num', 'surface_Hard', 'surface_Clay', 'surface_Grass',
    'diff_age', 'diff_ht', 'p1_is_left', 'p2_is_left',
    'DIFF_ELO', 'DIFF_SURF_ELO', 'ATP_RANK_DIFF', 'DIFF_H2H', 'DIFF_N_GAMES',
    'WIN_LAST_5_DIFF', 'WIN_LAST_25_DIFF', 'WIN_LAST_50_DIFF', 'WIN_LAST_100_DIFF',
    'ELO_GRAD_20_DIFF', 'ELO_GRAD_35_DIFF', 'ELO_GRAD_50_DIFF', 'ELO_GRAD_100_DIFF',
    'ACE_L5_DIFF', 'ACE_L20_DIFF', 'ACE_L50_DIFF',
    'DF_L5_DIFF', 'DF_L20_DIFF', 'DF_L50_DIFF',
    '1ST_WIN_PCT_L5_DIFF', '1ST_WIN_PCT_L20_DIFF', '1ST_WIN_PCT_L50_DIFF',
    'BP_SAVE_PCT_L5_DIFF', 'BP_SAVE_PCT_L20_DIFF', 'BP_SAVE_PCT_L50_DIFF'
]

# FUNZIONI DI SUPPORTO
# Divisione dei dati per Train e Test basata sulla data del torneo
def get_train_test_split(file_path):
    """Carica il CSV e divide in Train e Test."""
    df = pd.read_csv(file_path)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    train_mask = df['tourney_date'] < '2023-01-01'
    test_mask = df['tourney_date'] >= '2023-01-01'
    
    X_train = df.loc[train_mask, FEATURES].fillna(0)
    y_train = df.loc[train_mask, 'target']
    X_test = df.loc[test_mask, FEATURES].fillna(0)
    y_test = df.loc[test_mask, 'target']
    
    return X_train, y_train, X_test, y_test

# Funzione per addestrare la MLP e valutare l'accuracy
def train_and_evaluate_mlp(X_train, y_train, X_test, y_test, return_model=False):
    """Scala i dati, addestra la MLP e ritorna l'accuracy."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 128, 64, 32),
        activation='relu',
        alpha=0.01, 
        max_iter=150, 
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    
    acc = mlp.score(X_test_scaled, y_test)
        
        # LA MAGIA È QUI:
    if return_model:
        return acc, mlp
    else:
        return acc

def get_or_compute_baseline(dataset_path):
    """
    Gestisce la persistenza della baseline su disco usando un file JSON.
    Salva l'accuratezza, la loss curve e le epoche di convergenza.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(base_dir, "baseline_metrics.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
        return data["accuracy"]
    
    print("Calcolo della Baseline e salvataggio delle metriche di addestramento...")
    X_train, y_train, X_test, y_test = get_train_test_split(dataset_path)
    
    acc, mlp = train_and_evaluate_mlp(X_train, y_train, X_test, y_test, return_model=True)
    
    # Struttura dei metadati da salvare
    cache_data = {
        "accuracy": float(acc),
        "n_iter": int(mlp.n_iter_),
        "loss_curve": [float(x) for x in mlp.loss_curve_]
    }
    
    # Salviamo il file JSON nella cartella 'src'
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=4)
        
    return acc


def load_baseline_cache():
    """
    Funzione helper per caricare l'intero dizionario delle metriche salvate in cache.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(base_dir, "baseline_metrics.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return None



# ESECUZIONE PRINCIPALE

if __name__ == "__main__":
    print("-- INIZIO PIPELINE DATA QUALITY --")
    
    # Calcolo dinamico del path rispetto alla posizione dello script (cartella 'src')
    base_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_folder, "..", "data","clean", "dataset_ml_ready.csv") 
    
    try:
        baseline_acc = get_or_compute_baseline(dataset_path)
        print(f"\nAccuracy Baseline = {baseline_acc:.4f}")
        
    except FileNotFoundError:
        print(f"\nERRORE: file {dataset_path} non trovato")
    except Exception as e:
        print(f"\nErrore durante l'esecuzione: {e}")