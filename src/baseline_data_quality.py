import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

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
def train_and_evaluate_mlp(X_train, y_train, X_test, y_test):
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
    
    y_pred = mlp.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)



# ESECUZIONE PRINCIPALE

if __name__ == "__main__":
    print("--- INIZIO PIPELINE DATA QUALITY ---")
    
    # Assicurati che il percorso del file sia corretto!
    dataset_path = "../data/dataset_ml_ready.csv" 
    
    try:
        print("Caricamento dati e split temporale...")
        X_train_clean, y_train_clean, X_test_clean, y_test_clean = get_train_test_split(dataset_path)
        
        print(f"   Train set: {len(X_train_clean)} match | Test set: {len(X_test_clean)} match")
        
        print("Addestramento Rete Neurale Baseline...")
        baseline_acc = train_and_evaluate_mlp(X_train_clean, y_train_clean, X_test_clean, y_test_clean)
        
        print(f"\nAccuracy Baseline = {baseline_acc:.4f}")
        
        
    except FileNotFoundError:
        print(f"\nERRORE: Non presente il file {dataset_path}")