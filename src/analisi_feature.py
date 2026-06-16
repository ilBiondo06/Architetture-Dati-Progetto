import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

# Importiamo le costanti e funzioni utili dalla baseline
from baseline_data_quality import get_train_test_split, FEATURES

def calcola_e_plotta_evoluzione(esperimento_nome, prefisso_file, test_clean_X, test_clean_y, color_palette="Reds_d"):
    """
    Legge i CSV sporchi di un esperimento, calcola la Feature Importance 
    per ogni step e genera un grafico a griglia dell'evoluzione dei pesi decisionali.
    """
    print(f"AVVIO ANALISI FEATURE IMPORTANCE: {esperimento_nome}")
    
    # Setup percorsi
    base_folder = os.path.dirname(os.path.abspath(__file__))
    dirty_dir = os.path.join(base_folder, "..", "data", "dirty")
    plots_dir = os.path.join(base_folder, "..", "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    percentuali_str = ["10", "20", "40", "60", "80"]
    
    importanza_per_step = {}
    
    # -------------------------------------
    # 1. TOP 8 DELLA BASELINE (DEGRADO 0%)
    # -------------------------------------
    clean_csv = os.path.join(base_folder, "..", "data", "clean", "dataset_ml_ready.csv")
    X_train_clean, y_train_clean, _, _ = get_train_test_split(clean_csv)
    
    # Rimuoviamo p1_id se presente
    X_train_clean = X_train_clean[FEATURES]
    
    scaler_clean = StandardScaler()
    X_train_scaled = scaler_clean.fit_transform(X_train_clean)
    X_test_scaled = scaler_clean.transform(test_clean_X)
    
    print("[1/3] Addestramento Baseline Sana...")
    # AGGIUNTO RANDOM_STATE=42 per isolare l'effetto del degrado
    mlp_clean = MLPClassifier(hidden_layer_sizes=(128, 128, 64, 32), activation='relu', 
                              alpha=0.01, max_iter=150, early_stopping=True, random_state=42)
    mlp_clean.fit(X_train_scaled, y_train_clean)
    
    print("[2/3] Estrazione pesi decisionali (Permutation Importance)...")
    res_clean = permutation_importance(mlp_clean, X_test_scaled, test_clean_y, n_repeats=5, random_state=42, n_jobs=-1)
    df_clean = pd.DataFrame({'Feature': FEATURES, 'Importanza': res_clean.importances_mean})
    
    # Fissiamo le 8 feature più importanti del modello sano per il confronto
    top_features = df_clean.sort_values(by='Importanza', ascending=False).head(8)['Feature'].tolist()
    max_importanza_assoluta = df_clean['Importanza'].max()
    
    # INSERIAMO LA BASELINE COME PRIMO GRAFICO (0%)
    importanza_per_step["0"] = df_clean[df_clean['Feature'].isin(top_features)]
    
    # ---------------------------------------------------------
    # 2. CICLO SUI DATASET SPORCHI
    # ---------------------------------------------------------
    print("[3/3] Scansione dei dataset corrotti in corso...")
    for p_str in percentuali_str:
        file_name = f"dataset_{prefisso_file}_{p_str}pct.csv"
        file_path = os.path.join(dirty_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"File {file_name} non trovato, salto.")
            continue
            
        print(f"  -> Elaborazione degrado {p_str}%...")
        df_dirty = pd.read_csv(file_path)
        
        if 'target' in df_dirty.columns:
             y_dirty = df_dirty['target']
             X_dirty = df_dirty.drop(columns=['target'])
             X_dirty = X_dirty[FEATURES]
        else:
             print("ERRORE: Colonna target non trovata nel CSV")
             continue

        # Addestriamo il modello corrotto sui dati sporchi
        scaler_dirty = StandardScaler()
        X_train_dirty_scaled = scaler_dirty.fit_transform(X_dirty)
        
        # AGGIUNTO RANDOM_STATE=42
        mlp_dirty = MLPClassifier(hidden_layer_sizes=(128, 128, 64, 32), activation='relu', 
                                  alpha=0.01, max_iter=150, early_stopping=True, random_state=42)
        mlp_dirty.fit(X_train_dirty_scaled, y_dirty)
        
        # Testiamo l'importanza sulle feature della realtà (Dati PULITI)
        res_dirty = permutation_importance(mlp_dirty, X_test_scaled, test_clean_y, n_repeats=5, random_state=42, n_jobs=-1)
        
        df_res = pd.DataFrame({'Feature': FEATURES, 'Importanza': res_dirty.importances_mean})
        importanza_per_step[p_str] = df_res[df_res['Feature'].isin(top_features)]

    # -----------------------------------
    # 3. GENERAZIONE GRAFICO A GRIGLIA
    # -----------------------------------
    if importanza_per_step:
        # Avendo aggiunto lo zero, ora i plot sono esattamente 6. Perfetto per 2 righe da 3 colonne!
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, (p_str, df_plot) in enumerate(importanza_per_step.items()):
            # Ordiniamo sempre in modo decrescente per leggibilità
            df_plot = df_plot.sort_values(by='Importanza', ascending=False)
            
            sns.barplot(x='Importanza', y='Feature', data=df_plot, palette=color_palette, ax=axes[i])
            axes[i].set_title(f"Degrado: {p_str}%", fontweight='bold')
            axes[i].set_xlabel('Impatto sull\'accuratezza')
            axes[i].set_ylabel('')
            
            # Fissiamo l'asse X per rendere le barre comparabili visivamente tra i vari step
            axes[i].set_xlim(0, max_importanza_assoluta * 1.15) 
            
            # AGGIUNTA DELLA GRIGLIA VERTICALE (sotto le barre)
            axes[i].set_axisbelow(True)
            axes[i].xaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
            
        # Nasconde eventuali riquadri vuoti (anche se con 6 grafici non ce ne saranno)
        for j in range(len(importanza_per_step), len(axes)):
            fig.delaxes(axes[j])
            
        plt.suptitle(f'Evoluzione della Feature Importance\n{esperimento_nome}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        nome_file_out = f"evoluzione_feature_{prefisso_file}.png"
        plt.savefig(os.path.join(plots_dir, nome_file_out), dpi=300, bbox_inches='tight')
        print(f"\nDashboard salvata con successo in: results/plots/{nome_file_out}")

# ==========================================
# ESECUZIONE MAIN
# ==========================================
if __name__ == "__main__":
    
    # 1. Prepariamo il Test Set Pulito Universale
    base_folder = os.path.dirname(os.path.abspath(__file__))
    clean_csv = os.path.join(base_folder, "..", "data", "clean", "dataset_ml_ready.csv")
    _, _, X_test_clean, y_test_clean = get_train_test_split(clean_csv)
    X_test_clean = X_test_clean[FEATURES]
    
    # 2. Lanciamo l'analisi per tutti i nostri esperimenti
    
    # --- DIMENSIONE: COERENZA (Rosso) ---
    calcola_e_plotta_evoluzione("Coerenza: Target Flipping", 
                                "coerenza_targetflip", X_test_clean, y_test_clean, color_palette="Reds_d")
    calcola_e_plotta_evoluzione("Coerenza: Contraddizione Elo vs Rank", 
                                "coerenza_elorank", X_test_clean, y_test_clean, color_palette="Reds_d")
    
    # --- DIMENSIONE: ACCURATEZZA (Blu) ---
    calcola_e_plotta_evoluzione("Accuratezza: Outliers (Età e Altezza)", 
                                "accuratezza_outliers", X_test_clean, y_test_clean, color_palette="Blues_d")
    calcola_e_plotta_evoluzione("Accuratezza: Rumore Gaussiano", 
                                "accuratezza_gaussiano", X_test_clean, y_test_clean, color_palette="Blues_d")
    
    # --- DIMENSIONE: COMPLETEZZA E TEMPESTIVITÀ (Arancione) ---
    calcola_e_plotta_evoluzione("Completezza: Blackout Sistemico", 
                                "completezza_blackout", X_test_clean, y_test_clean, color_palette="Oranges_d")
    calcola_e_plotta_evoluzione("Tempestività: Lag-10 (Dati Obsoleti)", 
                                "tempestivita_lag10", X_test_clean, y_test_clean, color_palette="Oranges_d")

    print("\nTutte le analisi di Feature Importance sono state completate")