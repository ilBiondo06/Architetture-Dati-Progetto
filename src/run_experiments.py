# ==========================================
# src/run_experiments.py (Incluso il calcolo del Break-Point e Grafici)
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from baseline_data_quality import get_train_test_split, train_and_evaluate_mlp
from accuratezza import mano_dominante, eta_altezza
from coerenza import sporca_elo

# Parametri stabili del framework
N_SIMULATIONS = 3
EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
BASELINE_ACC = 0.6850  
DATASET_PATH = "dataset_ml_ready.csv"
OUTPUT_PATH = "progetto.csv"

INJECTORS = {
  #  'Mano Dominante': mano_dominante,
    'Età e Altezza Outliers': eta_altezza,
    'Elo Zero': lambda X, p: sporca_elo(X, p, strategy='zero'),
  #  'Elo Mean': lambda X, p: sporca_elo(X, p, strategy='mean'),
   # 'Elo Stale': lambda X, p: sporca_elo(X, p, strategy='stale_50')
}

def main():
    risultati_raw = []
    print(" Avvio Stress Test Quantitativo (Monte Carlo)...")
    
    for name, injector_func in INJECTORS.items():
        print(f" Elaborazione: {name}")
        for eps in EPSILONS:
            for run in range(N_SIMULATIONS):
                X_train, y_train, X_test, y_test = get_train_test_split(DATASET_PATH)
                
                # Iniezione
                X_test_dirty = injector_func(X_test, eps) if eps > 0 else X_test.copy()
                # Valutazione
                acc_esito = train_and_evaluate_mlp(X_train, y_train, X_test_dirty, y_test)
                # Formula Delta
                delta_acc = (BASELINE_ACC - acc_esito) / BASELINE_ACC
                
                risultati_raw.append({
                    'Strategia': name, 'Epsilon': eps, 'Run': run,
                    'Accuracy': acc_esito, 'Delta_Accuracy': delta_acc
                })

    # Trasformazione in DataFrame ed Elaborazione Statistica immediata
    df_raw = pd.DataFrame(risultati_raw)
    
    print(" Calcolo metriche statistiche (Medie, Deviazioni Standard, IC 95%)...")
    
    #  Calcolo metriche statistiche
    df_stats = df_raw.groupby(['Strategia', 'Epsilon']).agg(
        Mean_Accuracy=('Accuracy', 'mean'),
        Std_Accuracy=('Accuracy', 'std'),
        Mean_Delta=('Delta_Accuracy', 'mean')
    ).reset_index()
    
    # Formula errore standard per intervalli di confidenza al 95%
    df_stats['IC_95'] = 1.96 * (df_stats['Std_Accuracy'] / np.sqrt(N_SIMULATIONS))
    df_stats.to_csv("risultati_aggregati.csv", index=False)
    
    #  STAMPA DEI BREAK-POINTS SUL TERMINALE
    print("\n ANALISI QUANTITATIVA DEI BREAK-POINTS (SOGLIA < 50%):")
    print("-" * 60)
    SOGLIA_CRITICA = 0.50
    for strategia in df_stats['Strategia'].unique():
        df_strat = df_stats[df_stats['Strategia'] == strategia]
        break_points = df_strat[df_strat['Mean_Accuracy'] <= SOGLIA_CRITICA]
        if not break_points.empty:
            print(f" [{strategia}] Inversione del segnale a epsilon >= {break_points.iloc[0]['Epsilon']}")
        else:
            print(f" [{strategia}] Modello resiliente fino a epsilon 0.50")

    #  GENERAZIONE AUTOMATICA DEI GRAFICI 
    print("\n Generazione grafici per il report...")
    sns.set_theme(style="whitegrid")

    # GRAFICO 1: Curve di decadimento (Performarce vs Noise Level)
    plt.figure(figsize=(10, 6))
    for strategia in df_stats['Strategia'].unique():
        df_strat = df_stats[df_stats['Strategia'] == strategia]
        
        # Linea della media
        plt.plot(df_strat['Epsilon'], df_strat['Mean_Accuracy'], label=strategia, marker='o', linewidth=2)
        
        # Ombreggiatura dell'Intervallo di Confidenza al 95%
        under_line = df_strat['Mean_Accuracy'] - df_strat['IC_95']
        over_line = df_strat['Mean_Accuracy'] + df_strat['IC_95']
        plt.fill_between(df_strat['Epsilon'], under_line, over_line, alpha=0.15)

    # Linea di soglia critica (Random Guessing = 50%)
    plt.axhline(y=SOGLIA_CRITICA, color='red', linestyle='--', alpha=0.7, label='Random Guessing (50%)')
    plt.title('Curve di Decadimento delle Performance', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Livello di Perturbazione (Epsilon)', fontsize=12)
    plt.ylabel('Test Accuracy Media', fontsize=12)
    plt.ylim(0.45, 0.75)
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    plt.savefig("grafico_1_curve_decadimento.png", dpi=300)
    plt.close()
    print("  -> Grafico 1 salvato in grafico_1_curve_decadimento.png'")
    
    # GRAFICO 2: heatmap della correlazione errore-incertezza
    df_pivot = df_stats.pivot(index='Strategia', columns='Epsilon', values='Mean_Delta')
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_pivot, annot=True, fmt=".2%", cmap="YlOrRd", cbar_kws={'label': 'Perdita Relativa di Accuracy (Delta)'})
    plt.title('Heatmap del Tallone d\'Achille dell\'Architettura Dati', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Livello di Perturbazione (Epsilon)', fontsize=12)
    plt.ylabel('Strategia di Errore', fontsize=12)
    plt.tight_layout()
    plt.savefig("grafico_2_heatmap_correlazione.png", dpi=300)
    plt.close()
    print("  -> Grafico 2 salvato in grafico_2_heatmap_correlazione.png'")

    # GRAFICO 3: instogramma comparativo dell'impatto marginale (epsilon=30%)
    EPSILON_CRITICO= 0.30
    df_critico = df_stats[df_stats['Epsilon'] == EPSILON_CRITICO].sort_values(by='Mean_Delta', ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Reds_r", len(df_critico))
    
    # Barre con errore standard basato su IC_95
    bars = plt.bar(df_critico['Strategia'], df_critico['Mean_Delta'] * 100, 
                   yerr=df_critico['IC_95'] * 100, capsize=5, color=colors, edgecolor='black', alpha=0.8)
    
    plt.title(f'Impatto Marginale sulla Degradazione delle Performance (Epsilon = {int(EPSILON_CRITICO*100)}%)', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Crollo Relativo dell\'Accuracy (%)', fontsize=12)
    plt.xlabel('Noise Injector', fontsize=12)
    plt.xticks(rotation=15)
    
    # Aggiunta etichette sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'-{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    # Salviamo l'immagine direttamente nella cartella docs/ così il report può leggerla staticamente
    plt.savefig("curve_decadimento.png", dpi=300)
    print(" Grafico salvato in curve_decadimento.png'")

if __name__ == "__main__":
    main()