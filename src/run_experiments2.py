# ==========================================
# src/run_experiments.py (SOLO GENERAZIONE GRAFICI DA DATI PRE-CALCOLATI)
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File di input 
INPUT_CSV = "risultati_aggregati.csv"  
SOGLIA_CRITICA = 0.50

def main():
    print(f" Caricamento dati pre-calcolati da: {INPUT_CSV}...")
    try:
        df_stats = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f" Errore: Il file '{INPUT_CSV}' non esiste in questa cartella!")
        return

    # 1. STAMPA DEI BREAK-POINTS SUL TERMINALE
    print("\n ANALISI QUANTITATIVA DEI BREAK-POINTS (SOGLIA < 50%):")
    print("-" * 60)
    for strategia in df_stats['Strategia'].unique():
        df_strat = df_stats[df_stats['Strategia'] == strategia]
        break_points = df_strat[df_strat['Mean_Accuracy'] <= SOGLIA_CRITICA]
        if not break_points.empty:
            print(f"  [{strategia}] Inversione del segnale a epsilon >= {break_points.iloc[0]['Epsilon']}")
        else:
            print(f" [{strategia}] Modello resiliente fino a epsilon 0.50")

    # 2. GENERAZIONE AUTOMATICA DEI GRAFICI 
    print("\n Generazione grafici per il report...")
    sns.set_theme(style="whitegrid")

    # GRAFICO 1: Curve di decadimento (Performance vs Noise Level)
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
    print(" -> Grafico 1 salvato in 'grafico_1_curve_decadimento.png'")
    
    # GRAFICO 2: Heatmap della correlazione errore-incertezza
    df_pivot = df_stats.pivot(index='Strategia', columns='Epsilon', values='Mean_Delta')
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_pivot, annot=True, fmt=".2%", cmap="YlOrRd", cbar_kws={'label': 'Perdita Relativa di Accuracy (Delta)'})
    plt.title('Heatmap del Tallone d\'Achille dell\'Architettura Dati', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Livello di Perturbazione (Epsilon)', fontsize=12)
    plt.ylabel('Dimensione DQ / Strategia di Errore', fontsize=12)
    plt.tight_layout()
    plt.savefig("grafico_2_heatmap_correlazione.png", dpi=300)
    plt.close()
    print(" -> Grafico 2 salvato in 'grafico_2_heatmap_correlazione.png'")

    # GRAFICO 3: Istogramma comparativo dell'impatto marginale (epsilon=30%)
    # Controlliamo prima se l'epsilon 0.3 esiste nel file dei professori
    EPSILON_CRITICO = 0.30
    if EPSILON_CRITICO in df_stats['Epsilon'].values:
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
        plt.savefig("curve_decadimento.png", dpi=300)
        plt.close()
        print(" -> Grafico 3 salvato in 'curve_decadimento.png'")
    else:
        print(f"⚠️ Nota: Impossibile generare il Grafico 3. Epsilon {EPSILON_CRITICO} non presente nei dati.")

    print("\n🎉 Pipeline completata con successo!")

if __name__ == "__main__":
    main()