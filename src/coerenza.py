import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from baseline_data_quality import train_and_evaluate_mlp, get_train_test_split, get_or_compute_baseline, load_baseline_cache

# ======================
# FUNZIONI DI DEGRADO
# ======================

def sporca_coerenza_target(y, percentage):
    """
    Esperimento Target Flipping
    Simula un errore critico di coerenza logica. I dati fisici del match sono corretti, ma l'etichetta del vincitore viene invertita.
    """
    y_dirty = y.copy()
    n_rows = len(y_dirty)
    n_changes = int(n_rows * percentage)
    
    idx = np.random.choice(y_dirty.index, n_changes, replace=False)
    y_dirty.loc[idx] = 1 - y_dirty.loc[idx] 
    
    return y_dirty


def sporca_coerenza_elo_ranking(X, percentage):
    """
    Esperimento Feature Contradiction
    Simula un'incoerenza di dominio: inverte il segno della differenza di Ranking ATP mantenendo l'Elo intatto. 
    """
    X_dirty = X.copy()
    n_rows = len(X_dirty)
    n_changes = int(n_rows * percentage)
    
    if 'ATP_RANK_DIFF' in X_dirty.columns:
        idx = np.random.choice(X_dirty.index, n_changes, replace=False)
        X_dirty.loc[idx, 'ATP_RANK_DIFF'] = X_dirty.loc[idx, 'ATP_RANK_DIFF'] * -1
        
    return X_dirty


if __name__ == "__main__":
    print("--- INIZIO PIPELINE DATA QUALITY: ESPERIMENTI DI COERENZA ---")
    
    # Definizione dei percorsi 
    base_folder = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(base_folder, "..", "data", "clean")
    dirty_dir = os.path.join(base_folder, "..", "data", "dirty")
    plots_dir = os.path.join(base_folder, "..", "results", "plots")
    
    os.makedirs(dirty_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    dataset_path = os.path.join(clean_dir, "dataset_ml_ready.csv") 
    
    try:
        # Recupero l'accuratezza
        baseline_acc = get_or_compute_baseline(dataset_path)
        print(f"Accuratezza Baseline: {baseline_acc:.4f}\n")
        
        baseline_cache = load_baseline_cache()
        if baseline_cache is not None:
            loss_baseline = baseline_cache["loss_curve"]
            epoche_baseline = baseline_cache["n_iter"]
        else:
            loss_baseline = []
            epoche_baseline = 0
            
        X_train_clean, y_train_clean, X_test_clean, y_test_clean = get_train_test_split(dataset_path)
        
        percentuali = [0.05, 0.10, 0.20, 0.30, 0.50]  
        N_RUNS = 10  
        
        risultati = {
            'Target_Flipping': {'media': [], 'std': []},
            'Elo_Ranking': {'media': [], 'std': []}
        }
        
        loss_target_50 = None
        epoche_target_50 = 0

        # -----------------------------
        # ESPERIMENTO TARGET FLIPPING
        # -----------------------------
        print(f"\n--- AVVIO ESPERIMENTO TARGET FLIPPING | {N_RUNS} RUNS PER STEP ---")
        for p in percentuali:
            run_accuracies = [] 
            for run in tqdm(range(N_RUNS), desc=f"Degrado Target {p*100:0.0f}%", colour='red'):
                y_stale = sporca_coerenza_target(y_train_clean, percentage=p)
                
                if run == 0:
                    df_to_save = X_train_clean.copy()
                    df_to_save['target'] = y_stale
                    dirty_filename = f"dataset_coerenza_targetflip_{int(p*100)}pct.csv"
                    df_to_save.to_csv(os.path.join(dirty_dir, dirty_filename), index=False)
                    
                    if p == 0.50:
                        acc, mlp_dirty = train_and_evaluate_mlp(X_train_clean, y_stale, X_test_clean, y_test_clean, return_model=True)
                        loss_target_50 = mlp_dirty.loss_curve_
                        epoche_target_50 = mlp_dirty.n_iter_
                    else:
                        acc = train_and_evaluate_mlp(X_train_clean, y_stale, X_test_clean, y_test_clean)
                else:
                    acc = train_and_evaluate_mlp(X_train_clean, y_stale, X_test_clean, y_test_clean)
                    
                run_accuracies.append(acc)
            
            media_acc, std_acc = np.mean(run_accuracies), np.std(run_accuracies)
            print(f"Accuracy Media: {media_acc:.4f} ± {std_acc:.4f} | Perdita: {(baseline_acc - media_acc)*100:.2f}%\n")
        
            risultati['Target_Flipping']['media'].append(media_acc)
            risultati['Target_Flipping']['std'].append(std_acc)

        # --------------------------------------------
        # ESPERIMENTO CONTRADDIZIONE ELO VS RANKING
        # --------------------------------------------
        print(f"--- AVVIO ESPERIMENTO CONTRADDIZIONE ELO VS RANKING | {N_RUNS} RUNS PER STEP ---")
        for p in percentuali:
            run_accuracies = [] 
            for run in tqdm(range(N_RUNS), desc=f"Degrado Elo-Rank {p*100:0.0f}%", colour='magenta'):
                X_stale = sporca_coerenza_elo_ranking(X_train_clean, percentage=p)
                
                if run == 0:
                    df_to_save = X_stale.copy()
                    df_to_save['target'] = y_train_clean
                    dirty_filename = f"dataset_coerenza_elorank_{int(p*100)}pct.csv"
                    df_to_save.to_csv(os.path.join(dirty_dir, dirty_filename), index=False)
                
                acc = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
                run_accuracies.append(acc)
            
            media_acc, std_acc = np.mean(run_accuracies), np.std(run_accuracies)
            print(f"Accuracy Media: {media_acc:.4f} ± {std_acc:.4f} | Perdita: {(baseline_acc - media_acc)*100:.2f}%\n")
        
            risultati['Elo_Ranking']['media'].append(media_acc)
            risultati['Elo_Ranking']['std'].append(std_acc)

        # ================= SALVATAGGIO RISULTATI =================
        risultati_standardizzati = {
            'Target_Flipping': {
                'media': [float(m) for m in risultati['Target_Flipping']['media']],
                'std': [float(s) for s in risultati['Target_Flipping']['std']]
            },
            'Elo_Ranking': {
                'media': [float(m) for m in risultati['Elo_Ranking']['media']],
                'std': [float(s) for s in risultati['Elo_Ranking']['std']]
            }
        }
        
        dati_da_salvare = {
            "baseline_acc": float(baseline_acc),
            "loss_baseline": [float(x) for x in loss_baseline] if hasattr(loss_baseline, "__iter__") else [],
            "epoche_baseline": int(epoche_baseline),
            "percentuali": [float(p) for p in percentuali],
            "risultati": risultati_standardizzati,
            "loss_target_50": [float(x) for x in loss_target_50] if loss_target_50 is not None else None,
            "epoche_target_50": int(epoche_target_50)
        }
        
        path_cache_risultati = os.path.join(plots_dir, "risultati_coerenza.json")
        with open(path_cache_risultati, "w") as f:
            json.dump(dati_da_salvare, f, indent=4)
        print(f"\n[OK] Risultati salvati correttamente in: {path_cache_risultati}")

        # ==========================================
        # GENERAZIONE E SALVATAGGIO DEI GRAFICI
        # ==========================================
        print("\nGenerazione e salvataggio dei grafici...")
        sns.set_theme(style="whitegrid")
        
        perc_labels = [int(p * 100) for p in percentuali]
        delta_target = [(baseline_acc - m) * 100 for m in risultati['Target_Flipping']['media']]
        delta_elo = [(baseline_acc - m) * 100 for m in risultati['Elo_Ranking']['media']]
        
        # --- FIGURA 1: ACCURATEZZA ASSOLUTA VS DELTA PERDITA ---
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
        
        axes1[0].errorbar(perc_labels, risultati['Target_Flipping']['media'], yerr=risultati['Target_Flipping']['std'], fmt='-o', label='Target Flipping', color='#d62728', capsize=5, linewidth=2)
        axes1[0].errorbar(perc_labels, risultati['Elo_Ranking']['media'], yerr=risultati['Elo_Ranking']['std'], fmt='-s', label='Incoerenza Elo-Ranking', color='#9467bd', capsize=5, linewidth=2)
        axes1[0].axhline(y=baseline_acc, color='#1f77b4', linestyle='--', linewidth=2, label=f'Baseline Intatta ({baseline_acc:.4f})')
        axes1[0].set_title('Accuratezza vs Percentuale di Errore', fontsize=13, fontweight='bold')
        axes1[0].set_xlabel('Record incoerenti (%)', fontsize=11)
        axes1[0].set_ylabel('Accuratezza Media', fontsize=11)
        axes1[0].set_xticks(perc_labels)
        axes1[0].legend(loc='lower left')
        
        x = np.arange(len(perc_labels))
        width = 0.35
        axes1[1].bar(x - width/2, delta_target, width, label='Perdita Target Flipping', color='#d62728', alpha=0.85)
        axes1[1].bar(x + width/2, delta_elo, width, label='Perdita Incoerenza Elo-Rank', color='#9467bd', alpha=0.85)
        axes1[1].set_title("Calo di Performance (Delta %)", fontsize=13, fontweight='bold')
        axes1[1].set_xlabel('Record incoerenti (%)', fontsize=11)
        axes1[1].set_ylabel('Perdita di Accuratezza rispetto alla Baseline (Delta %)', fontsize=11)
        axes1[1].set_xticks(x)
        axes1[1].set_xticklabels(perc_labels)
        axes1[1].legend(loc='upper left')
        
        axes1[1].text(x[-1] - width/2, delta_target[-1] + 0.3, f"{delta_target[-1]:.1f}%", ha='center', fontsize=10, fontweight='bold')
        axes1[1].text(x[-1] + width/2, delta_elo[-1] + 0.3, f"{delta_elo[-1]:.1f}%", ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        path_fig1 = os.path.join(plots_dir, "coerenza_accuratezza_vs_delta.png")
        plt.savefig(path_fig1, dpi=300)
        print(f"Grafico delle performance salvato in: {path_fig1}")
        
        # --- FIGURA 2: LOSS CURVE RUNTIME ---
        if loss_target_50 is not None and len(loss_target_50) > 0:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(loss_baseline, label=f'Baseline: Convergenza in {epoche_baseline} epoche', color='#2ca02c', linewidth=2.5)
            ax2.plot(loss_target_50, label=f'Target Flipping 50%: Convergenza in {epoche_target_50} epoche', color='#d62728', linewidth=2.5, linestyle='--')
            ax2.set_title("Analisi di Runtime: Come cambia l'apprendimento (Loss Curve)", fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoche di addestramento', fontsize=12)
            ax2.set_ylabel('Error Loss', fontsize=12)
            ax2.legend()
            
            plt.tight_layout()
            path_fig2 = os.path.join(plots_dir, "coerenza_loss_curve.png")
            plt.savefig(path_fig2, dpi=300)
            print(f"Grafico delle Loss Curve salvato in: {path_fig2}")
        
        plt.show()

    except FileNotFoundError:
        print(f"\nERRORE: file {dataset_path} non trovato")
    except Exception as e:
        print(f"\nErrore durante l'esecuzione: {e}")