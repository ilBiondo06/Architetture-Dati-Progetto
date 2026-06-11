import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from baseline_data_quality import train_and_evaluate_mlp, get_train_test_split, get_or_compute_baseline, load_baseline_cache

# ==========================================
# FUNZIONI DI DEGRADO (ACCURATEZZA)
# ==========================================

def eta_altezza_outliers(X, percentage):
    """
    ESPERIMENTO A: Outliers Estremi
    Inserisce differenze fisicamente impossibili (+- 150cm, +- 80 anni).
    """
    X_dirty = X.copy()
    n_changes = int(len(X_dirty) * percentage)
    
    idx = np.random.choice(X_dirty.index, n_changes, replace=False)
    X_dirty.loc[idx, 'diff_ht'] = np.random.choice([150.0, -150.0], size=n_changes)
    X_dirty.loc[idx, 'diff_age'] = np.random.choice([80.0, -80.0], size=n_changes)
    
    return X_dirty

def rumore_gaussiano(X, percentage):
    """
    ESPERIMENTO B: Rumore Gaussiano
    Applica una distorsione statistica (Gaussian Noise) alla colonna DIFF_ELO.
    Simula misurazioni imprecise o stime errate della forza dei giocatori.
    """
    X_dirty = X.copy()
    n_changes = int(len(X_dirty) * percentage)
    
    if 'DIFF_ELO' in X_dirty.columns:
        idx = np.random.choice(X_dirty.index, n_changes, replace=False)
        # Calcoliamo la deviazione standard originale e la moltiplichiamo per creare un rumore forte
        std_dev = X_dirty['DIFF_ELO'].std()
        noise = np.random.normal(loc=0.0, scale=std_dev * 1.5, size=n_changes)
        X_dirty.loc[idx, 'DIFF_ELO'] += noise
        
    return X_dirty

# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================

if __name__ == "__main__":
    print("--- INIZIO PIPELINE: ESPERIMENTI DI ACCURATEZZA ---")
    
    base_folder = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(base_folder, "..", "data", "clean")
    dirty_dir = os.path.join(base_folder, "..", "data", "dirty")
    plots_dir = os.path.join(base_folder, "..", "docs", "plots")
    
    os.makedirs(dirty_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    dataset_path = os.path.join(clean_dir, "dataset_ml_ready.csv") 
    
    try:
        baseline_acc = get_or_compute_baseline(dataset_path)
        baseline_cache = load_baseline_cache()
        if baseline_cache:
            loss_baseline = baseline_cache["loss_curve"]
            epoche_baseline = baseline_cache["n_iter"]
            print(f"Baseline caricata da JSON: {baseline_acc:.4f} (Convergenza in {epoche_baseline} epoche).\n")
        else:
            loss_baseline, epoche_baseline = [], 0
            
        X_train_clean, y_train_clean, X_test_clean, y_test_clean = get_train_test_split(dataset_path)
        
        percentuali = [0.05, 0.10, 0.20, 0.30, 0.50]  
        N_RUNS = 10  
        risultati = {'Outliers': {'media': [], 'std': []}, 'Gaussiano': {'media': [], 'std': []}}
        
        loss_target_50 = None
        epoche_target_50 = 0

        # ESPERIMENTO A: OUTLIERS
        print(f"--- AVVIO ESPERIMENTO A: OUTLIERS ETÀ-ALTEZZA | {N_RUNS} RUNS ---")
        for p in percentuali:
            run_accuracies = [] 
            for run in tqdm(range(N_RUNS), desc=f"Degrado Outliers {p*100:0.0f}%", colour='cyan'):
                X_stale = eta_altezza_outliers(X_train_clean, percentage=p)
                
                if run == 0:
                    df_to_save = X_stale.copy()
                    df_to_save['target'] = y_train_clean
                    df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_accuratezza_outliers_{int(p*100)}pct.csv"), index=False)
                    
                    if p == 0.50:
                        acc, mlp_dirty = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean, return_model=True)
                        loss_target_50 = mlp_dirty.loss_curve_
                        epoche_target_50 = mlp_dirty.n_iter_
                    else:
                        acc = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
                else:
                    acc = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
                run_accuracies.append(acc)
            
            m, s = np.mean(run_accuracies), np.std(run_accuracies)
            risultati['Outliers']['media'].append(m); risultati['Outliers']['std'].append(s)
            print(f"  -> Accuracy: {m:.4f} ± {s:.4f} | Delta: -{(baseline_acc - m)*100:.2f}%\n")

        # ESPERIMENTO B: RUMORE GAUSSIANO
        print(f"--- AVVIO ESPERIMENTO B: RUMORE GAUSSIANO | {N_RUNS} RUNS ---")
        for p in percentuali:
            run_accuracies = [] 
            for run in tqdm(range(N_RUNS), desc=f"Degrado Gauss {p*100:0.0f}%", colour='blue'):
                X_stale = rumore_gaussiano(X_train_clean, percentage=p)
                if run == 0:
                    df_to_save = X_stale.copy()
                    df_to_save['target'] = y_train_clean
                    df_to_save.to_csv(os.path.join(dirty_dir, f"dataset_accuratezza_gaussiano_{int(p*100)}pct.csv"), index=False)
                acc = train_and_evaluate_mlp(X_stale, y_train_clean, X_test_clean, y_test_clean)
                run_accuracies.append(acc)
            
            m, s = np.mean(run_accuracies), np.std(run_accuracies)
            risultati['Gaussiano']['media'].append(m); risultati['Gaussiano']['std'].append(s)
            print(f"  -> Accuracy: {m:.4f} ± {s:.4f} | Delta: -{(baseline_acc - m)*100:.2f}%\n")

        # ================= SALVATAGGIO RISULTATI =================
        import json
        
        # Convertiamo i float di NumPy (float64) in float nativi di Python
        risultati_standardizzati = {
            'Outliers': {
                'media': [float(m) for m in risultati['Outliers']['media']],
                'std': [float(s) for s in risultati['Outliers']['std']]
            },
            'Gaussiano': {
                'media': [float(m) for m in risultati['Gaussiano']['media']],
                'std': [float(s) for s in risultati['Gaussiano']['std']]
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
        
        path_cache_risultati = os.path.join(plots_dir, "risultati_accuratezza.json")
        with open(path_cache_risultati, "w") as f:
            json.dump(dati_da_salvare, f, indent=4)
        print(f"\n[OK] Risultati degli esperimenti salvati in: {path_cache_risultati}")

        # ================= GRAFICI =================
        print("\nGenerazione grafici in corso...")
        sns.set_theme(style="whitegrid")
        perc_labels = [int(p * 100) for p in percentuali]
        delta_outliers = [(baseline_acc - m) * 100 for m in risultati['Outliers']['media']]
        delta_gauss = [(baseline_acc - m) * 100 for m in risultati['Gaussiano']['media']]
        
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
        axes1[0].errorbar(perc_labels, risultati['Outliers']['media'], yerr=risultati['Outliers']['std'], fmt='-o', label='A: Outliers Estremi', color='#17becf', capsize=5, linewidth=2)
        axes1[0].errorbar(perc_labels, risultati['Gaussiano']['media'], yerr=risultati['Gaussiano']['std'], fmt='-s', label='B: Rumore Gaussiano', color='#1f77b4', capsize=5, linewidth=2)
        axes1[0].axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.4f})')
        axes1[0].set_title('Accuratezza: Impatto degli Errori di Misura', fontsize=13, fontweight='bold')
        axes1[0].set_xlabel('Percentuale di record corrotti (%)'); axes1[0].set_ylabel('Accuracy')
        axes1[0].set_xticks(perc_labels); axes1[0].legend()
        
        x = np.arange(len(perc_labels)); width = 0.35
        axes1[1].bar(x - width/2, delta_outliers, width, label='Perdita Outliers', color='#17becf', alpha=0.85)
        axes1[1].bar(x + width/2, delta_gauss, width, label='Perdita Gaussiano', color='#1f77b4', alpha=0.85)
        axes1[1].set_title("Calo di Performance (Delta %)", fontsize=13, fontweight='bold')
        axes1[1].set_xlabel('Percentuale di record corrotti (%)'); axes1[1].set_ylabel('Perdita (%)')
        axes1[1].set_xticks(x); axes1[1].set_xticklabels(perc_labels); axes1[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "accuratezza_performance.png"), dpi=300)
        
        if loss_target_50 is not None:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(loss_baseline, label=f'Baseline (Sana - {epoche_baseline} epoche)', color='gray', linewidth=2.5)
            ax2.plot(loss_target_50, label=f'Outliers 50% (Arresto a {epoche_target_50} epoche)', color='#17becf', linewidth=2.5, linestyle='--')
            ax2.set_title("Analisi Runtime: Loss Curve (Accuratezza)", fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoche'); ax2.set_ylabel('Loss'); ax2.legend()
            plt.savefig(os.path.join(plots_dir, "accuratezza_loss_curve.png"), dpi=300)
        
        plt.show()

    except Exception as e:
        print(f"\nErrore: {e}")