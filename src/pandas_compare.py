import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# ==========================================
# 1. GESTIONE DEI PERCORSI
# ==========================================
dir_corrente = os.path.dirname(os.path.abspath(__file__))

dir_clean = os.path.join(dir_corrente, "..", "data", "clean")
dir_dirty = os.path.join(dir_corrente, "..", "data", "dirty")
dir_results = os.path.join(dir_corrente, "..", "results")

os.makedirs(dir_results, exist_ok=True)

percorso_originale = os.path.join(dir_clean, "dataset_ml_ready.csv")


def estrai_info_coerenza(nome_file):
    match = re.match(r"dataset_coerenza_(targetflip|elorank)_(\d+)pct\.csv$", nome_file)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def safe_mean(df, column_name):
    if column_name not in df.columns:
        return None
    valori = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if valori.empty:
        return None
    return float(valori.mean())

# ==========================================
# 2. CARICAMENTO E ANALISI DATASET CLEANED
# ==========================================
print("Caricamento dataset Cleaned di base per l'analisi di coerenza...")
df_originale = pd.read_csv(percorso_originale)
colonne_clean = set(df_originale.columns)

target_mean_clean = safe_mean(df_originale, "target")
rank_diff_mean_clean = safe_mean(df_originale, "ATP_RANK_DIFF")

print("Generazione del report per il dataset Cleaned di riferimento...")
report_originale = ProfileReport(df_originale, title="Dataset Cleaned (Base)", progress_bar=False)

description_clean = report_originale.get_description()
info_clean = description_clean.table
n_righe_clean = info_clean.get("n_records", info_clean.get("n")) if isinstance(info_clean, dict) else getattr(info_clean, "n_records", getattr(info_clean, "n", 0))

dati_riepilogo = []

# ==========================================
# 3. CICLO SU TUTTI I FILE DIRTY
# ==========================================
file_sporchi = sorted(
    f for f in os.listdir(dir_dirty)
    if f.startswith("dataset_coerenza_") and f.endswith(".csv")
)

print(f"\nTrovati {len(file_sporchi)} dataset sporchi di coerenza. Inizio estrazione metriche avanzate...\n")

if not file_sporchi:
    print("Nessun file di coerenza trovato in data/dirty. Termino senza generare output.")
    raise SystemExit(0)

for file_name in file_sporchi:
    percorso_modificato = os.path.join(dir_dirty, file_name)
    print(f"--- Analizzando: {file_name} ---")
    
    df_modificato = pd.read_csv(percorso_modificato)
    colonne_dirty = set(df_modificato.columns)
    esperimento, percentuale = estrai_info_coerenza(file_name)
    
    colonne_perse = colonne_clean - colonne_dirty
    colonne_extra = colonne_dirty - colonne_clean
    num_colonne_perse = len(colonne_perse)
    esempi_colonne = ", ".join(list(colonne_perse)[:3]) if num_colonne_perse > 0 else "Nessuna"
    esempi_extra = ", ".join(list(colonne_extra)[:3]) if colonne_extra else "Nessuna"

    target_mean_dirty = safe_mean(df_modificato, "target")
    rank_diff_mean_dirty = safe_mean(df_modificato, "ATP_RANK_DIFF")

    shift_target = None
    if target_mean_clean is not None and target_mean_dirty is not None:
        shift_target = round(target_mean_dirty - target_mean_clean, 4)

    shift_rank = None
    if rank_diff_mean_clean is not None and rank_diff_mean_dirty is not None:
        shift_rank = round(rank_diff_mean_dirty - rank_diff_mean_clean, 4)
    
    report_modificato = ProfileReport(df_modificato, title=f"Dataset Dirty ({file_name})", progress_bar=False)
    info_dirty = report_modificato.get_description().table
    n_righe_dirty = info_dirty.get("n_records", info_dirty.get("n")) if isinstance(info_dirty, dict) else getattr(info_dirty, "n_records", getattr(info_dirty, "n", 0))
    
    righe_perse = n_righe_clean - n_righe_dirty
    percentuale_righe_perse = round((righe_perse / n_righe_clean) * 100, 2) if n_righe_clean > 0 else 0
    
    nome_breve = file_name.replace("dataset_", "").replace(".csv", "")
    
    dati_riepilogo.append({
        "Dataset": nome_breve,
        "Tipo Coerenza": esperimento if esperimento is not None else "N/D",
        "% Coerenza": percentuale if percentuale is not None else "N/D",
        "Righe Totali": n_righe_dirty,
        "Righe Perse": righe_perse,
        "% Righe Perse": percentuale_righe_perse,
        "N. Col. Perse": num_colonne_perse,
        "Colonne Aggiunte": len(colonne_extra),
        "Shift Media Target": shift_target if shift_target is not None else "N/D",
        "Shift Media ATP_RANK_DIFF": shift_rank if shift_rank is not None else "N/D",
        "Esempi Col. Perse": esempi_colonne,
        "Esempi Col. Aggiunte": esempi_extra
    })
    
    report_confronto = report_originale.compare(report_modificato)
    nome_html = f"confronto_{file_name.replace('.csv', '')}.html"
    report_confronto.to_file(os.path.join(dir_results, nome_html))
    print(f" Report HTML salvato: {nome_html}\n")

# ==========================================
# 4. CREAZIONE DEL REPORT FINALE E GRAFICI
# ==========================================
df_riepilogo = pd.DataFrame(dati_riepilogo)

riga_clean = pd.DataFrame([{
    "Dataset": "CLEANED_BASE",
    "Tipo Coerenza": "Riferimento",
    "% Coerenza": 0,
    "Righe Totali": n_righe_clean,
    "Righe Perse": 0,
    "% Righe Perse": 0.0,
    "N. Col. Perse": 0,
    "Colonne Aggiunte": 0,
    "Shift Media Target": 0.0,
    "Shift Media ATP_RANK_DIFF": 0.0,
    "Esempi Col. Perse": "Nessuna",
    "Esempi Col. Aggiunte": "Nessuna"
}])
df_riepilogo = pd.concat([riga_clean, df_riepilogo], ignore_index=True)

percorso_tabella = os.path.join(dir_results, "tabella_sintesi_coerenza.csv")
df_riepilogo.to_csv(percorso_tabella, index=False)

print("\n" + "="*80)
print(" 📊 TABELLA DI COMPLEMENTO SCIENTIFICO PER LA COERENZA")
print("="*80)
print(df_riepilogo.to_string(index=False))
print("="*80)

sns.set_theme(style="whitegrid")
df_plot = df_riepilogo[df_riepilogo["Dataset"] != "CLEANED_BASE"].copy()
df_plot["% Coerenza"] = pd.to_numeric(df_plot["% Coerenza"], errors="coerce")
df_plot["Shift Media Target"] = pd.to_numeric(df_plot["Shift Media Target"], errors="coerce")
df_plot["Shift Media ATP_RANK_DIFF"] = pd.to_numeric(df_plot["Shift Media ATP_RANK_DIFF"], errors="coerce")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

target_flip = df_plot[df_plot["Tipo Coerenza"] == "targetflip"].sort_values("% Coerenza")
elo_rank = df_plot[df_plot["Tipo Coerenza"] == "elorank"].sort_values("% Coerenza")

if not target_flip.empty:
    sns.barplot(data=target_flip, x="% Coerenza", y="Shift Media Target", color="#d62728", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Target Flipping: shift medio del target", fontweight='bold')
    axes[0].set_xlabel("Percentuale di record alterati (%)")
    axes[0].set_ylabel("Delta media target")
else:
    axes[0].set_axis_off()

if not elo_rank.empty:
    sns.barplot(data=elo_rank, x="% Coerenza", y="Shift Media ATP_RANK_DIFF", color="#1f77b4", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Elo vs Ranking: shift medio di ATP_RANK_DIFF", fontweight='bold')
    axes[1].set_xlabel("Percentuale di record alterati (%)")
    axes[1].set_ylabel("Delta media ATP_RANK_DIFF")
else:
    axes[1].set_axis_off()

plt.tight_layout()

percorso_grafico = os.path.join(dir_results, "grafico_coerenza_shift.png")
plt.savefig(percorso_grafico, dpi=300)

print(f"\n Elenco completo elaborato con successo!")
print(f" 🖼️ Nuovo grafico salvato: {percorso_grafico}")
print(f" 📄 Tabella avanzata pronta: {percorso_tabella}\n")