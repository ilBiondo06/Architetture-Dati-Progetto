import os
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

percorso_originale = os.path.join(dir_clean, "atp_matches_2000_2024_cleaned.csv")

# ==========================================
# 2. CARICAMENTO E ANALISI DATASET CLEANED
# ==========================================
print("Caricamento dataset Cleaned di base...")
df_originale = pd.read_csv(percorso_originale)
colonne_clean = set(df_originale.columns)

date_clean = pd.to_datetime(df_originale['tourney_date'], format='%Y%m%d', errors='coerce').dropna()
if not date_clean.empty:
    data_min_clean = date_clean.min().strftime('%Y-%m-%d')
    data_max_clean = date_clean.max().strftime('%Y-%m-%d')
else:
    data_min_clean, data_max_clean = "N/D", "N/D"

print("Generazione del report per il dataset Cleaned...")
report_originale = ProfileReport(df_originale, title="Dataset Cleaned (Base)", progress_bar=False)

description_clean = report_originale.get_description()
info_clean = description_clean.table
n_righe_clean = info_clean.get("n_records", info_clean.get("n")) if isinstance(info_clean, dict) else getattr(info_clean, "n_records", getattr(info_clean, "n", 0))

dati_riepilogo = []

# ==========================================
# 3. CICLO SU TUTTI I FILE DIRTY
# ==========================================
file_sporchi = [f for f in os.listdir(dir_dirty) if f.endswith('.csv')]

print(f"\nTrovati {len(file_sporchi)} dataset sporchi. Inizio estrazione metriche avanzate...\n")

for file_name in file_sporchi:
    percorso_modificato = os.path.join(dir_dirty, file_name)
    print(f"--- Analizzando: {file_name} ---")
    
    df_modificato = pd.read_csv(percorso_modificato)
    colonne_dirty = set(df_modificato.columns)
    
    colonne_perse = colonne_clean - colonne_dirty
    num_colonne_perse = len(colonne_perse)
    esempi_colonne = ", ".join(list(colonne_perse)[:3]) if num_colonne_perse > 0 else "Nessuna"
    
    # --- LA CORREZIONE È QUI: Controlliamo se la colonna esiste prima di leggerla ---
    if 'tourney_date' in df_modificato.columns:
        date_dirty = pd.to_datetime(df_modificato['tourney_date'], format='%Y%m%d', errors='coerce').dropna()
        data_max_dirty = date_dirty.max().strftime('%Y-%m-%d') if not date_dirty.empty else "N/D"
    else:
        data_max_dirty = "Colonna ELIMINATA"
        
    media_age_clean = df_originale['winner_age'].mean() if 'winner_age' in df_originale.columns else 0
    media_age_dirty = df_modificato['winner_age'].mean() if 'winner_age' in df_modificato.columns else 0
    shift_eta = round(media_age_dirty - media_age_clean, 2)
    
    report_modificato = ProfileReport(df_modificato, title=f"Dataset Dirty ({file_name})", progress_bar=False)
    info_dirty = report_modificato.get_description().table
    n_righe_dirty = info_dirty.get("n_records", info_dirty.get("n")) if isinstance(info_dirty, dict) else getattr(info_dirty, "n_records", getattr(info_dirty, "n", 0))
    
    righe_perse = n_righe_clean - n_righe_dirty
    percentuale_righe_perse = round((righe_perse / n_righe_clean) * 100, 2) if n_righe_clean > 0 else 0
    
    nome_breve = file_name.replace("dataset_timeliness_", "").replace(".csv", "")
    
    dati_riepilogo.append({
        "Dataset": nome_breve,
        "Righe Totali": n_righe_dirty,
        "Righe Perse": righe_perse,
        "% Righe Perse": percentuale_righe_perse,
        "N. Col. Perse": num_colonne_perse,
        "Data Max Rilevata": data_max_dirty,
        "Shift Media Età": f"{shift_eta:+} anni" if shift_eta != 0 else "0.0",
        "Esempi Col. Perse": esempi_colonne
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
    "Dataset": "CLEANED_BASE", "Righe Totali": n_righe_clean, "Righe Perse": 0,
    "% Righe Perse": 0.0, "N. Col. Perse": 0, "Data Max Rilevata": data_max_clean,
    "Shift Media Età": "Riferimento Base", "Esempi Col. Perse": "Nessuna"
}])
df_riepilogo = pd.concat([riga_clean, df_riepilogo], ignore_index=True)

percorso_tabella = os.path.join(dir_results, "tabella_sintesi_avanzata.csv")
df_riepilogo.to_csv(percorso_tabella, index=False)

print("\n" + "="*80)
print(" 📊 TABELLA DI COMPLEMENTO SCIENTIFICO PER LE SLIDE")
print("="*80)
print(df_riepilogo.to_string(index=False))
print("="*80)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(11, 5))

sns.barplot(data=df_riepilogo, x="Dataset", y="Righe Totali", hue="Dataset", palette="Blues_r", legend=False)
plt.axhline(n_righe_clean, color="red", linestyle="--", linewidth=2, label=f"Livello Base Cleaned ({n_righe_clean} righe)")

plt.title("Impatto del Timeliness Lag sulla Volumetria del Dataset", fontsize=12, fontweight='bold')
plt.xlabel("Configurazione Dataset")
plt.ylabel("Numero di Righe Totali")
plt.legend(loc="lower left")
plt.tight_layout()

percorso_grafico = os.path.join(dir_results, "grafico_perdita_righe.png")
plt.savefig(percorso_grafico, dpi=300)

print(f"\n Elenco completo elaborato con successo!")
print(f" 🖼️ Nuovo grafico salvato: {percorso_grafico}")
print(f" 📄 Tabella avanzata pronta: {percorso_tabella}\n")