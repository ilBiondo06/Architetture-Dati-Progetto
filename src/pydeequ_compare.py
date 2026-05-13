import os
import pandas as pd
from ydata_profiling import ProfileReport

# ==========================================
# 1. GESTIONE DEI PERCORSI (src -> data)
# ==========================================
# Trova la cartella 'src' (dove si trova questo script)
dir_corrente = os.path.dirname(os.path.abspath(__file__))

# Risali di un livello e scendi nella cartella 'data'
dir_dati = os.path.join(dir_corrente, "..", "data")

# Usa i file che mi hai indicato
percorso_originale = os.path.join(dir_dati, "atp_matches_2000_2024_cleaned.csv") 
percorso_modificato = os.path.join(dir_dati, "atp_matches_2000_2024_raw.csv")

print("Caricamento dei dataset in corso...")

# ==========================================
# 2. CARICAMENTO DATI CON PANDAS
# ==========================================
# Pandas legge i CSV in modo molto più semplice rispetto a Spark
df_originale = pd.read_csv(percorso_originale)
df_modificato = pd.read_csv(percorso_modificato)

print(f"File Originale (Cleaned) caricato: {df_originale.shape[0]} righe, {df_originale.shape[1]} colonne")
print(f"File Modificato (Raw) caricato: {df_modificato.shape[0]} righe, {df_modificato.shape[1]} colonne")

# ==========================================
# 3. GENERAZIONE DEI REPORT
# ==========================================
print("\nGenerazione del report per il dataset Originale...")
# Creiamo il primo report. 
report_originale = ProfileReport(df_originale, title="Dataset Cleaned (Originale)")

print("Generazione del report per il dataset Modificato...")
# Creiamo il secondo report.
report_modificato = ProfileReport(df_modificato, title="Dataset Raw (Modificato)")

# ==========================================
# 4. CONFRONTO E SALVATAGGIO
# ==========================================
print("\nCalcolo delle differenze (potrebbe volerci qualche istante)...")
# Confrontiamo i due report
report_confronto = report_originale.compare(report_modificato)

# Definiamo dove salvare il report finale (lo salviamo nella root del progetto)
percorso_output = os.path.join(dir_corrente, "..", "confronto_atp_matches.html")

# Salviamo il risultato in un file HTML interattivo
report_confronto.to_file(percorso_output)

print(f"\n🎉 Finito! Vai nella cartella principale del tuo progetto e apri il file:")
print(f"👉 {os.path.abspath(percorso_output)}")
print("Puoi aprirlo con Chrome, Edge o Firefox per esplorare le differenze!")