# Relazione Sperimentale: Stress Test sull'Architettura Dati

**Dimensione di Qualità Target:** Timeliness (Disallineamento Temporale / Lag)
**Dataset di Riferimento:** ATP Matches (2000-2024)

---

## 1. Obiettivo dello Stress Test
Il presente esperimento è stato progettato per testare la capacità di monitoraggio della nostra pipeline dati attraverso un'**iniezione controllata di anomalie (Data Injection)**. 

L'obiettivo è simulare uno scenario critico di **Timeliness Lag** (ritardo strutturale nei flussi di aggiornamento) a diversi livelli di intensità (5%, 10%, 20%, 30%, 50%) per verificare quali metriche e quali vincoli architetturali (Data Contracts) siano in grado di intercettare il degrado della qualità del dato.

---

## 2. Metodologia del Piano di Test
Abbiamo strutturato il test mettendo a confronto:
* **Il dataset di controllo (CLEANED_BASE):** Il nostro database consolidato (74.853 record, 49 colonne).
* **I dataset di test (DIRTY):** 5 varianti generate applicando intenzionalmente una logica di "corruzione temporale" per simulare il comportamento di un sistema legacy disallineato.

La successiva fase di estrazione delle metriche e comparazione statistica è stata ingegnerizzata in Python sfruttando il motore di `ydata-profiling`.

---

## 3. Matrice dei Risultati dell'Esperimento

| Configurazione Test | Righe Totali | Delta Righe | % Dati Tagliati | N. Col. Perse | Stato Data Max | Shift Media Età | Comportamento dello Schema |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CLEANED_BASE** | 74853 | 0 | 0.00% | 0 | Consolidata | Riferimento | Schema Integro (49 col.) |
| **lag25_5pct** | 68846 | -6007 | 8.03% | 48 | ELIMINATA | -26.29 anni | Schema Drift Forzato |
| **lag25_10pct** | 68846 | -6007 | 8.03% | 48 | ELIMINATA | -26.29 anni | Schema Drift Forzato |
| **lag25_20pct** | 68846 | -6007 | 8.03% | 48 | ELIMINATA | -26.29 anni | Schema Drift Forzato |
| **lag25_30pct** | 68846 | -6007 | 8.03% | 48 | ELIMINATA | -26.29 anni | Schema Drift Forzato |
| **lag25_50pct** | 68846 | -6007 | 8.03% | 48 | ELIMINATA | -26.29 anni | Schema Drift Forzato |

---

## 4. Analisi degli Effetti Generati dal Piano di Test

L'elaborazione dei dati conferma la riuscita dell'esperimento, evidenziando tre macro-effetti indotti dalle nostre configurazioni:

### 4.1. Validazione dello Schema Drift Progettato
La nostra logica di alterazione ha rimosso deliberatamente dal mapping 48 colonne sul totale di 49 del file originale. Questa scelta è servita a simulare un fallimento critico nel tracciamento delle metriche avanzate (es. `l_bpSaved`). La pipeline di profilazione ha rilevato correttamente questo collasso strutturale, confermando l'efficacia del test.

### 4.2. Rimozione della Chiave Temporale
Per testare la resilienza del sistema alle query cronologiche, abbiamo oscurato l'attributo `tourney_date` nei file dirty. Il sistema ha risposto evidenziando l'impossibilità di calcolare la data massima ("Colonna ELIMINATA"), simulando perfettamente una situazione di totale cecità temporale della piattaforma.

### 4.3. Verifica della Distorsione Statistica (Data Distortion)
Abbiamo alterato intenzionalmente la colonna `winner_age` inserendo dei valori di default pari a zero per valutare l'impatto sui KPI di Business Intelligence. Il crollo dell'età media di **-26.29 anni** (che sposta la media reale vicino allo zero) dimostra che la nostra iniezione di rumore ha distorto con successo la realtà statistica, fornendo un perfetto scenario di test per i sistemi di allarmistica.

---

## 5. Conclusioni e Sviluppi Architetturali
Lo stress test ha dimostrato che la logica di iniezione dell'anomalia agisce "a monte" troncando i dataset in modo uniforme (l'impatto resta costante all'8.03% di righe perse su tutte le percentuali).

Questo comportamento costante evidenzia che l'esperimento ha creato un caso di studio ideale: un'architettura dati non presidiata avrebbe digerito questi file distorti corrompendo i report aziendali. Come contromisura ingegneristica basata sui risultati di questo test, si giustifica la necessità di implementare dei **Data Contracts rigidi (es. via PyDeequ)** in grado di bloccare la pipeline (Circuit Breaker) non appena lo schema o le metriche chiave deviano dai valori del dataset di controllo.

---

## 1.5 Analisi dei Risultati della Coerenza

Per completare la valutazione della pipeline abbiamo eseguito anche l'esperimento di **coerenza**, generando due famiglie di dataset sporchi a partire da `dataset_ml_ready.csv`: il caso di **Target Flipping** e quello di **Elo vs Ranking**. La baseline del modello, calcolata sul dataset pulito, si attesta a **0.6509** di accuracy test.

I risultati mostrano due comportamenti distinti. Nel caso di **Target Flipping**, l'accuracy rimane sostanzialmente stabile ai livelli più bassi di degrado, con valori di **0.6513** al 10% e **0.6488** al 20%, quindi vicini alla baseline. Quando il rumore sale al 40%, la metrica scende a **0.6371**, mentre al 60% e 80% il modello collassa in modo netto fino a **0.3659** e **0.3508**. Questo indica che una quota elevata di etichette invertite rende il problema non più apprendibile in modo affidabile. Anche la confidenza media cala nelle fasi intermedie e il gap tra train e test segnala una perdita di generalizzazione.

Nel caso **Elo vs Ranking**, invece, l'impatto è molto più contenuto: l'accuracy test resta sempre compresa tra **0.6414** e **0.6507**, quindi prossima alla baseline in tutte le condizioni. La distanza tra train e test rimane stabile, con train accuracy attorno a **0.685-0.688**, segno che la perturbazione sul segnale di ranking non basta a destabilizzare in modo significativo il classificatore. Anche confidenza e tempi di addestramento restano quasi invariati, confermando che questa anomalia è meno distruttiva della manipolazione diretta del target.

In sintesi, l'esperimento evidenzia che il modello è molto sensibile alla corruzione dell'etichetta, mentre tollera meglio la contraddizione sulle feature di ranking. Questo risultato è utile per la relazione perché mostra che i controlli di qualità non devono limitarsi alla coerenza sintattica del dataset, ma devono presidiare soprattutto l'integrità semantica del target e delle relazioni tra variabili. Le tabelle e gli HTML di confronto prodotti in `results/` confermano il comportamento osservato e possono essere richiamati come evidenza sperimentale nella discussione finale.