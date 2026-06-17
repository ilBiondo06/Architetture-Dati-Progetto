Questo progetto esplora l'impatto della Data Quality sulle performance di un Multilayer Perceptron addestrato per prevedere l'esito dei match di tennis ATP. Dopo aver massimizzato l'accuratezza su dati ideali, abbiamo ingegnerizzato dei Noise Injectors per corrompere i dati e misurare la Fault Tolerance del modello attraverso quattro dimensioni di qualità: Coerenza, Accuratezza, Completezza e Tempestività.

MEMBRI DEL GRUPPO
- Baiardi Davide 894430
- Passarelli Beatrice 
- Brandino Alessandro 

per l'esecuzione di qualsiasi script è necessario spostarsi nella cartella /src del progetto, in ordine sono stati eseguiti gli script:

1. `baseline_data_quality.py` : questo script genererà il file `baseline_metrics.json` nella cartella results/

2. `coerenza.py`, `accuratezza.py`, `completezza.py`: Questi script genereranno i rispettivi file .json dei risultati e salveranno i CSV sporchi in data/dirty/, oltre ai file `risultati_coerenza.json`, `risultati_accuratezza.json`, `risultati_completezza.json`. E' importante specificare il fatto che l'esecuzione di tutti e tre i file contemporaneamente è di circa 3 ore totali, data la dimensione dei dati e dall'utilizzo di una rete neurale

3. `plot_coerenza.py`, `plot_accuratezza.py`, `plot_completezza.py`: Questi script generano tutte le immagini PNG contenenti i plot, queste immagini verranno salvate all'interno di results/plots/. nel report presente nella cartella doc/ sono state inserite solo le immagini più significative, comunque tutte le immagini sono disponibili nella cartella

4. `analisi_feature.py`: Questo script leggerà i CSV sporchi precedentemente salvati, generando le griglie di evoluzione nella cartella dei results/plots

