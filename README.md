# Transportation Mode Detection

## proprocessing

Necessita avere il file **geolife.pkl** che contenga tutto il dataset Geolife originale

- Eseguire il file preprocessing.py

Viene creato il file: geolife_mov_with_features_correct2.pkl

Il quale contiene il dataset delle sole **mov** pronto per essere implementato nel file CNN.py per eseguire i calcoli della rete neurale.

Inoltre vengono creati i seguenti file: traj_correct2.pkl, mov.pkl, stop.pkl

- Eseguire il file preprocessing_stop_mov.py

il quale necessita il file **traj_correct2.pkl** che contiene le stop e le mov ma senza le features cinematiche

Poi viene creato il file: geolife_stop_mov_features_correct.pkl

Che contiene le **stop** e le **mov**

Adesso siamo pronti per eseguire la CNN

## CNN

impostare i parametri del modello CNN sul file config.yaml

- la variabile **flag** serve a selezionare il file con le sole mov o quello con le stop e le mov

- la variabile **features** indica quali features cinematiche vogliamo utilizzare

Poi eseguire direttamente il file CNN.py che deve printare i risultati del modello e salvare i risultati in un file excel.
