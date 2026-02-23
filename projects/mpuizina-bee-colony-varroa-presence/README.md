# Bee Colony Varroa Presence GLMM

Projekt je baziran na znanstvenom radu [_Climatic predictors of prominent honey bee (Apis mellifera) disease agents_](https://journals.plos.org/climate/article?id=10.1371/journal.pclm.0000485).

Cilj projekta je predstaviti problem koji parazit _Varroa destructor_ predstavlja pčelarstvu, a zatim iskoristiti podatke koje su autori znanstvenog rada javno objavili zajedno s R kodom za statističku analizu i na temelju toga opisati put od linearne regresije do jednog GLMM-a (_Generalized Linear Mixed Model_) kojeg su istraživači koristili u radu.

Projekt je napravljen u Pythonu unutar **Marimo** bilježnice, a kako bi se interakcija Pythona i R-a mogla vjerno reproducirati koristi se alat **Anaconda** (dovoljna je **Miniconda** instalacija).

Za pokretanje projekta izvršite sljedeće tri naredbe:

```
conda env create -f environment.yml
conda activate zp_projekt
marimo edit project.py
```

Alternativna naredba `marimo run project.py` na mom računalu uredno otvara i izvršava bilježnicu, ali R dependency počne stvarati probleme s terminalom u VS Code-u zbog čega se ne može izaći sa Ctrl+C pa je potrebno "na silu" zatvoriti terminal. Zato je lakše otvoriti s `marimo edit project.py` i ući u App view nakon pokretanja.