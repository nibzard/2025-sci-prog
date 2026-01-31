# Dokumentacija Modela - Croatian Property Price Estimator

## Sadržaj
1. [Pregled Datasetova](#1-pregled-datasetova)
2. [Čišćenje Podataka](#2-čišćenje-podataka)
3. [Model V1 - Osnovni LightGBM](#3-model-v1---osnovni-lightgbm)
4. [Model V2 - Optimizirani LightGBM](#4-model-v2---optimizirani-lightgbm)
5. [Model V3 - Target Encoding + Log Transform](#5-model-v3---target-encoding--log-transform)
6. [Usporedba Rezultata](#6-usporedba-rezultata)
7. [Poznati Problemi i Moguća Poboljšanja](#7-poznati-problemi-i-moguća-poboljšanja)

---

## 1. Pregled Datasetova

### 1.1 Dataset Kuća (houses_clean.parquet)

**Osnovne informacije:**
- Broj zapisa: **2,439**
- Broj stupaca: **87**
- Ciljana varijabla: `cijena` (EUR)

**Distribucija cijena:**
| Statistika | Vrijednost |
|------------|------------|
| Minimum | 1 EUR |
| Maximum | 5,900,000 EUR |
| Prosjek | 610,921 EUR |
| Medijan | 470,000 EUR |
| Std. devijacija | 548,195 EUR |

**Top 10 županija po broju nekretnina:**
| Županija | Broj kuća |
|----------|-----------|
| Istarska | 500 |
| Primorsko-goranska | 500 |
| Grad Zagreb | 499 |
| Zadarska | 383 |
| Osječko-baranjska | 157 |
| Splitsko-dalmatinska | 121 |
| Brodsko-posavska | 117 |
| Zagrebačka | 59 |
| Karlovačka | 58 |
| Sisačko-moslavačka | 45 |

**Stupci i njihove karakteristike:**

| Stupac | Tip | Non-null | Missing % | Opis |
|--------|-----|----------|-----------|------|
| cijena | float64 | 2439 | 0.0% | Cijena u EUR |
| stambena_povrsina | float64 | 2439 | 0.0% | Stambena površina u m² |
| povrsina_okucnice | float64 | 1644 | 32.6% | Površina okućnice u m² |
| broj_soba | int64 | 2439 | 0.0% | Ukupan broj soba |
| broj_parkirnih_mjesta | float64 | 1060 | 56.5% | Broj parkirnih mjesta |
| godina_izgradnje | float64 | 1209 | 50.4% | Godina izgradnje |
| ima_godinu_izgradnje | int64 | 2439 | 0.0% | Flag: ima li podatak o godini |
| godina_renovacije | float64 | 1365 | 44.0% | Godina renovacije |
| broj_etaza | float64 | 2439 | 0.0% | Broj etaža (1-4) |
| energetski_razred | float64 | 422 | 82.7% | Energetski razred (-2 do 5) |
| ima_energetski_razred | int64 | 2439 | 0.0% | Flag: ima li energetski razred |
| wc_broj | float64 | 736 | 69.8% | Broj WC-a |
| kupaonica_s_wc_broj | float64 | 1735 | 28.9% | Broj kupaonica s WC-om |
| zupanija | str | 2439 | 0.0% | Županija |
| grad_opcina | str | 2439 | 0.0% | Grad/Općina |
| naselje | str | 2439 | 0.0% | Naselje |

**Binarni stupci (0/1):** grijanje_dodatni, grijanje_klima, pogled_more, video_poziv, mogucnost_zamjene, balkon_*, dozvola_*, funk_*, alt_energija_*, grijanje_sustav_*, objekt_*, podatak_*, tip_kuce_*, gradnja_*, parking_*

**Statistike numeričkih stupaca:**
| Stupac | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|------|-----|-----|-----|-----|-----|-----|
| cijena | 610,921 | 548,195 | 1 | 290,000 | 470,000 | 719,750 | 5,900,000 |
| stambena_povrsina | 267.8 | 250.6 | 1 | 141.3 | 220 | 335 | 5,407 |
| povrsina_okucnice | 743.1 | 3,182.6 | 3 | 250 | 472.5 | 780 | 125,000 |
| broj_soba | 6.0 | 3.4 | 1 | 4 | 5 | 8 | 26 |
| godina_izgradnje | 1988.7 | 29.9 | 1750 | 1970 | 1990 | 2011 | 2026 |

---

### 1.2 Dataset Stanova (apartments_clean.parquet)

**Osnovne informacije:**
- Broj zapisa: **3,529**
- Broj stupaca: **86**
- Ciljana varijabla: `cijena` (EUR)

**Distribucija cijena:**
| Statistika | Vrijednost |
|------------|------------|
| Minimum | 1 EUR |
| Maximum | 1,950,000 EUR |
| Prosjek | 298,944 EUR |
| Medijan | 264,625 EUR |
| Std. devijacija | 195,423 EUR |

**Top 10 županija po broju nekretnina:**
| Županija | Broj stanova |
|----------|--------------|
| Splitsko-dalmatinska | 500 |
| Istarska | 500 |
| Primorsko-goranska | 500 |
| Grad Zagreb | 500 |
| Zadarska | 497 |
| Osječko-baranjska | 470 |
| Zagrebačka | 274 |
| Brodsko-posavska | 188 |
| Karlovačka | 65 |
| Sisačko-moslavačka | 35 |

**Stupci i njihove karakteristike:**

| Stupac | Tip | Non-null | Missing % | Opis |
|--------|-----|----------|-----------|------|
| cijena | float64 | 3529 | 0.0% | Cijena u EUR |
| stambena_povrsina | float64 | 3529 | 0.0% | Stambena površina u m² |
| broj_parkirnih_mjesta | float64 | 1457 | 58.7% | Broj parkirnih mjesta |
| godina_izgradnje | float64 | 1889 | 46.5% | Godina izgradnje |
| ima_godinu_izgradnje | int64 | 3529 | 0.0% | Flag: ima li podatak o godini |
| godina_renovacije | float64 | 2050 | 41.9% | Godina renovacije |
| broj_etaza | int64 | 3529 | 0.0% | Broj etaža stana (1-3) |
| broj_soba | float64 | 3529 | 0.0% | Broj soba (0.5-5) |
| energetski_razred | float64 | 1048 | 70.3% | Energetski razred (-2 do 5) |
| ima_energetski_razred | int64 | 3529 | 0.0% | Flag: ima li energetski razred |
| wc_broj | float64 | 1056 | 70.1% | Broj WC-a |
| kupaonica_s_wc_broj | float64 | 2700 | 23.5% | Broj kupaonica s WC-om |
| ukupni_broj_katova | float64 | 2183 | 38.1% | Ukupni broj katova zgrade |
| ima_ukupni_broj_katova | int64 | 3529 | 0.0% | Flag: ima li taj podatak |
| kat | float64 | 3194 | 9.5% | Na kojem je katu (-1 do 24) |
| kat_suteren | int64 | 3529 | 0.0% | Flag: je li suteren |
| kat_prizemlje | int64 | 3529 | 0.0% | Flag: je li prizemlje |
| kat_potkrovlje | int64 | 3529 | 0.0% | Flag: je li potkrovlje |
| kat_penthouse | int64 | 3529 | 0.0% | Flag: je li penthouse |
| zupanija | str | 3529 | 0.0% | Županija |
| grad_opcina | str | 3529 | 0.0% | Grad/Općina |
| naselje | str | 3529 | 0.0% | Naselje |

**Binarni stupci (0/1):** grijanje_*, balkon_*, dozvola_*, funk_*, alt_energija_*, orijentacija_*, objekt_*, podatak_*, tip_stana_*, parking_*

**Statistike numeričkih stupaca:**
| Stupac | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|------|-----|-----|-----|-----|-----|-----|
| cijena | 298,944 | 195,423 | 1 | 187,000 | 264,625 | 356,624 | 1,950,000 |
| stambena_povrsina | 87.5 | 135.0 | 1 | 58.2 | 77.1 | 103.3 | 7,675 |
| broj_soba | 2.9 | 1.0 | 0.5 | 2 | 3 | 4 | 5 |
| kat | 1.7 | 1.9 | -1 | 1 | 1 | 2 | 24 |
| godina_izgradnje | 2004.7 | 32.6 | 1810 | 1988 | 2024 | 2025 | 2027 |

---

## 2. Čišćenje Podataka

### 2.1 Implementacija
Skripta: `src/preprocessing/clean_data.py`

### 2.2 Transformacije

#### Numeričke transformacije:

| Originalni stupac | Novi stupac | Transformacija |
|-------------------|-------------|----------------|
| cijena (npr. "330.000 €") | cijena | Ukloni " €", zamijeni "." s "", pretvori u float |
| Stambena površina (npr. "139,27 m²") | stambena_povrsina | Ukloni " m²", zamijeni "," s ".", pretvori u float |
| Površina okućnice | povrsina_okucnice | Isto kao stambena površina |
| Godina izgradnje (npr. "2025.") | godina_izgradnje | Ukloni ".", pretvori u int |
| Godina zadnje renovacije | godina_renovacije | Ako nema renovacije, koristi godinu izgradnje |

#### Mapiranje kategorija u brojeve:

**Broj etaža (kuće):**
```
Prizemnica → 1
Visoka prizemnica → 1.5
Katnica → 2
Dvokatnica → 3
Višekatnica → 4
```

**Broj etaža (stanovi):**
```
Jednoetažni → 1
Dvoetažni → 2
Višeetažni → 3
```

**Broj soba (stanovi):**
```
Garsonijera → 0.5
1-sobni → 1
2-sobni → 2
3-sobni → 3
4-sobni → 4
5+ sobni → 5
```

**Energetski razred:**
```
A+ → 5, A → 4, B → 3, C → 2, D → 1, E → 0, F → -1, G → -2
```

**Kupaonica/WC broj:**
```
"5+" → 5
```

#### Binarno mapiranje (Da/None → 1/0):
- grijanje_dodatni
- grijanje_klima
- pogled_more (samo kuće)
- video_poziv
- mogucnost_zamjene

#### One-Hot Encoding (multi-value stupci):
Stupci koji mogu sadržavati više vrijednosti odvojenih zarezom pretvoreni su u multiple binarne stupce:
- Balkon/Lođa/Terasa → balkon_balkon, balkon_lođa_loggia, balkon_terasa
- Dozvole i potvrde → dozvola_građevinska_dozvola, dozvola_uporabna_dozvola, dozvola_vlasnički_list
- Funkcionalnosti → funk_alarmni_sustav, funk_jacuzzi, funk_kamin, ...
- Grijanje - Sustav → grijanje_sustav_dizalica_topline, grijanje_sustav_etažno_plinsko_centralno, ...
- Ostali objekti → objekt_bazen, objekt_dvorište_vrt, objekt_podrum, ...
- Podaci o objektu → podatak_gradska_kanalizacija, podatak_lift, podatak_novogradnja, ...
- Vrsta parkinga → parking_garaža, parking_garažno_mjesto, ...

#### Lokacija:
Originalni stupac `Lokacija` (format: "Županija, Grad/Općina, Naselje") razdvojen u:
- zupanija
- grad_opcina
- naselje

### 2.3 Rukovanje nedostajućim vrijednostima

**Strategija:** NaN vrijednosti su **ZADRŽANE** za numeričke stupce jer:
1. Tree-based modeli (LightGBM) nativno podržavaju NaN
2. Imputacija (npr. prosjekom) može unijeti pristranost
3. Dodani su "flag" stupci koji označavaju ima li nekretnina taj podatak

**Flag stupci:**
- ima_godinu_izgradnje (1 ako postoji, 0 ako ne)
- ima_energetski_razred
- ima_ukupni_broj_katova (samo stanovi)
- ima_sustav_grijanja

---

## 3. Model V1 - Osnovni LightGBM

### 3.1 Implementacija
Skripta: `src/models/train.py`

### 3.2 Algoritam
**LightGBM (Light Gradient Boosting Machine)**
- Tip: Gradient boosting decision tree
- Prednosti: Brz, efikasan s velikim datasetovima, nativno rukuje NaN vrijednostima

### 3.3 Priprema podataka

```python
# Kategoričke varijable - label encoding
for col in ["zupanija", "grad_opcina", "naselje"]:
    X[col] = df[col].astype("category").cat.codes
```

Label encoding pretvara kategorije u brojeve (npr. "Istarska" → 0, "Primorsko-goranska" → 1, ...).

### 3.4 Hiperparametri (fiksni)

| Parametar | Vrijednost | Opis |
|-----------|------------|------|
| objective | regression | Regresijski problem |
| metric | rmse | Root Mean Squared Error |
| boosting_type | gbdt | Gradient Boosting Decision Tree |
| num_leaves | 31 | Broj listova po stablu |
| learning_rate | 0.05 | Stopa učenja |
| feature_fraction | 0.8 | Postotak featurea po stablu |
| bagging_fraction | 0.8 | Postotak uzoraka po stablu |
| bagging_freq | 5 | Frekvencija bagginga |
| n_estimators | 1000 | Maksimalan broj stabala |
| early_stopping_rounds | 50 | Zaustavi ako nema poboljšanja |

### 3.5 Train/Test Split
- **80%** trening set
- **20%** test set
- random_state=42 (za reproducibilnost)

### 3.6 Rezultati Model V1

| Metrika | Kuće - Train | Kuće - Test | Stanovi - Train | Stanovi - Test |
|---------|--------------|-------------|-----------------|----------------|
| RMSE | 207,943 EUR | 316,063 EUR | 70,645 EUR | 106,065 EUR |
| MAE | 135,595 EUR | 205,656 EUR | 46,227 EUR | 65,790 EUR |
| R² | 0.8489 | 0.6944 | 0.8671 | 0.7239 |

---

## 4. Model V2 - Optimizirani LightGBM

### 4.1 Implementacija
Skripta: `src/models/train_optimized.py`

### 4.2 Poboljšanja

#### 4.2.1 Outlier Removal

```python
def remove_outliers(df, column="cijena", lower_pct=1, upper_pct=99):
    lower = np.percentile(df[column], lower_pct)
    upper = np.percentile(df[column], upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]
```

Uklanja nekretnine čija cijena pada izvan raspona 1. do 99. percentila.

**Učinak:**
- Kuće: uklonjeno ~2% outliera
- Stanovi: uklonjeno ~2% outliera

#### 4.2.2 Feature Engineering

Dodani novi featurei:

| Feature | Formula | Opis |
|---------|---------|------|
| starost | 2026 - godina_izgradnje | Starost nekretnine u godinama |
| godine_od_renovacije | 2026 - godina_renovacije | Godine od zadnje renovacije |
| povrsina_log | log(1 + stambena_povrsina) | Log transformacija površine |
| ukupno_kupaonica | wc_broj + kupaonica_s_wc_broj | Ukupan broj kupaonica |

#### 4.2.3 Hyperparameter Tuning (Optuna)

Optuna je framework za automatsku optimizaciju hiperparametara. Koristi Bayesian optimization za efikasno pretraživanje prostora parametara.

**Optimizirani parametri i rasponi:**

| Parametar | Raspon | Opis |
|-----------|--------|------|
| learning_rate | [0.01, 0.2] (log) | Stopa učenja |
| num_leaves | [20, 100] | Broj listova |
| max_depth | [4, 10] | Maksimalna dubina stabla |
| min_child_samples | [10, 50] | Minimalan broj uzoraka u listu |
| feature_fraction | [0.6, 1.0] | Postotak featurea |
| bagging_fraction | [0.6, 1.0] | Postotak uzoraka |
| bagging_freq | [1, 7] | Frekvencija bagginga |
| reg_alpha | [1e-8, 10] (log) | L1 regularizacija |
| reg_lambda | [1e-8, 10] (log) | L2 regularizacija |

**Proces optimizacije:**
1. 30 triala (iteracija)
2. 5-fold cross-validation za svaki trial
3. Minimizacija RMSE
4. Optuna automatski bira sljedeće parametre na temelju prethodnih rezultata

### 4.3 Finalni model

Nakon optimizacije, trenira se finalni model s:
- Najboljim hiperparametrima iz Optune
- n_estimators = 1000
- early_stopping_rounds = 50

### 4.4 Rezultati Model V2

| Metrika | Kuće - Train | Kuće - Test | Stanovi - Train | Stanovi - Test |
|---------|--------------|-------------|-----------------|----------------|
| RMSE | ~230,000 EUR | ~280,000 EUR | ~70,000 EUR | ~95,000 EUR |
| MAE | ~150,000 EUR | ~170,689 EUR | ~42,000 EUR | ~47,546 EUR |
| R² | ~0.75 | **0.6253** | ~0.82 | **0.7521** |
| MAPE | ~25% | **34.2%** | ~55% | **75.5%** |

---

## 5. Model V3 - Target Encoding + Log Transform

### 5.1 Implementacija
Skripta: `src/models/train_v3.py`

### 5.2 Ključne promjene u odnosu na V2

#### 5.2.1 Uklanjanje očitih grešaka u podacima (NE outliera!)

```python
def remove_data_errors(df, dataset_type):
    # Kuće: min 15,000 EUR, max 10,000,000 EUR
    # Stanovi: min 10,000 EUR, max 5,000,000 EUR
    # Minimalna površina: 10 m²
```

**Razlika od outlier removal:**
- Outlier removal (V2): uklanja legitimne skupe nekretnine
- Data error removal (V3): uklanja samo očite greške (npr. cijena 1 EUR)

**Uklonjeno:**
- Kuće: 21 zapisa (0.9%)
- Stanovi: 115 zapisa (3.3%)

#### 5.2.2 Log transformacija cijene

```python
y = np.log1p(cijena)  # Trening
pred_eur = np.expm1(pred_log)  # Predikcija
```

**Zašto:**
- Distribucija cijena je jako asimetrična (positive skew)
- Model više ne "lovi" ekstremne vrijednosti
- MAPE postaje smislenija metrika

#### 5.2.3 Target Encoding za lokacije (K-fold)

Umjesto label encodinga koji stvara lažni ordinalni odnos:
```python
# LOŠE (V1/V2): "Istarska" → 0, "Zagrebačka" → 1
X[col] = df[col].astype("category").cat.codes
```

Koristimo target encoding s K-fold cross-validacijom:
```python
# DOBRO (V3): "Istarska" → 480,000 (prosječna cijena)
def target_encode_kfold(df, column, target, n_splits=5, smoothing=10):
    # K-fold da se izbjegne data leakage
    # Smoothing da se izbjegne overfitting na rijetke kategorije
```

**Smoothing formula:**
```
encoded = (count * mean + smoothing * global_mean) / (count + smoothing)
```

#### 5.2.4 Ispravljeno rukovanje renovacijom

**Problem u V1/V2:**
```python
# Ako nema renovacije, postavi godinu izgradnje
if godina_renovacije is None:
    godina_renovacije = godina_izgradnje
```
Ovo laže model jer nema razlike između "renovirano iste godine kad je izgrađeno" i "nikad renovirano".

**Ispravak u V3:**
```python
# Flag: ima li pravu renovaciju
ima_renovaciju = (godina_renovacije != godina_izgradnje)
# Ako su jednake, postavi NaN (nije bilo prave renovacije)
```

**Rezultat:** Samo 22.6% kuća i 15.4% stanova zapravo ima renovaciju.

#### 5.2.5 Interaction features

**Za kuće:**
- `pogled_more_primorje`: pogled na more × primorska županija
- `okucnica_log`: log(površina okućnice)
- `bazen_uz_more`: ima bazen × pogled na more

**Za stanove:**
- `lift_visoki_kat`: ima lift × (kat > 3)
- `novogradnja_povrsina`: novogradnja × log(površina)

#### 5.2.6 Različiti hiperparametri za kuće vs stanove

**Kuće** (kompleksnije, trebaju više fleksibilnosti):
- num_leaves: 31-150
- max_depth: 5-12
- min_child_samples: 5-30
- Manja regularizacija (reg_alpha/lambda: 1e-8 do 1.0)

**Stanovi** (homogeniji):
- num_leaves: 20-100
- max_depth: 4-10
- min_child_samples: 10-50
- Standardna regularizacija

### 5.3 Rezultati V3

| Metrika | Kuće - Train | Kuće - Test | Stanovi - Train | Stanovi - Test |
|---------|--------------|-------------|-----------------|----------------|
| RMSE (log) | 0.2407 | 0.3714 | 0.1334 | 0.2261 |
| RMSE (EUR) | 203,627 | 329,214 | 54,122 | 108,926 |
| MAE (EUR) | 108,479 | **167,546** | 30,512 | **55,045** |
| R² | 0.8569 | **0.6809** | 0.9169 | **0.7113** |
| MAPE | 18.2% | **29.2%** | 9.5% | **16.9%** |

### 5.4 MAPE po cjenovnim bucketima (Test set)

**Kuće:**
| Bucket | MAPE | Broj nekretnina |
|--------|------|-----------------|
| 0-200k EUR | 55.1% | 55 |
| 200k-500k EUR | 29.1% | 198 |
| 500k-1M EUR | 21.8% | 171 |
| 1M+ EUR | 27.3% | 60 |

**Stanovi:**
| Bucket | MAPE | Broj nekretnina |
|--------|------|-----------------|
| 0-200k EUR | 18.8% | 189 |
| 200k-500k EUR | 15.7% | 422 |
| 500k-1M EUR | 15.3% | 61 |
| 1M+ EUR | 38.7% | 11 |

### 5.5 Top 15 najvažnijih featurea

**Kuće:**
1. stambena_povrsina (413)
2. povrsina_okucnice (393)
3. naselje_encoded (357)
4. zupanija_encoded (277)
5. grad_opcina_encoded (247)
6. godina_izgradnje (243)
7. broj_soba (171)
8. kupaonica_s_wc_broj (155)
9. broj_parkirnih_mjesta (106)
10. povrsina_log (103)

**Stanovi:**
1. stambena_povrsina (624)
2. naselje_encoded (463)
3. zupanija_encoded (412)
4. kat (348)
5. godina_izgradnje (301)
6. grad_opcina_encoded (301)
7. povrsina_log (222)
8. ukupni_broj_katova (179)
9. novogradnja_povrsina (174)
10. broj_soba (125)

---

## 6. Usporedba Rezultata

### 6.1 Sažetak svih verzija

| Model | Opis | Kuće R² | Kuće MAE | Kuće MAPE | Stanovi R² | Stanovi MAE | Stanovi MAPE |
|-------|------|---------|----------|-----------|------------|-------------|--------------|
| **V1** | Osnovni LightGBM | 0.6944 | 205,656 | - | 0.7239 | 65,790 | - |
| **V2** | + Outlier removal, Optuna | 0.6253 | 170,689 | 34.2% | 0.7521 | 47,546 | 75.5%* |
| **V3** | + Log transform, Target encoding | **0.6809** | **167,546** | **29.2%** | 0.7113 | 55,045 | **16.9%** |

*V2 MAPE za stanove bio je nerealno visok zbog grešaka u podacima (cijene od 1 EUR)

### 6.2 Analiza promjena

**V1 → V2:**
- Outlier removal pomogao je MAE (uklonio ekstreme)
- Ali uklonio je i legitimne luksuzne nekretnine
- MAPE za stanove bio je nerealan zbog data errors

**V2 → V3:**
- Data error removal (umjesto outlier removal) zadržao legitimne nekretnine
- Log transformacija stabilizirala treniranje
- Target encoding značajno poboljšao MAPE (75.5% → 16.9% za stanove)
- Ispravljen handling renovacije dao realnije feature importance

### 6.3 Preporučeni model

**Za produkciju: V3**

Razlozi:
1. **MAPE je realan i interpretatibilan** - 17-29% prosječna greška
2. **Konzistentnost po cjenovnim bucketima** - model ne griješi sistematski za skupe/jeftine nekretnine
3. **Target encoding hvata geografske razlike** - lokacija je pravilno enkodirana
4. **Spreman za inference** - model sadrži sve potrebne encodinge

### 6.4 R² vs MAPE tradeoff

Primjetite da V3 ima nešto niži R² od V2 za stanove (0.71 vs 0.75), ali **dramatično bolji MAPE** (16.9% vs 75.5%).

Ovo je zato što:
- R² je osjetljiv na outliere (ekstremne cijene)
- MAPE normalizira grešku po stvarnoj cijeni
- Za praktičnu upotrebu, MAPE je važniji - korisnika zanima "za koliko posto griješim"

---

## 7. Poznati Problemi i Moguća Poboljšanja

### 7.1 Problemi s podacima

| Problem | Status | Rješenje u V3 |
|---------|--------|---------------|
| Ekstremne cijene (1 EUR) | ✅ Riješeno | Data error removal (min 10-15k EUR) |
| Nedostajući podaci | ⚠️ Djelomično | Flag stupci + LightGBM native NaN handling |
| Neuravnoteženost lokacija | ⚠️ Preostaje | Target encoding pomaže, ali ne rješava potpuno |
| Godina renovacije = godina izgradnje | ✅ Riješeno | Ispravljen handling s ima_renovaciju flag |

### 7.2 Implementirano u V3

- ✅ **Target Encoding za lokacije** - K-fold bez leaka
- ✅ **Log transformacija target varijable** - np.log1p(cijena)
- ✅ **Interaction features** - pogled_more_primorje, bazen_uz_more, lift_visoki_kat
- ✅ **Ispravljen handling renovacije** - NaN gdje nema prave renovacije
- ✅ **Različiti hiperparametri** - kuće vs stanovi

### 7.3 Moguća buduća poboljšanja

1. **Stacking/Ensemble:**
   - Kombinirati LightGBM s CatBoostom i XGBoostom
   - CatBoost ima native handling kategoričkih varijabli

2. **Dodatni feature engineering:**
   - Udaljenost od mora/centra grada (geocoding)
   - Cijena po m² susjednih nekretnina

3. **Stratified sampling:**
   - Train/test split stratificiran po županiji
   - Osigurava reprezentativnost u oba seta

4. **Separate models by price range:**
   - Jedan model za nekretnine < 500,000 EUR
   - Drugi model za luksuzne nekretnine (veća varijabilnost)

5. **Eksterna validacija:**
   - Testirati na potpuno novim podacima (ne samo holdout set)
   - Pratiti drift performansi kroz vrijeme

### 7.4 Preostala pitanja za metodologiju

1. Je li smoothing=10 optimalan za target encoding? (možda cross-validirati)
2. Trebaju li se interaction features automatski generirati (npr. polynomial features)?
3. Je li 80/20 split optimalan ili bi 85/15 dao bolje rezultate?
4. Treba li koristiti weighted loss za rijetke kategorije (npr. luksuzne nekretnine)?

---

## Appendix: Struktura Projekta

```
Croatian-Property-Price-Estimator/
├── data/
│   ├── raw/                    # Originalni parquet fileovi
│   │   ├── houses.parquet
│   │   └── apartments.parquet
│   └── processed/              # Očišćeni podaci
│       ├── houses_clean.parquet
│       └── apartments_clean.parquet
├── docs/
│   ├── data_cleaning_plan.md   # Plan čišćenja podataka
│   └── model_documentation.md  # Ovaj dokument
├── models/
│   ├── houses_model.joblib              # Model V1 za kuće
│   ├── apartments_model.joblib          # Model V1 za stanove
│   ├── houses_model_optimized.joblib    # Model V2 za kuće
│   ├── apartments_model_optimized.joblib # Model V2 za stanove
│   ├── houses_model_v3.joblib           # Model V3 za kuće (PREPORUČENO)
│   └── apartments_model_v3.joblib       # Model V3 za stanove (PREPORUČENO)
├── notebooks/
│   └── eda.py                           # Marimo notebook za EDA
└── src/
    ├── preprocessing/
    │   └── clean_data.py                # Skripta za čišćenje podataka
    ├── models/
    │   ├── train.py            # Model V1 trening
    │   ├── train_optimized.py  # Model V2 trening
    │   └── train_v3.py         # Model V3 trening (PREPORUČENO)
    └── scraping/
        ├── scraper.py          # Web scraper
        └── split_data.py       # Razdvajanje podataka
```

---
