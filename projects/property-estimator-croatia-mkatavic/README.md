# Croatian Property Price Estimator

**Live:** [croatian-property-price-estimator.maksimilijankatavic.com](https://croatian-property-price-estimator.maksimilijankatavic.com/) | **API:** [Hugging Face Space](https://huggingface.co/spaces/maksimilijankatavic/Croatian-Property-Price-Estimator)

Sustav za procjenu cijena nekretnina u Hrvatskoj temeljen na strojnom ucenju. Koristi LightGBM modele trenirane na stvarnim podacima s hrvatskog trzista nekretnina, s interaktivnim web suceljem za korisnike i FastAPI backendom za predikcije.

---

## Sadrzaj

- [Pregled](#pregled)
- [Arhitektura](#arhitektura)
- [Struktura projekta](#struktura-projekta)
- [Pipeline podataka](#pipeline-podataka)
  - [Web scraping](#1-web-scraping)
  - [Ciscenje podataka](#2-ciscenje-podataka)
  - [Eksplorativna analiza](#3-eksplorativna-analiza-eda)
- [Modeli](#modeli)
  - [V1 - Baseline](#v1---baseline)
  - [V2 - Optimizirani](#v2---optimizirani)
  - [V3 - Produkcijski](#v3---produkcijski-preporuceni)
- [Web aplikacija](#web-aplikacija)
- [API Backend](#api-backend-hugging-face-spaces)
- [Pokretanje projekta](#pokretanje-projekta)
- [Tehnologije](#tehnologije)

---

## Pregled

Na hrvatskom trzistu ne postoji javno dostupan alat za objektivnu procjenu vrijednosti nekretnina. Ovaj projekt popunjava tu prazninu:

- **Prikupljanje podataka**: Automatizirani web scraper za prikupljanje oglasa nekretnina
- **ML pipeline**: Tri iteracije modela (V1-V3) s progresivnim poboljsanjima
- **Web sucelje**: Interaktivna aplikacija s kartom, grafovima i formom za predikciju
- **API**: FastAPI backend na Hugging Face Spaces koji sluzi V3 modele

### Dataset

| Tip nekretnine | Broj zapisa | Broj znacajki | Prosjecna cijena | Medijan cijene |
|---|---|---|---|---|
| Stanovi | ~3,529 | 92 | 298,944 EUR | 264,625 EUR |
| Kuce | ~2,439 | 94 | 610,921 EUR | 470,000 EUR |

---

## Arhitektura

```
Korisnik
   |
   v
Next.js Web App (localhost:3000)
   |
   |--- Staticni podaci (eda.json, form.json) --> Grafovi, karta, lokacijska hijerarhija
   |
   |--- POST /predict --> Hugging Face Spaces (FastAPI)
                              |
                              v
                         LightGBM V3 Model
                              |
                              v
                         Procijenjena cijena + raspon
```

---

## Struktura projekta

```
Croatian-Property-Price-Estimator/
|
|-- src/
|   |-- scraping/
|   |   |-- scraper.py          # Web scraper (Steel API + BeautifulSoup)
|   |   |-- split_data.py       # Razdvajanje na stanove i kuce
|   |
|   |-- preprocessing/
|   |   |-- clean_data.py       # Ciscenje podataka i feature engineering
|   |
|   |-- models/
|       |-- train.py            # V1 model - baseline LightGBM
|       |-- train_optimized.py  # V2 model - Optuna optimizacija
|       |-- train_v3.py         # V3 model - produkcijski
|
|-- models/                     # Trenirani modeli (.joblib)
|   |-- apartments_model.joblib
|   |-- apartments_model_optimized.joblib
|   |-- apartments_model_v3.joblib
|   |-- houses_model.joblib
|   |-- houses_model_optimized.joblib
|   |-- houses_model_v3.joblib
|
|-- notebooks/
|   |-- eda.py                  # Marimo interaktivni EDA notebook
|
|-- hf-space/                   # Hugging Face Spaces deployment
|   |-- app.py                  # FastAPI backend
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- models/                 # Kopije V3 modela za deployment
|
|-- web/                        # Next.js web aplikacija
|   |-- src/
|   |   |-- app/
|   |   |   |-- page.tsx        # Glavna stranica (forma, grafovi, predikcija)
|   |   |   |-- layout.tsx      # App layout
|   |   |   |-- globals.css     # Stilovi
|   |   |
|   |   |-- components/
|   |   |   |-- Map.tsx         # Leaflet karta s gradovima
|   |   |
|   |   |-- lib/
|   |       |-- api.ts          # API servis za HF Spaces komunikaciju
|   |       |-- types.ts        # TypeScript tipovi
|   |       |-- utils.ts        # Pomocne funkcije
|   |
|   |-- public/data/
|   |   |-- eda.json            # Preracunate EDA statistike
|   |   |-- form.json           # Lokacijska hijerarhija i encodinzi
|   |
|   |-- generate_data.py        # Generira eda.json i form.json iz dataseta
|
|-- docs/
|   |-- model_documentation.md  # Detaljna dokumentacija modela
|   |-- data_cleaning_plan.md   # Plan ciscenja podataka
|
|-- data/                       # Podaci (nije u git-u)
    |-- raw/                    # Neobradjeni scrapirani podaci
    |-- processed/              # Ocisceni podaci
```

---

## Pipeline podataka

### 1. Web scraping

**Datoteka:** `src/scraping/scraper.py`

Scraper koristi Steel API za browser-based renderiranje i prikuplja strukturirane podatke s oglasa nekretnina:

- **Ekstrakcija podataka**: Cijena, lokacija, povrsina, broj soba, godina izgradnje, energetski razred, grijanje, dozvole, opis
- **Obrada slika**: Do 5 slika po oglasu, konvertirane u WebP format (512x512px, 85% kvaliteta)
- **Deduplikacija**: MD5 hash za detekciju vec prikupljenih oglasa
- **Rate limiting**: 2-4 sekunde pauze izmedju zahtjeva
- **Inkrementalno spremanje**: Svaka 10 oglasa

**Output**: `properties_raw.parquet`

Skripta `split_data.py` razdvaja podatke na `apartments.parquet` i `houses.parquet`.

### 2. Ciscenje podataka

**Datoteka:** `src/preprocessing/clean_data.py`

Transformacije primijenjene na sirove podatke:

| Kategorija | Transformacija | Primjer |
|---|---|---|
| Cijena | Parsiranje stringa u float | "330.000 EUR" -> 330000.0 |
| Povrsina | Parsiranje s decimalima | "139,27 m2" -> 139.27 |
| Godina | Uklanjanje interpunkcije | "2025." -> 2025 |
| Etaze | Kategorija u broj | Prizemnica -> 1, Katnica -> 2 |
| Sobe | Kategorija u broj | Garsonijera -> 0.5, 2-sobni -> 2 |
| Energetski razred | Ordinalno kodiranje | A+ -> 5, A -> 4, ..., G -> -2 |
| Binarne znacajke | Da/None u 1/0 | Klima: Da -> 1, None -> 0 |
| Visevrijednosne | One-hot kodiranje | Balkon, terasa -> zasebni stupci |
| Lokacija | Razdvajanje | "Zupanija, Grad, Naselje" -> 3 stupca |

**Strategija za nedostajuce vrijednosti:** NaN se zadrzava — LightGBM ih obradjuje nativno. Dodani su flag stupci (`ima_godinu_izgradnje`, `ima_energetski_razred`, `ima_sustav_grijanja`) za indikaciju prisutnosti podataka.

**Output:**
- `apartments_clean.parquet` — 86 stupaca
- `houses_clean.parquet` — 87 stupaca

### 3. Eksplorativna analiza (EDA)

**Datoteka:** `notebooks/eda.py` (Marimo interaktivni notebook)

Analiza ukljucuje:
- Distribucije cijena (histogrami, box plotovi)
- Korelacija povrsina-cijena (scatter plotovi)
- Geografska analiza po zupanijama i gradovima
- Postotak nedostajucih vrijednosti po stupcima
- Korelacijska matrica sa cijenom
- Distribucija energetskih razreda, broja soba, godina izgradnje
- Analiza cijene po m2

---

## Modeli

Sva tri modela koriste **LightGBM** (Light Gradient Boosting Machine) s log-transformiranom ciljnom varijablom (`log1p(cijena)`) u V3.

### V1 - Baseline

**Datoteka:** `src/models/train.py`

- Label encoding za kategoricke varijable (zupanija, grad, naselje)
- Fiksni hiperparametri: `num_leaves=31`, `learning_rate=0.05`, `n_estimators=1000`
- Early stopping na validacijskom setu (50 rundi)

| Metrika | Stanovi | Kuce |
|---|---|---|
| RMSE | 106,065 EUR | 316,063 EUR |
| MAE | 65,790 EUR | 205,656 EUR |
| R2 | 0.724 | 0.694 |

### V2 - Optimizirani

**Datoteka:** `src/models/train_optimized.py`

Poboljsanja nad V1:
- **Uklanjanje outliera**: Ispod 1. i iznad 99. percentila
- **Feature engineering**: `starost`, `godine_od_renovacije`, `povrsina_log`, `ukupno_kupaonica`
- **Optuna optimizacija**: 30 rundi bayesovske optimizacije s 5-fold CV

| Metrika | Stanovi | Kuce |
|---|---|---|
| RMSE | ~95,000 EUR | ~280,000 EUR |
| MAE | 47,546 EUR | 170,689 EUR |
| R2 | 0.752 | 0.625 |
| MAPE | 75.5% | 34.2% |

### V3 - Produkcijski (preporuceni)

**Datoteka:** `src/models/train_v3.py`

Kljucna poboljsanja:

**Kvaliteta podataka:**
- Uklanjanje gresaka u podacima (ne legitimnih outliera)
  - Kuce: min 15,000 EUR, max 10,000,000 EUR, min povrsina 10 m2
  - Stanovi: min 10,000 EUR, max 5,000,000 EUR, min povrsina 10 m2

**Log transformacija ciljne varijable:**
- Treniranje na `log1p(cijena)`, predikcija s `expm1(prediction)`
- Bolje rukovanje asimetricnom distribucijom cijena

**Target encoding lokacija (K-fold):**
- Zamjenjuje label encoding smislenim numerickim vrijednostima
- K-fold cross-validacija sprecava data leakage
- Smoothing formula: `(count * mean + smoothing * global_mean) / (count + smoothing)`

**Interakcijske znacajke:**

| Znacajka | Tip | Opis |
|---|---|---|
| `pogled_more_primorje` | Kuce | Pogled na more * primorska zupanija |
| `okucnica_log` | Kuce | log(povrsina okucnice) |
| `bazen_uz_more` | Kuce | Bazen * pogled na more |
| `lift_visoki_kat` | Stanovi | Lift * (kat > 3) |
| `novogradnja_povrsina` | Stanovi | Novogradnja * log(povrsina) |

**Rezultati (test set):**

| Metrika | Stanovi | Kuce |
|---|---|---|
| RMSE | 108,926 EUR | 329,214 EUR |
| MAE | 55,045 EUR | 167,546 EUR |
| R2 | 0.711 | 0.681 |
| **MAPE** | **16.9%** | **29.2%** |

**MAPE po cjenovnom rangu (stanovi):**

| Rang | MAPE |
|---|---|
| 0 - 200k EUR | 18.8% |
| 200k - 500k EUR | 15.7% |
| 500k - 1M EUR | 15.3% |
| 1M+ EUR | 38.7% |

**Top 5 najvaznijih znacajki:**
- **Kuce:** stambena_povrsina, povrsina_okucnice, naselje_encoded, zupanija_encoded, godina_izgradnje
- **Stanovi:** stambena_povrsina, naselje_encoded, zupanija_encoded, kat, godina_izgradnje

---

## Web aplikacija

**Direktorij:** `web/`

Interaktivna single-page aplikacija s tri glavne sekcije:

### Procjena cijene
- Forma s 40+ atributa nekretnine (lokacija, povrsina, sobe, kat, grijanje, oprema...)
- Lokacijska hijerarhija: zupanija -> grad -> naselje (dinamicki filtrirana)
- Collapsible "Dodatne informacije" sekcija za napredne korisnike
- Usporedba s prosjecnim cijenama u 5 drugih gradova
- Raspon pouzdanosti predikcije baziran na MAPE metrikama modela

### Interaktivna karta
- Leaflet karta Hrvatske s 10 najvecih gradova
- Krugovi proporcionalni broju nekretnina
- Klik na grad filtrira sve statistike i grafove

### Analiza trzista (EDA)
- 8 interaktivnih grafova (Recharts):
  - Prosjecne cijene po zupanijama
  - Distribucija cijena
  - Povrsina vs cijena (scatter)
  - Cijena po m2 distribucija i po zupanijama
  - Distribucija povrsine
  - Distribucija broja soba
  - Distribucija godine izgradnje
- Statisticki pregled: ukupno, prosjek, medijan, povrsina, cijena/m2
- Tablice najskupljih i najpovoljnijih gradova
- Filtriranje po tipu nekretnine (stanovi/kuce) i po gradu

---

## API Backend (Hugging Face Spaces)

**Direktorij:** `hf-space/`

FastAPI aplikacija deployana na Hugging Face Spaces putem Docker SDK-a.

### Endpointi

**`GET /health`**
```json
{
  "status": "ok",
  "models_loaded": ["apartments", "houses"],
  "apartments_features": 92,
  "houses_features": 94
}
```

**`POST /predict`**

Request:
```json
{
  "property_type": "apartments",
  "zupanija": "Grad Zagreb",
  "grad_opcina": "Zagreb",
  "stambena_povrsina": 65,
  "broj_soba": 2,
  "godina_izgradnje": 2010,
  "kat": 3,
  "podatak_lift": 1
}
```

Response:
```json
{
  "predicted_price": 201338,
  "price_per_m2": 3097,
  "price_range": {
    "low": 167312,
    "high": 235364
  },
  "model_version": "v3",
  "features_used": 92
}
```

### Feature engineering u API-ju
- Automatski racuna engineered znacajke: starost, povrsina_log, ukupno_kupaonica, interakcije
- Target encoding lokacija koristeci spremljene mappinge iz treniranog modela
- One-hot kodiranje kategorickih varijabli (parking, grijanje, tip kuce, gradnja)
- Nedostajuce vrijednosti ostaju NaN (LightGBM ih obradjuje nativno)
- Raspon pouzdanosti baziran na MAPE metrikama iz test seta

---

## Pokretanje projekta

### Preduvjeti
- Python 3.11+
- Node.js 18+
- npm

### Web aplikacija (frontend)

```bash
cd web
npm install
cp .env.example .env.local
# Uredite .env.local - postavite NEXT_PUBLIC_HF_SPACES_URL na URL vaseg HF Space-a
npm run dev
```

Aplikacija ce biti dostupna na `http://localhost:3000`.

### API Backend (lokalno testiranje)

```bash
cd hf-space
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### API Backend (Hugging Face Spaces deployment)

1. Kreirajte novi Space na [huggingface.co/spaces](https://huggingface.co/spaces) s Docker SDK-om
2. Pushajte sadrzaj `hf-space/` direktorija u Space repozitorij
3. Postavite `NEXT_PUBLIC_HF_SPACES_URL` u `web/.env.local` na URL vaseg Space-a

### ML pipeline (treniranje modela)

```bash
# 1. Scraping (zahtijeva Steel API kljuc u .env)
python src/scraping/scraper.py

# 2. Razdvajanje podataka
python src/scraping/split_data.py

# 3. Ciscenje
python src/preprocessing/clean_data.py

# 4. Treniranje V3 modela
python src/models/train_v3.py

# 5. Generiranje web podataka
cd web && python generate_data.py
```

---

## Tehnologije

### ML Pipeline
| Tehnologija | Namjena |
|---|---|
| LightGBM | Gradient boosting model za predikciju cijena |
| Optuna | Bayesovska optimizacija hiperparametara |
| scikit-learn | Preprocessing, metrike, train/test split |
| pandas | Manipulacija podataka |
| NumPy | Numericke operacije |
| Marimo | Interaktivni EDA notebook |

### Web Frontend
| Tehnologija | Namjena |
|---|---|
| Next.js 16 | React framework |
| TypeScript | Type-safe kod |
| Tailwind CSS 4 | Stiliziranje |
| Recharts | Interaktivni grafovi |
| Leaflet / react-leaflet | Interaktivna karta |
| Lucide React | Ikone |

### API Backend
| Tehnologija | Namjena |
|---|---|
| FastAPI | Web framework za API |
| Pydantic | Validacija request/response shema |
| uvicorn | ASGI server |
| Docker | Kontejnerizacija za HF Spaces |

### Scraping
| Tehnologija | Namjena |
|---|---|
| Steel API | Browser-based renderiranje JS stranica |
| BeautifulSoup | HTML parsiranje |
| Pillow | Obrada i kompresija slika |

---

## Licenca

Ovaj projekt je razvijen u edukativne svrhe.
