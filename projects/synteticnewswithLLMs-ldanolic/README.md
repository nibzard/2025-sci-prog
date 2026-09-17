# Usporedba modela za detekciju manipuliranih vijesti prije i nakon LLM augmentacije

## Opis projekta

Cilj projekta je ispitati može li LLM-generirana augmentacija podataka poboljšati prepoznavanje manipuliranih vijesti.

Uspoređuju se dva modela:

- **Baseline model** – treniran samo na originalnim podacima
- **Augmented model** – treniran na originalnim i sintetički generiranim podacima

Oba modela koriste isti klasifikacijski pristup:

- word-level TF-IDF
- character-level TF-IDF
- LinearSVC

Promatraju se tri vrste manipulacije:

- `fact_change` – promjena jedne činjenice
- `clickbait` – promjena naslova tako da više privlači pažnju
- `tone_shift` – promjena tona teksta

Za generiranje sintetičkih trening podataka koristi se `gemini-3.5-flash-lite`, dok se challenge test generira pomoću `gemini-3.5-flash`.

## Podaci

Koristi se ISOT Fake News Dataset koji sadrži:

- `True.csv` – stvarne vijesti
- `Fake.csv` – lažne vijesti

Nakon čišćenja podaci se dijele na:

- 80 % train
- 20 % test

Sintetički trening podaci generiraju se samo iz stvarnih vijesti iz train skupa, dok se challenge test radi iz stvarnih vijesti iz test skupa.

Datoteke `Fake.csv` i `True.csv` nisu uključene u repozitorij. Potrebno ih je preuzeti s Kagglea i smjestiti u `data/raw/`.

## Struktura projekta

```text
synteticnewswithLLMs-ldanolic/
│
├── data/
│   ├── raw/
│   │   └── README.md
│   │
│   └── processed/
│       ├── test.csv
│       ├── synthetic_train.csv
│       └── challenge_test.csv
│
├── models/
│   ├── baseline_model.joblib
│   └── augmented_model.joblib
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_synthetic_generation.ipynb
│   └── 03_training_evaluation.ipynb
│
├── results/
│   ├── challenge_metrics.csv
│   └── figures/
│       └── challenge_comparison.png
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── generation.py
│   ├── modeling.py
│   └── evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Datoteke `train.csv` i `augmented_train.csv` postoje lokalno tijekom rada projekta, ali nisu uključene u GitHub repozitorij zbog veličine (> 100 mb).

## Pokretanje projekta

Projekt je potrebno otvoriti iz glavnog direktorija `synteticnewswithLLMs-ldanolic`.

### 1. Instalacija biblioteka

U terminalu pokrenuti:

```bash
pip install -r requirements.txt
```

### 2. Ulazni podaci

Preuzeti ISOT Fake News Dataset i datoteke `Fake.csv` i `True.csv` smjestiti u:

```text
data/raw/
```

### 3. Gemini API ključ

Ako se sintetički podaci ponovno generiraju, u glavnom direktoriju projekta potrebno je napraviti `.env` datoteku:

```text
GEMINI_API_KEY=vas_api_kljuc
```

Datoteka `.env` nije uključena u Git repozitorij.

### 4. Pokretanje notebookova

Notebookovi se pokreću sljedećim redoslijedom:

```text
01_data_preparation.ipynb
02_synthetic_generation.ipynb
03_training_evaluation.ipynb
```

#### `01_data_preparation.ipynb`

Učitava `Fake.csv` i `True.csv`, čisti podatke, dodjeljuje labele te radi podjelu na train i test skup.

Nastaju:

```text
data/processed/train.csv
data/processed/test.csv
```

#### `02_synthetic_generation.ipynb`

Iz stvarnih vijesti train skupa generiraju se sintetičke manipulacije pomoću LLM-a.

Nastaju:

```text
data/processed/synthetic_train.csv
data/processed/augmented_train.csv
data/processed/challenge_test.csv
```

Challenge test generira se iz stvarnih vijesti test skupa pomoću drugog LLM modela.

#### `03_training_evaluation.ipynb`

Baseline model trenira se na `train.csv`, dok se Augmented model trenira na `augmented_train.csv`.

Modeli se spremaju u:

```text
models/baseline_model.joblib
models/augmented_model.joblib
```

Nakon toga oba modela evaluiraju se na tri challenge testa:

- `fact_change`
- `clickbait`
- `tone_shift`

Računaju se Accuracy, Precision, Recall i F1.

Rezultati se spremaju u:

```text
results/challenge_metrics.csv
results/figures/challenge_comparison.png
```

## Pokretanje bez ponovnog LLM generiranja i treniranja

U repozitoriju su već spremljeni `synthetic_train.csv`, `challenge_test.csv`, istrenirani modeli i završni rezultati. Zbog toga za pregled rezultata i ponovnu evaluaciju nije potrebno ponovno pozivati Gemini niti ponovno trenirati modele.

U `02_synthetic_generation.ipynb` mogu se učitati već generirani podaci:

```python
synthetic_train = pd.read_csv(
    "data/processed/synthetic_train.csv"
)

challenge_test = pd.read_csv(
    "data/processed/challenge_test.csv"
)
```

U tom slučaju ne pokreću se ćelije koje pozivaju:

```python
generate_synthetic_dataset(...)
```

U `03_training_evaluation.ipynb` spremljeni modeli mogu se učitati pomoću:

```python
baseline_model = joblib.load(
    "models/baseline_model.joblib"
)

augmented_model = joblib.load(
    "models/augmented_model.joblib"
)
```

Tada nije potrebno ponovno pokretati:

```python
baseline_model.fit(...)
augmented_model.fit(...)
```

Nakon učitavanja spremljenih modela može se ponovno pokrenuti evaluacija, izračunati metrike i generirati graf rezultata.

## Rezultati

| Vrsta manipulacije | Baseline F1 | Augmented F1 | Poboljšanje |
|---|---:|---:|---:|
| Clickbait | 0.99 % | 66.45 % | +65.46 p.b. |
| Fact change | 0.99 % | 1.97 % | +0.98 p.b. |
| Tone shift | 28.21 % | 95.31 % | +67.11 p.b. |

Najveće poboljšanje ostvareno je kod `clickbait` i `tone_shift` manipulacija. Kod `fact_change` manipulacije rezultat je ostao nizak jer se mijenja samo manji dio sadržaja, dok ostatak teksta ostaje gotovo jednak originalnoj vijesti.

## Napomena o velikim datotekama

Datoteke `train.csv` i `augmented_train.csv` nisu uključene u repozitorij jer su veće od 100 MB. Mogu se ponovno izraditi pokretanjem odgovarajućih notebookova.