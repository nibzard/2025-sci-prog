# Praktični zaključci iz faza_9 analize

## 🎯 SAŽETAK - ŠTO MOŽEŠ ODMAH PRIMIJENITI

### ✅ Najbolje opcije (testirano i preporučeno):
1. **Post-processing calibration** ✅ **IMPLEMENTIRANO I TESTIRANO** - Poboljšalo performanse:
   - R²: 0.9057 → 0.9244 (+2.06%)
   - RMSE: 1177.58 → 1054.70 minuta (-10.43%)
   - MAE: 774.22 → 729.89 minuta (-5.73%)
   - Long segment MAE: 1774.70 → 1603.70 minuta (-9.6%)
2. **Heteroskedastični prediction intervals** ✅ **IMPLEMENTIRANO** - Koristi quantile regression za dinamičke intervale
3. **Separate model za Long segment** ⚠️ - Testirano, ali nije poboljšalo kada se kombinuje sa calibration-om

### ⚠️ Što NE radi (testirano):
- **Sample weighting sa weight=3.0** - Pogoršalo performanse (R² pad sa 0.9057 na 0.8420)
- Probaj manje agresivne weights (1.5-2.0) ako želiš eksperimentisati

### 📊 Ključni nalazi:
- Long segment ima najveće greške (MAE = 1774.70 minuta)
- Model podcjenjuje duga trajanja za ~1167 minuta u prosjeku
- Prediction intervals su konstantne širine (trebaju biti dinamički)
- 2 od 27 test podataka van intervala - oba su Long segment

---

## 1. PROBLEM SA LONG SEGMENTOM (Kritično)

**Nalaz:**
- Model značajno **podcjenjuje** duga trajanja (Long segment: 75%+)
- Mean error: **+1167 minuta** (model predviđa prenisko)
- MAE: **1774.70 minuta** - najveća greška od svih segmenata
- 2 od 27 test podataka su van 95% intervala - **oba su Long segment** (6768 i 20521 minuta)

**Što možeš primijeniti:**
- ⚠️ **Sample weighting**: Testirano sa weight=3.0 - **POGORŠALO performanse** (R² pad sa 0.9057 na 0.8420)
  - Probaj manje agresivne weights (1.5-2.0) ili adaptive weighting
- ✅ **Oversampling**: Dupliraj ili povećaj Long segment u training setu (SMOTE ili ručno)
- ✅ **Feature engineering**: Dodaj specifične feature-e za Long segment (npr. interakcije koje su relevantne samo za duga trajanja)
- ✅ **Separate model**: Treniraj poseban model samo za Long segment (>75% kvantil) - **najbolja opcija**
- ✅ **Post-processing calibration**: Dodaj bias correction za Long segment nakon predikcije

---

## 2. KONSTANTNA ŠIRINA PREDICTION INTERVALA (Problem)

**Nalaz:**
- Sve prediction intervals imaju **istu širinu** (4585.53 minuta)
- To nije realno - veće predikcije trebaju šire intervale

**Što možeš primijeniti:**
- ✅ **Heteroskedastični modeli**: Koristi quantile regression (XGBoost quantile loss)
- ✅ **Dynamic intervals**: Izračunaj interval width kao funkciju predikcije: `width = a * predicted + b`
- ✅ **Segment-specific intervals**: Različite širine intervala za različite segmente

---

## 3. BIAS PO SEGMENTIMA

**Nalaz:**
- **Very Short** (0-25%): Mean error **-162 min** → Model precjenjuje
- **Short** (25-50%): Mean error **-133 min** → Model precjenjuje  
- **Medium** (50-75%): Mean error **+165 min** → Model blago podcjenjuje
- **Long** (75%+): Mean error **+1167 min** → Model značajno podcjenjuje

**Što možeš primijeniti:**
- ✅ **Post-processing calibration**: Dodaj segment-specific bias correction
- ✅ **Cost-sensitive learning**: Veća kazna za greške u Long segmentu
- ✅ **Ensemble approach**: Kombiniraj modele trenirane na različitim segmentima

---

## 4. COVERAGE ANALIZA

**Nalaz:**
- 95% interval pokriva **92.59%** podataka (blizu očekivanog)
- Ali svi intervali (68%, 80%, 90%, 95%, 99%) imaju **istu coverage** (92.59%)
- To znači da su intervali preširoki za niže confidence nivoe

**Što možeš primijeniti:**
- ✅ **Recalibrate intervals**: Koristi quantile regression za tačnije intervale
- ✅ **Bootstrap intervals**: Koristi bootstrap metodu za confidence intervale
- ✅ **Conformal prediction**: Implementiraj conformal prediction za garancije coverage-a

---

## 5. RELATIVNA GREŠKA

**Nalaz:**
- Very Short: Relativna greška je najveća (jer su apsolutne vrijednosti male)
- Long: Apsolutna greška je najveća, ali relativna može biti manja

**Što možeš primijeniti:**
- ✅ **Dual metric**: Optimizuj i za MAE i za MAPE (Mean Absolute Percentage Error)
- ✅ **Weighted loss**: Koristi weighted loss funkciju koja uzima u obzir relativnu grešku

---

## 6. PRIORITETNE AKCIJE (Redoslijed)

### Visok prioritet:
1. **Post-processing calibration** - Dodaj bias correction za Long segment (najbrže, najsigurnije)
2. **Heteroskedastični prediction intervals** - poboljšava pouzdanost predikcija
3. **Separate model za Long segment** - Najbolje rješenje, ali kompleksnije

### Srednji prioritet:
4. **Feature engineering za Long segment** - zahtijeva domen ekspertizu
5. **Oversampling Long segmenta** - Može pomoći, ali treba paziti na overfitting

### Niski prioritet (ako imaš vremena):
6. **Conformal prediction** - napredna metoda za garancije coverage-a
7. **Adaptive sample weighting** - Eksperimentiši sa različitim weight strategijama

---

## 7. KONKRETNI KOD PREPORUKE

### Post-Processing Calibration (PREPORUČENO - TESTIRANO I RADI):
```python
# Dodaj bias correction za Long segment nakon predikcije
# Optimalni factor se izračunava na training setu
train_long_mask = y_train > long_threshold
y_train_pred_long = model.predict(X_train[train_long_mask])
calibration_factor = y_train[train_long_mask].mean() / np.expm1(y_train_pred_long).mean()

# Primijeni na test setu
test_long_mask = y_test > long_threshold
y_pred_calibrated = y_pred.copy()
y_pred_calibrated[test_long_mask] = y_pred[test_long_mask] * calibration_factor
# U našem slučaju: calibration_factor ≈ 1.071 (7.1% povećanje)
```

### Sample Weighting (OPREZ - testirano, nije pomoglo):
```python
# Probaj manje agresivne weights
weights = np.ones(len(y_train))
long_threshold = y_train.quantile(0.75)
weights[y_train > long_threshold] = 1.5  # Umjesto 3.0, probaj 1.5-2.0
model.fit(X_train, y_train_log, sample_weight=weights)
```

### Quantile Regression (IMPLEMENTIRANO):
```python
# Treniraj model za različite kvantile
model_q05 = xgb.XGBRegressor(
    objective='reg:quantileerror', 
    quantile_alpha=0.05,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    random_state=42
)
model_q95 = xgb.XGBRegressor(
    objective='reg:quantileerror', 
    quantile_alpha=0.95,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    random_state=42
)

# Predikcije
y_pred_lower = np.expm1(model_q05.predict(X_test))
y_pred_upper = np.expm1(model_q95.predict(X_test))
# Interval width je dinamički (varira sa predikcijom)
```

### Dynamic Intervals:
```python
# Izračunaj interval width kao funkciju predikcije
interval_width = 0.5 * y_pred + 500  # Primjer
lower = y_pred - interval_width / 2
upper = y_pred + interval_width / 2
```

---

## 8. REZULTATI FINALNOG MODELA (model_final_v2.py)

### Implementirane strategije:
- ✅ **Post-processing calibration** - Calibration factor: 1.0710 (7.1% povećanje za Long segment)
- ✅ **Quantile regression** - Dinamički prediction intervals (5th-95th percentile)

### Performanse:

| Metrika | Baseline | Final Model | Poboljšanje |
|---------|----------|-------------|-------------|
| **R²** | 0.9057 | **0.9244** | **+2.06%** ✅ |
| **RMSE** | 1177.58 min | **1054.70 min** | **-10.43%** ✅ |
| **MAE** | 774.22 min | **729.89 min** | **-5.73%** ✅ |

### Long Segment Performanse:
- **Baseline MAE**: 1774.70 minuta
- **Final MAE**: 1603.70 minuta
- **Poboljšanje**: -9.6% ✅

### Prediction Intervals:
- **Mean width**: 3493.60 minuta (dinamički, varira sa predikcijom)
- **Min width**: 2015.07 minuta
- **Max width**: 26377.71 minuta
- **Coverage**: 74.07% (očekivano ~90% za 5th-95th percentile)

### Zaključak:
✅ **Post-processing calibration je uspješno implementirana i poboljšala performanse**
✅ **Quantile regression omogućava dinamičke prediction intervals**
⚠️ **Coverage je niži od očekivanog - možda treba fino-tunirati quantile alpha vrijednosti**

### Fajlovi:
- `model_final_v2.py` - Finalni model sa implementiranim preporukama
- `model_final_v2_performance.png` - Vizualizacija performansi
- `model_final_v2_predictions.csv` - Detaljne predikcije sa intervalima
- `model_final_v2_metrics.csv` - Metričke komparacije
- `model_final_v2_segment_stats.csv` - Statistike po segmentima
