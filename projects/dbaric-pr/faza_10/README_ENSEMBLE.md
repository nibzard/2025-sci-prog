# Ensemble Model V3 - Dva Modela za Različite Tipove PR-ova

## Pregled

Ovaj model implementira **ensemble pristup** koji koristi dva odvojena modela:
1. **Normal Model** - za normalne PR-ove (≤ 2880 minuta)
2. **Long Model** - za velike PR-ove (> 2880 minuta)

## Rezultati

### Performanse Modela

| Model | R² | RMSE (min) | MAE (min) | Primjena |
|-------|----|-----------|-----------|----------|
| Normal Model (Normal PRs) | 0.2998 | 734.3 | 555.0 | Normalni PR-ovi |
| Normal Model (Long PRs) | -0.5372 | 7665.8 | 4924.5 | ❌ Ne koristiti |
| Long Model (Long PRs) | 0.4682 | 4508.9 | 2853.0 | Veliki PR-ovi |
| **Ensemble (All PRs)** | **0.6643** | **2222.0** | **1065.7** | ✅ Svi PR-ovi |

### Ključne Nalaze

✅ **Normal Model** je optimalan za normalne PR-ove, ali **ne radi dobro** na velikim PR-ovima (negativan R²!)

✅ **Long Model** značajno poboljšava performanse na velikim PR-ovima:
- RMSE: ↓ 41.2% (7665.8 → 4508.9 min)
- MAE: ↓ 42.1% (4924.5 → 2853.0 min)
- R²: +1.0054 (od -0.54 do +0.47)

✅ **Ensemble Model** kombinuje prednosti oba modela i daje najbolji ukupni rezultat.

## Kako Koristiti

### Pokretanje Modela

```bash
python3 model_ensemble_v3.py
```

### Generirani Fajlovi

1. **model_ensemble_v3_comparison.png** - Vizualizacije usporedbe modela
2. **model_ensemble_v3_analytics_dashboard.png** - 📊 Analitički dashboard s pregledom svih metrika i performansi
3. **model_ensemble_v3_metrics.csv** - Metričke performanse svih modela
4. **model_ensemble_v3_predictions.csv** - Detaljne predikcije za sve PR-ove
5. **model_ensemble_v3_long_prs_comparison.csv** - Detaljna usporedba za velike PR-ove
6. **model_ensemble_v3_segment_stats.csv** - Statistike po segmentima

### Generiranje Analitičkog Dashboarda

```bash
python3 create_analytics_dashboard.py
```

Dashboard sadrži:
- Ključne metrike performansi (tabela)
- R² score usporedbu
- RMSE i MAE usporedbu
- Poboljšanje Long Modela
- Distribucije grešaka (Normal i Long PRs)
- MAE po segmentima
- Actual vs Predicted scatter plotove
- Poboljšanje po pojedinačnim PR-ovima

### Sažetak Rezultata

```bash
python3 model_ensemble_v3_summary.py
```

## Logika Modela

Model automatski bira odgovarajući model na temelju thresholda:

```python
if predicted_time <= 2880:  # ili actual_time <= 2880
    use Normal Model
else:
    use Long Model
```

**Threshold**: 2880 minuta (75. percentil training seta)

## Detaljna Analiza

### Poboljšanje na Velikim PR-ovima

Long Model pokazuje značajno poboljšanje na svim velikim PR-ovima osim 2 slučaja gdje je Normal Model bio bolji (ali i dalje s velikom greškom).

### Statistike Grešaka (Veliki PR-ovi)

| Metrika | Normal Model | Long Model | Poboljšanje |
|---------|--------------|------------|-------------|
| Prosječna greška | 4924.5 min | 2853.0 min | ↓ 42.1% |
| Medijan greške | 2536.7 min | 1337.1 min | ↓ 47.3% |
| Std devijacija | 6435.6 min | 3824.7 min | ↓ 40.6% |
| Maksimalna greška | 17464.7 min | 10476.5 min | ↓ 40.0% |

## Preporuke

1. ✅ Koristiti **Normal Model** za PR-ove ≤ 2880 minuta
2. ✅ Koristiti **Long Model** za PR-ove > 2880 minuta
3. ✅ **Ensemble pristup** osigurava optimalne predikcije za sve tipove PR-ova

## Razlika od model_final_v2.py

- `model_final_v2.py` koristi jedan model s post-processing calibration za velike PR-ove
- `model_ensemble_v3.py` koristi **dva odvojena modela** trenirana specifično za različite tipove PR-ova
- Ensemble pristup daje bolje rezultate jer svaki model je optimiziran za svoj segment
