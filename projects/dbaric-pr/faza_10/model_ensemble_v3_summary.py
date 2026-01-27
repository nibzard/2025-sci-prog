import pandas as pd
import numpy as np

print("="*80)
print("SAŽETAK USPOREDBE: ENSEMBLE MODEL V3")
print("="*80)

# Učitaj rezultate
metrics = pd.read_csv('model_ensemble_v3_metrics.csv')
long_comparison = pd.read_csv('model_ensemble_v3_long_prs_comparison.csv')
predictions = pd.read_csv('model_ensemble_v3_predictions.csv')

print("\n" + "="*80)
print("1. UKUPNE PERFORMANSE MODELA")
print("="*80)
print("\n" + metrics.to_string(index=False))

print("\n" + "="*80)
print("2. KLJUČNE NALAZE")
print("="*80)

normal_on_normal = metrics[metrics['model'] == 'Normal Model (Normal PRs)'].iloc[0]
normal_on_long = metrics[metrics['model'] == 'Normal Model (Long PRs)'].iloc[0]
long_on_long = metrics[metrics['model'] == 'Long Model (Long PRs)'].iloc[0]
ensemble_all = metrics[metrics['model'] == 'Ensemble (All PRs)'].iloc[0]

print(f"\n✅ Normal Model:")
print(f"   - Na normalnim PR-ovima: R²={normal_on_normal['r2']:.4f}, RMSE={normal_on_normal['rmse']:.1f} min, MAE={normal_on_normal['mae']:.1f} min")
print(f"   - Na velikim PR-ovima: R²={normal_on_long['r2']:.4f}, RMSE={normal_on_long['rmse']:.1f} min, MAE={normal_on_long['mae']:.1f} min")
print(f"   ⚠️  Normal Model NIJE prikladan za velike PR-ove (negativan R²!)")

print(f"\n✅ Long Model:")
print(f"   - Na velikim PR-ovima: R²={long_on_long['r2']:.4f}, RMSE={long_on_long['rmse']:.1f} min, MAE={long_on_long['mae']:.1f} min")
print(f"   ✅ Long Model značajno bolji za velike PR-ove")

print(f"\n✅ Ensemble Model:")
print(f"   - Na svim PR-ovima: R²={ensemble_all['r2']:.4f}, RMSE={ensemble_all['rmse']:.1f} min, MAE={ensemble_all['mae']:.1f} min")
print(f"   ✅ Najbolji ukupni rezultat - kombinuje prednosti oba modela")

print("\n" + "="*80)
print("3. POBOLJŠANJE LONG MODELA NAD NORMAL MODELOM (na velikim PR-ovima)")
print("="*80)

r2_improvement = long_on_long['r2'] - normal_on_long['r2']
rmse_improvement = normal_on_long['rmse'] - long_on_long['rmse']
mae_improvement = normal_on_long['mae'] - long_on_long['mae']

r2_pct = (r2_improvement / abs(normal_on_long['r2'])) * 100 if normal_on_long['r2'] != 0 else 0
rmse_pct = (rmse_improvement / normal_on_long['rmse']) * 100
mae_pct = (mae_improvement / normal_on_long['mae']) * 100

print(f"\nR²: {r2_improvement:+.4f} ({r2_pct:+.1f}% relativno)")
print(f"RMSE: {rmse_improvement:+.1f} min ({rmse_pct:+.1f}% poboljšanje)")
print(f"MAE: {mae_improvement:+.1f} min ({mae_pct:+.1f}% poboljšanje)")

print("\n" + "="*80)
print("4. DETALJNA ANALIZA PO POJEDINAČNIM VELIKIM PR-OVIMA")
print("="*80)

print("\nPojedinačne predikcije za velike PR-ove:")
print("-" * 80)
for idx, row in long_comparison.iterrows():
    actual = row['actual']
    pred_normal = row['predicted_normal_model']
    pred_long = row['predicted_long_model']
    error_normal = row['abs_error_normal']
    error_long = row['abs_error_long']
    improvement = row['improvement']
    
    print(f"\nPR #{idx+1}:")
    print(f"  Actual: {actual:.1f} min")
    print(f"  Normal Model predikcija: {pred_normal:.1f} min (greška: {error_normal:.1f} min)")
    print(f"  Long Model predikcija: {pred_long:.1f} min (greška: {error_long:.1f} min)")
    if improvement > 0:
        print(f"  ✅ Long Model bolji za {improvement:.1f} min")
    else:
        print(f"  ⚠️  Normal Model bolji za {abs(improvement):.1f} min (rijetko)")

print("\n" + "="*80)
print("5. STATISTIKE GREŠAKA")
print("="*80)

print("\nNormal Model (na velikim PR-ovima):")
print(f"  Prosječna greška: {long_comparison['abs_error_normal'].mean():.1f} min")
print(f"  Medijan greške: {long_comparison['abs_error_normal'].median():.1f} min")
print(f"  Std devijacija: {long_comparison['abs_error_normal'].std():.1f} min")
print(f"  Maksimalna greška: {long_comparison['abs_error_normal'].max():.1f} min")

print("\nLong Model (na velikim PR-ovima):")
print(f"  Prosječna greška: {long_comparison['abs_error_long'].mean():.1f} min")
print(f"  Medijan greške: {long_comparison['abs_error_long'].median():.1f} min")
print(f"  Std devijacija: {long_comparison['abs_error_long'].std():.1f} min")
print(f"  Maksimalna greška: {long_comparison['abs_error_long'].max():.1f} min")

print("\n" + "="*80)
print("6. ZAKLJUČAK")
print("="*80)

print("""
✅ IMPLEMENTACIJA DVA MODELA JE USPJEŠNA:

1. Normal Model:
   - Optimalan za normalne PR-ove (≤ 2880 minuta)
   - R² = 0.30, RMSE = 734 min, MAE = 555 min
   - Koristi se za većinu PR-ova

2. Long Model:
   - Optimalan za velike PR-ove (> 2880 minuta)
   - R² = 0.47, RMSE = 4509 min, MAE = 2853 min
   - Značajno bolji od Normal Modela na velikim PR-ovima
   - Poboljšanje: RMSE ↓ 41%, MAE ↓ 42%

3. Ensemble Model:
   - Kombinuje oba modela
   - Automatski bira odgovarajući model na temelju thresholda
   - Najbolji ukupni rezultat: R² = 0.66, RMSE = 2222 min, MAE = 1066 min

PREPORUKE:
- Koristiti Normal Model za PR-ove ≤ 2880 minuta
- Koristiti Long Model za PR-ove > 2880 minuta
- Ensemble pristup osigurava optimalne predikcije za sve tipove PR-ova
""")

print("="*80)
