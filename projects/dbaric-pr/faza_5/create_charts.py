"""
Faza 5: Ponovna Analiza Feature Importance - Chartovi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
df = pd.read_csv('feature_importance_results.csv')

# Postavi stil
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Chart 1: Top 10 Feature Importance (nakon čišćenja)
fig, ax = plt.subplots(figsize=(10, 6))
top_features = df.nlargest(10, 'rf_importance')
ax.barh(range(len(top_features)), top_features['rf_importance'], color='forestgreen', edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=10)
ax.set_xlabel('važnost (random forest)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 najvažnijih feature-a (nakon čišćenja)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Dodaj vrijednosti
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax.text(row['rf_importance'], i, f' {row["rf_importance"]:.3f}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('faza_5_feature_importance_cleaned.png', dpi=300, bbox_inches='tight')
print("✅ Kreiran chart: faza_5_feature_importance_cleaned.png")
plt.close()
