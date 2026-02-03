"""
Faza 4: Eksplorativna Analiza - Chartovi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Učitaj podatke
top_corr = pd.read_csv('top_correlations.csv')

# Postavi stil
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Chart 1: Top 10 Korelacije
fig, ax = plt.subplots(figsize=(10, 6))
top_10 = top_corr.head(10)
y_pos = np.arange(len(top_10))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_10)))

bars = ax.barh(y_pos, top_10['correlation'], color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['feature_1']} ↔ {row['feature_2']}" for _, row in top_10.iterrows()], 
                   fontsize=9)
ax.set_xlabel('korelacija', fontsize=12, fontweight='bold')
ax.set_title('Top 10 najviših korelacija između feature-a', fontsize=14, fontweight='bold')
ax.set_xlim([0.85, 1.0])
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Dodaj vrijednosti
for i, (idx, row) in enumerate(top_10.iterrows()):
    ax.text(row['correlation'], i, f' {row["correlation"]:.3f}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('faza_4_top_correlations.png', dpi=300, bbox_inches='tight')
print("✅ Kreiran chart: faza_4_top_correlations.png")
plt.close()
