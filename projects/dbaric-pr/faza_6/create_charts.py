"""
Faza 6: Usporedba Algoritama - Chartovi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
df = pd.read_csv('algorithm_comparison_results.csv')

# Postavi stil
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Chart 1: Usporedba R² Score po Algoritmima
fig, ax = plt.subplots(figsize=(10, 6))
df_sorted = df.sort_values('test_r2', ascending=True)
colors = ['crimson' if x < 0 else 'steelblue' for x in df_sorted['test_r2']]

bars = ax.barh(range(len(df_sorted)), df_sorted['test_r2'], color=colors, edgecolor='black')
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted['algorithm'], fontsize=10)
ax.set_xlabel('R² score', fontsize=12, fontweight='bold')
ax.set_title('Usporedba performansi algoritama', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Dodaj vrijednosti
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    ax.text(row['test_r2'], i, f' {row["test_r2"]:.3f}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('faza_6_algorithm_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Kreiran chart: faza_6_algorithm_comparison.png")
plt.close()
