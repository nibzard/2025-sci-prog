"""
Faza 2: Feature Engineering - Chartovi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
try:
    df = pd.read_csv('source_v2.csv')
except:
    df = pd.read_csv('source.csv')

# Postavi stil
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Chart 1: Distribucija efektivnog vremena
fig, ax = plt.subplots(figsize=(10, 6))
if 'effective_minutes' in df.columns:
    effective_time = df['effective_minutes']
elif 'effective_hours' in df.columns:
    effective_time = df['effective_hours'] * 60  # Konvertiraj u minute
else:
    effective_time = None

if effective_time is not None:
    effective_time.hist(bins=30, color='steelblue', edgecolor='black', ax=ax)
    ax.set_xlabel('efektivne minute', fontsize=12, fontweight='bold')
    ax.set_ylabel('broj PR-ova', fontsize=12, fontweight='bold')
    ax.set_title('Distribucija efektivnog vremena do merge-a', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Dodaj statistike
    mean_val = effective_time.mean()
    median_val = effective_time.median()
    stats_text = f'Prosjek: {mean_val:.0f} min\nMedijan: {median_val:.0f} min'
    ax.text(0.65, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontweight='bold')
else:
    ax.text(0.5, 0.5, 'Nema podataka o efektivnom vremenu', 
            ha='center', va='center', fontsize=14)
    ax.set_title('Distribucija efektivnog vremena', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('faza_2_effective_time_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Kreiran chart: faza_2_effective_time_distribution.png")
plt.close()
