"""
Faza 10: Ensemble Model - Chartovi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
try:
    metrics = pd.read_csv('model_ensemble_v3_metrics.csv')
    
    # Postavi stil
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    
    # Chart 1: Usporedba Modela (R² Score)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filtriraj samo relevantne modele za prikaz
    display_models = metrics[metrics['model'].isin([
        'Normal Model (Normal PRs)',
        'Long Model (Long PRs)',
        'Ensemble (All PRs)'
    ])].copy()
    
    colors = ['steelblue', 'forestgreen', 'orange']
    bars = ax.bar(range(len(display_models)), display_models['r2'], 
                  color=colors[:len(display_models)], edgecolor='black', alpha=0.8)
    
    ax.set_xticks(range(len(display_models)))
    ax.set_xticklabels([m.replace(' (Normal PRs)', '').replace(' (Long PRs)', '').replace(' (All PRs)', '') 
                        for m in display_models['model']], fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('R² score', fontsize=12, fontweight='bold')
    ax.set_title('Usporedba performansi ensemble modela', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(display_models['r2']) * 1.2])
    ax.grid(axis='y', alpha=0.3)
    
    # Dodaj vrijednosti
    for i, (idx, row) in enumerate(display_models.iterrows()):
        ax.text(i, row['r2'], f'{row["r2"]:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('faza_10_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Kreiran chart: faza_10_ensemble_comparison.png")
    plt.close()
except Exception as e:
    print(f"⚠️  Greška pri kreiranju charta: {e}")
