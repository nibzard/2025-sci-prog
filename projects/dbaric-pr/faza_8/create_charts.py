"""
Faza 8: XGBoost Optimizacija - Chartovi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
try:
    metrics = pd.read_csv('model_scores.csv')
    improved_metrics = pd.read_csv('model_scores_improved.csv')
    
    # Postavi stil
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    
    # Chart 1: Poboljšanje Performansi (R² Score)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_r2 = metrics[metrics['metric'] == 'R²']['test'].values[0]
    improved_r2 = improved_metrics[improved_metrics['metric'] == 'R²']['test'].values[0]
    
    models = ['Baseline\nXGBoost', 'Optimizirani\nXGBoost']
    r2_scores = [baseline_r2, improved_r2]
    colors = ['lightcoral', 'steelblue']
    
    bars = ax.bar(models, r2_scores, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('R² score', fontsize=12, fontweight='bold')
    ax.set_title('Poboljšanje performansi nakon optimizacije', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(r2_scores) * 1.15])
    ax.grid(axis='y', alpha=0.3)
    
    # Dodaj vrijednosti i poboljšanje
    for i, (bar, score) in enumerate(zip(bars, r2_scores)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    improvement = improved_r2 - baseline_r2
    improvement_pct = (improvement / baseline_r2) * 100
    ax.text(0.5, 0.95, f'Poboljšanje: +{improvement:.4f} (+{improvement_pct:.1f}%)',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('faza_8_optimization_improvement.png', dpi=300, bbox_inches='tight')
    print("✅ Kreiran chart: faza_8_optimization_improvement.png")
    plt.close()
except Exception as e:
    print(f"⚠️  Greška pri kreiranju charta: {e}")
