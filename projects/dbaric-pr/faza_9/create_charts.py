"""
Faza 9: Napredna Poboljšanja - Chartovi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
try:
    baseline_metrics = pd.read_csv('model_final_metrics.csv')
    
    # Postavi stil
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    
    # Chart 1: Poboljšanje Metrika (Baseline vs Final)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    baseline = baseline_metrics[baseline_metrics['model'] == 'Baseline'].iloc[0]
    final = baseline_metrics[baseline_metrics['model'] == 'Final'].iloc[0]
    
    metrics = ['R²', 'RMSE', 'MAE']
    baseline_vals = [baseline['r2'], baseline['rmse'], baseline['mae']]
    final_vals = [final['r2'], final['rmse'], final['mae']]
    
    x = range(len(metrics))
    width = 0.35
    
    for i, (metric, base_val, fin_val) in enumerate(zip(metrics, baseline_vals, final_vals)):
        bars1 = axes[i].bar([0], [base_val], width, label='Baseline', color='lightcoral', edgecolor='black')
        bars2 = axes[i].bar([1], [fin_val], width, label='Final', color='steelblue', edgecolor='black')
        
        axes[i].set_ylabel(metric, fontsize=11, fontweight='bold')
        axes[i].set_title(f'{metric} score', fontsize=12, fontweight='bold')
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Baseline', 'Final'], fontsize=10)
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].legend(fontsize=9)
        
        # Dodaj vrijednosti
        for bar in bars1:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Dodaj poboljšanje
        if metric == 'R²':
            improvement = fin_val - base_val
            improvement_pct = (improvement / base_val) * 100
        else:
            improvement = base_val - fin_val
            improvement_pct = (improvement / base_val) * 100
        
        axes[i].text(0.5, 0.95, f'{improvement_pct:+.1f}%',
                    transform=axes[i].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10, fontweight='bold')
    
    plt.suptitle('Poboljšanje performansi nakon naprednih poboljšanja', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('faza_9_advanced_improvements.png', dpi=300, bbox_inches='tight')
    print("✅ Kreiran chart: faza_9_advanced_improvements.png")
    plt.close()
except Exception as e:
    print(f"⚠️  Greška pri kreiranju charta: {e}")
