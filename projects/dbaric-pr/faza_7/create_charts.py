"""
Faza 7: Poboljšanje Kvalitete Podataka - Chartovi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
try:
    df = pd.read_csv('source.csv')
    
    # Postavi stil
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    
    # Chart 1: Distribucija PR-ova po Tipu (ako postoji)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tip PR-a
    if 'is_bug_fix' in df.columns or 'is_new_feature' in df.columns:
        pr_types = []
        if 'is_bug_fix' in df.columns:
            pr_types.append(('Bug fix', df['is_bug_fix'].sum()))
        if 'is_new_feature' in df.columns:
            pr_types.append(('New feature', df['is_new_feature'].sum()))
        if 'is_update' in df.columns:
            pr_types.append(('Update', df['is_update'].sum()))
        
        if pr_types:
            types, counts = zip(*pr_types)
            axes[0].bar(types, counts, color=['crimson', 'steelblue', 'forestgreen'][:len(types)], 
                       edgecolor='black')
            axes[0].set_ylabel('broj PR-ova', fontsize=12, fontweight='bold')
            axes[0].set_title('Distribucija PR-ova po tipu', fontsize=13, fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)
            for i, (t, c) in enumerate(zip(types, counts)):
                axes[0].text(i, c, f' {c}', va='bottom', ha='center', fontsize=10, fontweight='bold')
    
    # Tehnologija
    if 'is_backend' in df.columns or 'is_frontend' in df.columns:
        tech_counts = []
        tech_labels = []
        if 'is_backend' in df.columns:
            tech_counts.append(df['is_backend'].sum())
            tech_labels.append('Backend')
        if 'is_frontend' in df.columns:
            tech_counts.append(df['is_frontend'].sum())
            tech_labels.append('Frontend')
        if 'is_backend' in df.columns and 'is_frontend' in df.columns:
            both = ((df['is_backend'] == True) & (df['is_frontend'] == True)).sum()
            if both > 0:
                tech_counts.append(both)
                tech_labels.append('Fullstack')
        
        if tech_counts:
            axes[1].bar(tech_labels, tech_counts, color=['steelblue', 'forestgreen', 'orange'][:len(tech_labels)],
                       edgecolor='black')
            axes[1].set_ylabel('broj PR-ova', fontsize=12, fontweight='bold')
            axes[1].set_title('Distribucija PR-ova po tehnologiji', fontsize=13, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
            for i, (l, c) in enumerate(zip(tech_labels, tech_counts)):
                axes[1].text(i, c, f' {c}', va='bottom', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('faza_7_data_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Kreiran chart: faza_7_data_distribution.png")
    plt.close()
except Exception as e:
    print(f"⚠️  Greška pri kreiranju charta: {e}")
