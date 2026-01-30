import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9

print("Kreiranje analitičkog dashboarda za Ensemble Model...")

# Učitaj podatke
metrics = pd.read_csv('model_ensemble_v3_metrics.csv')
predictions = pd.read_csv('model_ensemble_v3_predictions.csv')

# Izvuci ensemble metrike
ensemble_metrics = metrics[metrics['model'] == 'Ensemble (All PRs)'].iloc[0]

# Kreiraj figure
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3, left=0.05, right=0.98, top=0.96, bottom=0.04)
fig.suptitle('ENSEMBLE MODEL V3 - ANALITIČKI PREGLED', 
             fontsize=20, fontweight='bold', y=0.99)

# ============================================================================
# 1. KLJUČNE METRIKE ENSEMBLE MODELA
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.axis('off')

metrics_data = [
    ['Metrika', 'Vrijednost'],
    ['R² Score', f"{ensemble_metrics['r2']:.4f}"],
    ['RMSE', f"{ensemble_metrics['rmse']:.1f} min"],
    ['MAE', f"{ensemble_metrics['mae']:.1f} min"],
    ['Broj uzoraka', f"{int(ensemble_metrics['n_samples'])}"],
    ['Threshold', f"{predictions[predictions['is_long'] == True]['actual'].min():.0f} min"],
    ['Normal PRs', f"{(predictions['is_long'] == False).sum()}"],
    ['Long PRs', f"{(predictions['is_long'] == True).sum()}"]
]

table = ax1.table(cellText=metrics_data, cellLoc='center', loc='center',
                  colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Stiliziraj header
for i in range(2):
    table[(0, i)].set_facecolor('#6C5CE7')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Stiliziraj redove
for i in range(1, len(metrics_data)):
    for j in range(2):
        table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')
        if i == 1:  # R² row
            table[(i, j)].set_text_props(weight='bold', color='#2E7D32')
        elif i in [2, 3]:  # RMSE, MAE rows
            table[(i, j)].set_text_props(weight='bold', color='#C62828')

ax1.set_title('Ključne Metrike Ensemble Modela', fontsize=13, fontweight='bold', pad=15)

# ============================================================================
# 2. ACTUAL VS PREDICTED - SVI PR-OVI
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2:])
colors_scatter = ['#4472C4' if not is_long else '#FF6B6B' for is_long in predictions['is_long']]
ax2.scatter(predictions['actual'], predictions['predicted_ensemble'], 
           alpha=0.7, c=colors_scatter, s=80, edgecolors='black', linewidth=0.5)

min_val = min(predictions['actual'].min(), predictions['predicted_ensemble'].min())
max_val = max(predictions['actual'].max(), predictions['predicted_ensemble'].max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Idealna linija', alpha=0.8)

ax2.set_xlabel('Actual (minute)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted (minute)', fontsize=11, fontweight='bold')
ax2.set_title(f'Actual vs Predicted - Ensemble Model\n(R²={ensemble_metrics["r2"]:.4f}, RMSE={ensemble_metrics["rmse"]:.1f} min)', 
              fontsize=12, fontweight='bold')
ax2.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4472C4', 
               markersize=10, label='Normal PRs', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
               markersize=10, label='Long PRs', markeredgecolor='black')
], fontsize=9)
ax2.grid(alpha=0.3)

# ============================================================================
# 3. DISTRIBUCIJA GREŠAKA - HISTOGRAM
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(predictions['abs_error_ensemble'], bins=20, color='#6C5CE7', 
        alpha=0.7, edgecolor='black', linewidth=1.2)
mean_err = predictions['abs_error_ensemble'].mean()
median_err = predictions['abs_error_ensemble'].median()
ax3.axvline(mean_err, color='red', linestyle='--', linewidth=2.5, 
           label=f'Mean: {mean_err:.0f} min')
ax3.axvline(median_err, color='green', linestyle='--', linewidth=2.5, 
           label=f'Median: {median_err:.0f} min')
ax3.set_xlabel('Apsolutna Greška (minute)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Frekvencija', fontsize=10, fontweight='bold')
ax3.set_title('Distribucija Apsolutnih Grešaka', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')

# ============================================================================
# 4. DISTRIBUCIJA GREŠAKA - BOX PLOT PO SEGMENTIMA
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])
segment_order = ['Very Short', 'Short', 'Medium', 'Long']
segment_data = [predictions[predictions['segment'] == seg]['abs_error_ensemble'].values 
                for seg in segment_order]
bp = ax4.boxplot(segment_data, labels=segment_order, patch_artist=True,
                 boxprops=dict(facecolor='#6C5CE7', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
ax4.set_ylabel('Apsolutna Greška (minute)', fontsize=10, fontweight='bold')
ax4.set_title('Distribucija Grešaka po Segmentima', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# ============================================================================
# 5. MAE PO SEGMENTIMA
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2:])
mae_by_segment = predictions.groupby('segment')['abs_error_ensemble'].agg(['mean', 'median', 'std'])
mae_mean = [mae_by_segment.loc[seg, 'mean'] for seg in segment_order]
mae_median = [mae_by_segment.loc[seg, 'median'] for seg in segment_order]

x = np.arange(len(segment_order))
width = 0.35
bars1 = ax5.bar(x - width/2, mae_mean, width, label='Mean MAE', 
                color='#4472C4', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax5.bar(x + width/2, mae_median, width, label='Median MAE', 
                color='#52B788', alpha=0.8, edgecolor='black', linewidth=1.2)

ax5.set_xlabel('Segment', fontsize=10, fontweight='bold')
ax5.set_ylabel('MAE (minute)', fontsize=10, fontweight='bold')
ax5.set_title('Mean i Median MAE po Segmentima', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(segment_order)
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ============================================================================
# 6. RESIDUALS PLOT
# ============================================================================
ax6 = fig.add_subplot(gs[2, 0])
residuals = predictions['actual'] - predictions['predicted_ensemble']
ax6.scatter(predictions['predicted_ensemble'], residuals, 
           alpha=0.7, c=colors_scatter, s=80, edgecolors='black', linewidth=0.5)
ax6.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax6.set_xlabel('Predicted (minute)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Residuals (Actual - Predicted)', fontsize=10, fontweight='bold')
ax6.set_title('Residuals Plot', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)

# ============================================================================
# 7. STATISTIKE GREŠAKA - TABLICA
# ============================================================================
ax7 = fig.add_subplot(gs[2, 1])
ax7.axis('off')

error_stats = predictions['abs_error_ensemble'].describe()
percentiles = predictions['abs_error_ensemble'].quantile([0.25, 0.50, 0.75, 0.90, 0.95])

stats_data = [
    ['Statistika', 'Vrijednost (min)'],
    ['Mean', f"{error_stats['mean']:.1f}"],
    ['Median', f"{error_stats['50%']:.1f}"],
    ['Std Dev', f"{error_stats['std']:.1f}"],
    ['Min', f"{error_stats['min']:.1f}"],
    ['Max', f"{error_stats['max']:.1f}"],
    ['Q25', f"{percentiles[0.25]:.1f}"],
    ['Q75', f"{percentiles[0.75]:.1f}"],
    ['Q90', f"{percentiles[0.90]:.1f}"],
    ['Q95', f"{percentiles[0.95]:.1f}"]
]

table2 = ax7.table(cellText=stats_data, cellLoc='center', loc='center',
                   colWidths=[0.5, 0.5])
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 1.8)

for i in range(2):
    table2[(0, i)].set_facecolor('#52B788')
    table2[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(stats_data)):
    for j in range(2):
        table2[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')

ax7.set_title('Statistike Apsolutnih Grešaka', fontsize=12, fontweight='bold', pad=10)

# ============================================================================
# 8. DISTRIBUCIJA PO TIPOVIMA PR-OVA
# ============================================================================
ax8 = fig.add_subplot(gs[2, 2:])
normal_prs = predictions[predictions['is_long'] == False]
long_prs = predictions[predictions['is_long'] == True]

categories = ['Normal PRs', 'Long PRs']
counts = [len(normal_prs), len(long_prs)]
mae_values = [normal_prs['abs_error_ensemble'].mean(), long_prs['abs_error_ensemble'].mean()]
colors_pie = ['#4472C4', '#FF6B6B']

# Pie chart za distribuciju
ax8_twin = ax8.twinx()
wedges, texts, autotexts = ax8.pie(counts, labels=categories, autopct='%1.1f%%',
                                   colors=colors_pie, startangle=90,
                                   textprops={'fontsize': 10, 'fontweight': 'bold'})
ax8.set_title('Distribucija PR-ova i Prosječne Greške', fontsize=12, fontweight='bold')

# Dodaj tekst s MAE vrijednostima
for i, (cat, mae) in enumerate(zip(categories, mae_values)):
    ax8.text(0, -1.3 - i*0.3, f'{cat}: MAE = {mae:.1f} min', 
            ha='center', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor=colors_pie[i], alpha=0.3))

# ============================================================================
# 9. Q-Q PLOT ZA RESIDUALE
# ============================================================================
ax9 = fig.add_subplot(gs[3, 0])
stats.probplot(residuals, dist="norm", plot=ax9)
ax9.set_title('Q-Q Plot Residuala', fontsize=12, fontweight='bold')
ax9.grid(alpha=0.3)

# ============================================================================
# 10. ERROR BY MAGNITUDE
# ============================================================================
ax10 = fig.add_subplot(gs[3, 1])
ax10.scatter(predictions['actual'], predictions['abs_error_ensemble'], 
            alpha=0.7, c=colors_scatter, s=80, edgecolors='black', linewidth=0.5)
ax10.set_xlabel('Actual Value (minute)', fontsize=10, fontweight='bold')
ax10.set_ylabel('Absolute Error (minute)', fontsize=10, fontweight='bold')
ax10.set_title('Greška vs Magnituda', fontsize=12, fontweight='bold')
ax10.grid(alpha=0.3)

# Trend line
z = np.polyfit(predictions['actual'], predictions['abs_error_ensemble'], 1)
p = np.poly1d(z)
ax10.plot(predictions['actual'], p(predictions['actual']), "r--", alpha=0.8, linewidth=2)

# ============================================================================
# 11. PERFORMANSE PO SEGMENTIMA - DETALJNO
# ============================================================================
ax11 = fig.add_subplot(gs[3, 2:])
segment_perf = []
for seg in segment_order:
    seg_data = predictions[predictions['segment'] == seg]
    if len(seg_data) > 0:
        r2_seg = 1 - (np.sum((seg_data['actual'] - seg_data['predicted_ensemble'])**2) / 
                      np.sum((seg_data['actual'] - seg_data['actual'].mean())**2))
        rmse_seg = np.sqrt(np.mean((seg_data['actual'] - seg_data['predicted_ensemble'])**2))
        mae_seg = seg_data['abs_error_ensemble'].mean()
        segment_perf.append([seg, r2_seg, rmse_seg, mae_seg])

perf_df = pd.DataFrame(segment_perf, columns=['Segment', 'R²', 'RMSE', 'MAE'])

x_perf = np.arange(len(segment_order))
width_perf = 0.25

r2_values_perf = [perf_df[perf_df['Segment'] == seg]['R²'].values[0] if seg in perf_df['Segment'].values else 0 
                  for seg in segment_order]
rmse_values_perf = [perf_df[perf_df['Segment'] == seg]['RMSE'].values[0] if seg in perf_df['Segment'].values else 0 
                    for seg in segment_order]
mae_values_perf = [perf_df[perf_df['Segment'] == seg]['MAE'].values[0] if seg in perf_df['Segment'].values else 0 
                   for seg in segment_order]

# Normaliziraj za prikaz (R² na lijevoj y-osi, RMSE/MAE na desnoj)
ax11_twin = ax11.twinx()

bars1 = ax11.bar(x_perf - width_perf, r2_values_perf, width_perf, 
                label='R²', color='#6C5CE7', alpha=0.8, edgecolor='black')
bars2 = ax11_twin.bar(x_perf, rmse_values_perf, width_perf, 
                     label='RMSE (min)', color='#FF6B6B', alpha=0.8, edgecolor='black')
bars3 = ax11_twin.bar(x_perf + width_perf, mae_values_perf, width_perf, 
                     label='MAE (min)', color='#52B788', alpha=0.8, edgecolor='black')

ax11.set_xlabel('Segment', fontsize=10, fontweight='bold')
ax11.set_ylabel('R² Score', fontsize=10, fontweight='bold', color='#6C5CE7')
ax11_twin.set_ylabel('RMSE / MAE (minute)', fontsize=10, fontweight='bold')
ax11.set_xticks(x_perf)
ax11.set_xticklabels(segment_order)
ax11.set_title('Performanse po Segmentima', fontsize=12, fontweight='bold')
ax11.tick_params(axis='y', labelcolor='#6C5CE7')
ax11.grid(axis='y', alpha=0.3)

# Legende
lines1, labels1 = ax11.get_legend_handles_labels()
lines2, labels2 = ax11_twin.get_legend_handles_labels()
ax11.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.savefig('model_ensemble_v3_analytics_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Kreiran 'model_ensemble_v3_analytics_dashboard.png'")
