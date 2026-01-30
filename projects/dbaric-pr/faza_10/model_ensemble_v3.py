import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print("="*70)
print("ENSEMBLE MODEL V3 - DVA MODELA: STANDARDNI + VELIKI PR-OVI")
print("="*70)
print("\nStandardni model za normalne PR-ove, poseban model za velike PR-ove")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_csv('source.csv')
target_col = 'effective_minutes'
df = df[df[target_col].notna()].copy()
df = df[df[target_col] >= 0].copy()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Columns to exclude
exclude_cols = [
    'non_working_minutes', 'pr_number', 'pr_id',
    'created_at', 'closed_at', 'merged_at', 'updated_at',
    'ready_for_review_time', 'workflow_start_time',
    'first_review_time', 'first_approval_time',
    'title', 'description', 'body', 'author', 'merged_by_login', 'task_id',
]

feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
X = df[feature_cols].copy()
y = df[target_col].copy()
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_numeric = X.select_dtypes(include=[np.number])
X_numeric_imputed = pd.DataFrame(
    imputer.fit_transform(X_numeric),
    columns=X_numeric.columns,
    index=X_numeric.index
)

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
X_encoded = X_numeric_imputed.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_col_filled = X[col].fillna('unknown').astype(str)
    X_encoded[col] = le.fit_transform(X_col_filled)
    label_encoders[col] = le

X_base = X_encoded.reset_index(drop=True)
y_base = y.reset_index(drop=True)

# Feature Engineering
X_engineered = X_base.copy()

if 'additions' in X_engineered.columns and 'deletions' in X_engineered.columns:
    X_engineered['additions_deletions_ratio'] = (X_engineered['additions'] + 1) / (X_engineered['deletions'] + 1)
if 'commits' in X_engineered.columns and 'changed_files' in X_engineered.columns:
    X_engineered['commits_per_file'] = X_engineered['commits'] / (X_engineered['changed_files'] + 1)
if 'total_lines_changed' in X_engineered.columns and 'commits' in X_engineered.columns:
    X_engineered['lines_per_commit'] = X_engineered['total_lines_changed'] / (X_engineered['commits'] + 1)
if 'review_count' in X_engineered.columns and 'reviewer_count' in X_engineered.columns:
    X_engineered['reviews_per_reviewer'] = X_engineered['review_count'] / (X_engineered['reviewer_count'] + 1)
if 'time_to_first_review_minutes' in X_engineered.columns and 'time_to_first_approval_minutes' in X_engineered.columns:
    X_engineered['review_to_approval_time'] = np.maximum(0, 
        X_engineered['time_to_first_approval_minutes'] - X_engineered['time_to_first_review_minutes'])

important_features = ['time_to_first_approval_minutes', 'commits', 'review_count', 
                     'total_lines_changed', 'changed_files']
for feat in important_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_squared'] = X_engineered[feat] ** 2

skewed_features = ['additions', 'deletions', 'total_lines_changed', 'commits', 
                   'review_count', 'comments', 'review_comments']
for feat in skewed_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_log'] = np.log1p(X_engineered[feat])

if 'created_at' in df.columns:
    df['created_at_parsed'] = pd.to_datetime(df['created_at'], errors='coerce')
    X_engineered['created_hour'] = df['created_at_parsed'].dt.hour
    X_engineered['created_day_of_week'] = df['created_at_parsed'].dt.dayofweek
    X_engineered['created_is_weekend'] = (X_engineered['created_day_of_week'] >= 5).astype(int)

# Feature Selection
n_features_to_select = min(40, X_engineered.shape[1])
selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
X_selected = selector.fit_transform(X_engineered, y_base)
selected_features = X_engineered.columns[selector.get_support()].tolist()
X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X_engineered.index)

# Target Transformation
y_log = np.log1p(y_base)

# Train-Test Split
X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
    X_selected_df, y_base, y_log, test_size=0.2, random_state=123, shuffle=True
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Define Long segment threshold (75th percentile)
long_threshold = y_train.quantile(0.75)
print(f"\nLong PR threshold: {long_threshold:.2f} minutes")
print(f"Training - Normal PRs: {(y_train <= long_threshold).sum()} samples")
print(f"Training - Long PRs: {(y_train > long_threshold).sum()} samples")
print(f"Test - Normal PRs: {(y_test <= long_threshold).sum()} samples")
print(f"Test - Long PRs: {(y_test > long_threshold).sum()} samples")

# ============================================================================
# MODEL 1: STANDARDNI MODEL (za normalne PR-ove)
# ============================================================================
print("\n" + "="*70)
print("MODEL 1: STANDARDNI MODEL (za normalne PR-ove)")
print("="*70)

train_normal_mask = y_train <= long_threshold
X_train_normal = X_train[train_normal_mask]
y_train_normal = y_train[train_normal_mask]
y_train_normal_log = y_train_log[train_normal_mask]

model_normal = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model_normal.fit(X_train_normal, y_train_normal_log)

# Test na normalnim PR-ovima
test_normal_mask = y_test <= long_threshold
X_test_normal = X_test[test_normal_mask]
y_test_normal = y_test[test_normal_mask]

y_pred_normal_log = model_normal.predict(X_test_normal)
y_pred_normal = np.expm1(y_pred_normal_log)

normal_r2 = r2_score(y_test_normal, y_pred_normal)
normal_rmse = np.sqrt(mean_squared_error(y_test_normal, y_pred_normal))
normal_mae = mean_absolute_error(y_test_normal, y_pred_normal)

print(f"Normal Model Performance (on normal PRs):")
print(f"  R²: {normal_r2:.4f}")
print(f"  RMSE: {normal_rmse:.2f} minutes")
print(f"  MAE: {normal_mae:.2f} minutes")

# ============================================================================
# MODEL 2: MODEL ZA VELIKE PR-OVE
# ============================================================================
print("\n" + "="*70)
print("MODEL 2: MODEL ZA VELIKE PR-OVE")
print("="*70)

train_long_mask = y_train > long_threshold
X_train_long = X_train[train_long_mask]
y_train_long = y_train[train_long_mask]
y_train_long_log = y_train_log[train_long_mask]

print(f"Training samples for Long model: {len(X_train_long)}")

# Optimizirani hiperparametri za velike PR-ove (možda trebaju drugačiji pristup)
model_long = xgb.XGBRegressor(
    n_estimators=400,  # Više stabala za kompleksnije obrasce
    max_depth=6,  # Dublje stabla za kompleksnije obrasce
    learning_rate=0.02,  # Niža learning rate za bolju konvergenciju
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,  # Niža vrijednost za veće PR-ove
    reg_alpha=0.15,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model_long.fit(X_train_long, y_train_long_log)

# Test na velikim PR-ovima
test_long_mask = y_test > long_threshold
X_test_long = X_test[test_long_mask]
y_test_long = y_test[test_long_mask]

y_pred_long_log = model_long.predict(X_test_long)
y_pred_long = np.expm1(y_pred_long_log)

long_r2 = r2_score(y_test_long, y_pred_long)
long_rmse = np.sqrt(mean_squared_error(y_test_long, y_pred_long))
long_mae = mean_absolute_error(y_test_long, y_pred_long)

print(f"Long Model Performance (on long PRs):")
print(f"  R²: {long_r2:.4f}")
print(f"  RMSE: {long_rmse:.2f} minutes")
print(f"  MAE: {long_mae:.2f} minutes")

# ============================================================================
# ENSEMBLE PREDIKCIJE: Kombinacija oba modela
# ============================================================================
print("\n" + "="*70)
print("ENSEMBLE PREDIKCIJE")
print("="*70)

# Kreiraj predikcije za cijeli test set
y_pred_ensemble = np.zeros(len(y_test))
y_pred_ensemble[test_normal_mask] = y_pred_normal
y_pred_ensemble[test_long_mask] = y_pred_long

ensemble_r2 = r2_score(y_test, y_pred_ensemble)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)

print(f"Ensemble Model Performance (on full test set):")
print(f"  R²: {ensemble_r2:.4f}")
print(f"  RMSE: {ensemble_rmse:.2f} minutes")
print(f"  MAE: {ensemble_mae:.2f} minutes")

# ============================================================================
# USPOREDBA: Kako bi standardni model radio na velikim PR-ovima
# ============================================================================
print("\n" + "="*70)
print("USPOREDBA: Standardni model na velikim PR-ovima")
print("="*70)

y_pred_normal_on_long_log = model_normal.predict(X_test_long)
y_pred_normal_on_long = np.expm1(y_pred_normal_on_long_log)

normal_on_long_r2 = r2_score(y_test_long, y_pred_normal_on_long)
normal_on_long_rmse = np.sqrt(mean_squared_error(y_test_long, y_pred_normal_on_long))
normal_on_long_mae = mean_absolute_error(y_test_long, y_pred_normal_on_long)

print(f"Normal Model Performance (on long PRs - WRONG MODEL):")
print(f"  R²: {normal_on_long_r2:.4f}")
print(f"  RMSE: {normal_on_long_rmse:.2f} minutes")
print(f"  MAE: {normal_on_long_mae:.2f} minutes")

print(f"\nLong Model Performance (on long PRs - CORRECT MODEL):")
print(f"  R²: {long_r2:.4f}")
print(f"  RMSE: {long_rmse:.2f} minutes")
print(f"  MAE: {long_mae:.2f} minutes")

improvement_r2 = long_r2 - normal_on_long_r2
improvement_rmse = normal_on_long_rmse - long_rmse
improvement_mae = normal_on_long_mae - long_mae

print(f"\nImprovement using Long Model:")
print(f"  R²: {improvement_r2:+.4f} ({improvement_r2/normal_on_long_r2*100:+.2f}%)")
print(f"  RMSE: {improvement_rmse:+.2f} minutes ({improvement_rmse/normal_on_long_rmse*100:+.2f}%)")
print(f"  MAE: {improvement_mae:+.2f} minutes ({improvement_mae/normal_on_long_mae*100:+.2f}%)")

# ============================================================================
# DETALJNA ANALIZA PO SEGMENTIMA
# ============================================================================
print("\n" + "="*70)
print("DETALJNA ANALIZA PO SEGMENTIMA")
print("="*70)

# Segmentacija
q25 = y_test.quantile(0.25)
median = y_test.median()
q75 = y_test.quantile(0.75)

y_test_segments = pd.cut(y_test, 
                        bins=[0, q25, median, q75, float('inf')],
                        labels=['Very Short', 'Short', 'Medium', 'Long'])

# Kreiraj DataFrame za analizu
comparison_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted_normal': model_normal.predict(X_test),
    'predicted_long': np.zeros(len(y_test)),
    'predicted_ensemble': y_pred_ensemble,
    'segment': y_test_segments,
    'is_long': y_test > long_threshold
})

# Za normalne PR-ove, long model ne bi trebao biti korišten, ali za usporedbu
y_pred_long_full_log = model_long.predict(X_test)
y_pred_long_full = np.expm1(y_pred_long_full_log)
comparison_df['predicted_long'] = y_pred_long_full

# Transformiraj predikcije iz log space
comparison_df['predicted_normal'] = np.expm1(comparison_df['predicted_normal'])

# Izračunaj greške
comparison_df['error_normal'] = comparison_df['actual'] - comparison_df['predicted_normal']
comparison_df['error_long'] = comparison_df['actual'] - comparison_df['predicted_long']
comparison_df['error_ensemble'] = comparison_df['actual'] - comparison_df['predicted_ensemble']

comparison_df['abs_error_normal'] = np.abs(comparison_df['error_normal'])
comparison_df['abs_error_long'] = np.abs(comparison_df['error_long'])
comparison_df['abs_error_ensemble'] = np.abs(comparison_df['error_ensemble'])

# Statistike po segmentima
segment_comparison = comparison_df.groupby('segment').agg({
    'abs_error_normal': ['mean', 'median', 'std'],
    'abs_error_long': ['mean', 'median', 'std'],
    'abs_error_ensemble': ['mean', 'median', 'std']
}).round(2)

print("\nMean Absolute Error by Segment:")
print(segment_comparison)

# Statistike za normalne vs velike PR-ove
print("\n" + "-"*70)
print("Performance by PR Type:")
print("-"*70)

normal_prs_stats = comparison_df[comparison_df['is_long'] == False].agg({
    'abs_error_normal': ['mean', 'median'],
    'abs_error_ensemble': ['mean', 'median']
}).round(2)

long_prs_stats = comparison_df[comparison_df['is_long'] == True].agg({
    'abs_error_normal': ['mean', 'median'],
    'abs_error_long': ['mean', 'median'],
    'abs_error_ensemble': ['mean', 'median']
}).round(2)

print("\nNormal PRs (using Normal Model vs Ensemble):")
print(normal_prs_stats)

print("\nLong PRs (Normal Model vs Long Model vs Ensemble):")
print(long_prs_stats)

# ============================================================================
# VIZUALIZACIJA
# ============================================================================
print("\n" + "="*70)
print("KREIRANJE VIZUALIZACIJA")
print("="*70)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('Ensemble Model V3 - Comparison: Normal vs Long PR Models', 
             fontsize=18, fontweight='bold', y=0.995)

# Plot 1: Normal Model - Actual vs Predicted (normalne PR-ove)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test_normal.values, y_pred_normal, alpha=0.6, color='steelblue', s=50)
ax1.plot([y_test_normal.min(), y_test_normal.max()], 
         [y_test_normal.min(), y_test_normal.max()], 'r--', lw=2)
ax1.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax1.set_title(f'Normal Model - Normal PRs\n(R²={normal_r2:.4f}, RMSE={normal_rmse:.1f})', 
              fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Plot 2: Long Model - Actual vs Predicted (velike PR-ove)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test_long.values, y_pred_long, alpha=0.6, color='crimson', s=50)
ax2.plot([y_test_long.min(), y_test_long.max()], 
         [y_test_long.min(), y_test_long.max()], 'r--', lw=2)
ax2.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax2.set_title(f'Long Model - Long PRs\n(R²={long_r2:.4f}, RMSE={long_rmse:.1f})', 
              fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Plot 3: Ensemble - Actual vs Predicted (svi PR-ovi)
ax3 = fig.add_subplot(gs[0, 2])
colors = ['steelblue' if not is_long else 'crimson' for is_long in comparison_df['is_long']]
ax3.scatter(comparison_df['actual'], comparison_df['predicted_ensemble'], 
           alpha=0.6, c=colors, s=50)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax3.set_title(f'Ensemble Model - All PRs\n(R²={ensemble_r2:.4f}, RMSE={ensemble_rmse:.1f})', 
              fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)
ax3.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', 
               markersize=8, label='Normal PRs'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', 
               markersize=8, label='Long PRs')
], loc='upper left')

# Plot 4: Comparison - Normal Model na velikim PR-ovima (LOŠE)
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_test_long.values, y_pred_normal_on_long, alpha=0.6, color='orange', s=50)
ax4.plot([y_test_long.min(), y_test_long.max()], 
         [y_test_long.min(), y_test_long.max()], 'r--', lw=2)
ax4.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax4.set_title(f'Normal Model - Long PRs (WRONG)\n(R²={normal_on_long_r2:.4f}, RMSE={normal_on_long_rmse:.1f})', 
              fontsize=12, fontweight='bold', color='darkred')
ax4.grid(alpha=0.3)

# Plot 5: Comparison - Long Model na velikim PR-ovima (DOBRO)
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y_test_long.values, y_pred_long, alpha=0.6, color='forestgreen', s=50)
ax5.plot([y_test_long.min(), y_test_long.max()], 
         [y_test_long.min(), y_test_long.max()], 'r--', lw=2)
ax5.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax5.set_title(f'Long Model - Long PRs (CORRECT)\n(R²={long_r2:.4f}, RMSE={long_rmse:.1f})', 
              fontsize=12, fontweight='bold', color='darkgreen')
ax5.grid(alpha=0.3)

# Plot 6: MAE Comparison by Segment
ax6 = fig.add_subplot(gs[1, 2])
segment_order = ['Very Short', 'Short', 'Medium', 'Long']
mae_normal = comparison_df.groupby('segment')['abs_error_normal'].mean()
mae_long = comparison_df.groupby('segment')['abs_error_long'].mean()
mae_ensemble = comparison_df.groupby('segment')['abs_error_ensemble'].mean()

x = np.arange(len(segment_order))
width = 0.25
ax6.bar(x - width, [mae_normal.get(seg, 0) for seg in segment_order], 
        width, label='Normal Model', color='steelblue', alpha=0.8)
ax6.bar(x, [mae_long.get(seg, 0) for seg in segment_order], 
        width, label='Long Model', color='crimson', alpha=0.8)
ax6.bar(x + width, [mae_ensemble.get(seg, 0) for seg in segment_order], 
        width, label='Ensemble', color='forestgreen', alpha=0.8)

ax6.set_xlabel('Segment', fontsize=11, fontweight='bold')
ax6.set_ylabel('Mean Absolute Error (minutes)', fontsize=11, fontweight='bold')
ax6.set_title('MAE Comparison by Segment', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(segment_order, rotation=45, ha='right')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Plot 7: Error Distribution - Normal PRs
ax7 = fig.add_subplot(gs[2, 0])
normal_prs_errors = comparison_df[comparison_df['is_long'] == False]['abs_error_ensemble']
ax7.hist(normal_prs_errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax7.axvline(normal_prs_errors.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {normal_prs_errors.mean():.1f}')
ax7.set_xlabel('Absolute Error (minutes)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax7.set_title('Error Distribution - Normal PRs', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# Plot 8: Error Distribution - Long PRs
ax8 = fig.add_subplot(gs[2, 1])
long_prs_errors = comparison_df[comparison_df['is_long'] == True]['abs_error_ensemble']
ax8.hist(long_prs_errors, bins=30, color='crimson', alpha=0.7, edgecolor='black')
ax8.axvline(long_prs_errors.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {long_prs_errors.mean():.1f}')
ax8.set_xlabel('Absolute Error (minutes)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax8.set_title('Error Distribution - Long PRs', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(alpha=0.3)

# Plot 9: Model Performance Comparison
ax9 = fig.add_subplot(gs[2, 2])
models = ['Normal\nModel\n(Normal PRs)', 'Normal\nModel\n(Long PRs)', 
          'Long\nModel\n(Long PRs)', 'Ensemble\n(All PRs)']
r2_scores = [normal_r2, normal_on_long_r2, long_r2, ensemble_r2]
colors_bar = ['steelblue', 'orange', 'forestgreen', 'purple']

bars = ax9.bar(models, r2_scores, color=colors_bar, alpha=0.8, edgecolor='black')
ax9.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax9.set_title('Model Performance Comparison (R²)', fontsize=12, fontweight='bold')
ax9.set_ylim([0, max(r2_scores) * 1.1])
ax9.grid(axis='y', alpha=0.3)

for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.savefig('model_ensemble_v3_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'model_ensemble_v3_comparison.png'")

# ============================================================================
# SAČUVAJ REZULTATE
# ============================================================================
print("\n" + "="*70)
print("SAČUVAVANJE REZULTATA")
print("="*70)

# Detaljne predikcije
predictions_df = comparison_df.copy()
predictions_df['model_used'] = ['Normal' if not is_long else 'Long' 
                                 for is_long in predictions_df['is_long']]
predictions_df.to_csv('model_ensemble_v3_predictions.csv', index=False)
print("✅ Saved 'model_ensemble_v3_predictions.csv'")

# Metričke usporedbe
metrics_comparison = pd.DataFrame({
    'model': ['Normal Model (Normal PRs)', 'Normal Model (Long PRs)', 
              'Long Model (Long PRs)', 'Ensemble (All PRs)'],
    'r2': [normal_r2, normal_on_long_r2, long_r2, ensemble_r2],
    'rmse': [normal_rmse, normal_on_long_rmse, long_rmse, ensemble_rmse],
    'mae': [normal_mae, normal_on_long_mae, long_mae, ensemble_mae],
    'n_samples': [len(y_test_normal), len(y_test_long), 
                  len(y_test_long), len(y_test)]
})

metrics_comparison.to_csv('model_ensemble_v3_metrics.csv', index=False)
print("✅ Saved 'model_ensemble_v3_metrics.csv'")

# Segment statistike
segment_comparison.to_csv('model_ensemble_v3_segment_stats.csv')
print("✅ Saved 'model_ensemble_v3_segment_stats.csv'")

# Detaljna usporedba za velike PR-ove
long_prs_comparison = pd.DataFrame({
    'actual': y_test_long.values,
    'predicted_normal_model': y_pred_normal_on_long,
    'predicted_long_model': y_pred_long,
    'error_normal': y_test_long.values - y_pred_normal_on_long,
    'error_long': y_test_long.values - y_pred_long,
    'abs_error_normal': np.abs(y_test_long.values - y_pred_normal_on_long),
    'abs_error_long': np.abs(y_test_long.values - y_pred_long),
    'improvement': np.abs(y_test_long.values - y_pred_normal_on_long) - 
                  np.abs(y_test_long.values - y_pred_long)
})

long_prs_comparison.to_csv('model_ensemble_v3_long_prs_comparison.csv', index=False)
print("✅ Saved 'model_ensemble_v3_long_prs_comparison.csv'")

print("\n" + "="*70)
print("✅ ENSEMBLE MODEL V3 ZAVRŠEN!")
print("="*70)
print(f"\nSažetak:")
print(f"  Threshold za velike PR-ove: {long_threshold:.2f} minutes")
print(f"\nNormal Model Performance:")
print(f"  - Na normalnim PR-ovima: R²={normal_r2:.4f}, RMSE={normal_rmse:.1f}, MAE={normal_mae:.1f}")
print(f"  - Na velikim PR-ovima: R²={normal_on_long_r2:.4f}, RMSE={normal_on_long_rmse:.1f}, MAE={normal_on_long_mae:.1f}")
print(f"\nLong Model Performance:")
print(f"  - Na velikim PR-ovima: R²={long_r2:.4f}, RMSE={long_rmse:.1f}, MAE={long_mae:.1f}")
print(f"\nEnsemble Model Performance:")
print(f"  - Na svim PR-ovima: R²={ensemble_r2:.4f}, RMSE={ensemble_rmse:.1f}, MAE={ensemble_mae:.1f}")
print(f"\nPoboljšanje Long Modela nad Normal Modelom (na velikim PR-ovima):")
print(f"  R²: {improvement_r2:+.4f} ({improvement_r2/normal_on_long_r2*100:+.2f}%)")
print(f"  RMSE: {improvement_rmse:+.2f} minutes ({improvement_rmse/normal_on_long_rmse*100:+.2f}%)")
print(f"  MAE: {improvement_mae:+.2f} minutes ({improvement_mae/normal_on_long_mae*100:+.2f}%)")
