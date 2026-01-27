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
print("MODEL POBOLJŠANJA - PRIMJENA ZAKLJUČAKA IZ ANALIZE")
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

# ============================================================================
# POBOLJŠANJE 1: Sample Weighting za Long Segment
# ============================================================================
print("\n" + "="*70)
print("POBOLJŠANJE 1: SAMPLE WEIGHTING ZA LONG SEGMENT")
print("="*70)

# Definiraj Long segment threshold (75% kvantil)
long_threshold = y_train.quantile(0.75)
print(f"Long segment threshold: {long_threshold:.2f} minutes")

# Kreiraj sample weights - daj više težine Long segmentu
sample_weights = np.ones(len(y_train))
sample_weights[y_train > long_threshold] = 3.0  # 3x veća težina za Long segment

long_samples = (y_train > long_threshold).sum()
print(f"Long segment samples: {long_samples} ({long_samples/len(y_train)*100:.1f}%)")
print(f"Weight distribution: Normal=1.0, Long={sample_weights[y_train > long_threshold][0]:.1f}")

# Treniraj model sa sample weights
model_weighted = xgb.XGBRegressor(
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

print("\nTraining weighted model...")
model_weighted.fit(X_train, y_train_log, sample_weight=sample_weights)

y_pred_weighted_log = model_weighted.predict(X_test)
y_pred_weighted = np.expm1(y_pred_weighted_log)

test_r2_weighted = r2_score(y_test, y_pred_weighted)
test_rmse_weighted = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
test_mae_weighted = mean_absolute_error(y_test, y_pred_weighted)

print(f"\nWeighted Model Performance:")
print(f"  Test R²: {test_r2_weighted:.4f}")
print(f"  Test RMSE: {test_rmse_weighted:.2f} minutes")
print(f"  Test MAE: {test_mae_weighted:.2f} minutes")

# ============================================================================
# POBOLJŠANJE 2: Baseline Model (za poređenje)
# ============================================================================
print("\n" + "="*70)
print("BASELINE MODEL (za poređenje)")
print("="*70)

model_baseline = xgb.XGBRegressor(
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

model_baseline.fit(X_train, y_train_log)
y_pred_baseline_log = model_baseline.predict(X_test)
y_pred_baseline = np.expm1(y_pred_baseline_log)

test_r2_baseline = r2_score(y_test, y_pred_baseline)
test_rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
test_mae_baseline = mean_absolute_error(y_test, y_pred_baseline)

print(f"\nBaseline Model Performance:")
print(f"  Test R²: {test_r2_baseline:.4f}")
print(f"  Test RMSE: {test_rmse_baseline:.2f} minutes")
print(f"  Test MAE: {test_mae_baseline:.2f} minutes")

# ============================================================================
# POBOLJŠANJE 3: Analiza grešaka po segmentima
# ============================================================================
print("\n" + "="*70)
print("ANALIZA GREŠAKA PO SEGMENTIMA")
print("="*70)

# Definiraj segmente
q25 = y_test.quantile(0.25)
median = y_test.median()
q75 = y_test.quantile(0.75)

y_test_segments = pd.cut(y_test, 
                        bins=[0, q25, median, q75, float('inf')],
                        labels=['Very Short', 'Short', 'Medium', 'Long'])

# Baseline errors
baseline_errors = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred_baseline,
    'error': y_test.values - y_pred_baseline,
    'abs_error': np.abs(y_test.values - y_pred_baseline),
    'segment': y_test_segments
})

# Weighted errors
weighted_errors = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred_weighted,
    'error': y_test.values - y_pred_weighted,
    'abs_error': np.abs(y_test.values - y_pred_weighted),
    'segment': y_test_segments
})

print("\nBaseline Model - Error by Segment:")
baseline_segment_stats = baseline_errors.groupby('segment').agg({
    'error': ['mean', 'std'],
    'abs_error': ['mean', 'median']
}).round(2)
print(baseline_segment_stats)

print("\nWeighted Model - Error by Segment:")
weighted_segment_stats = weighted_errors.groupby('segment').agg({
    'error': ['mean', 'std'],
    'abs_error': ['mean', 'median']
}).round(2)
print(weighted_segment_stats)

# Poređenje
print("\n" + "="*70)
print("POREĐENJE REZULTATA")
print("="*70)

comparison = pd.DataFrame({
    'Metric': ['R²', 'RMSE', 'MAE'],
    'Baseline': [test_r2_baseline, test_rmse_baseline, test_mae_baseline],
    'Weighted': [test_r2_weighted, test_rmse_weighted, test_mae_weighted],
    'Improvement': [
        test_r2_weighted - test_r2_baseline,
        test_rmse_baseline - test_rmse_weighted,  # RMSE improvement (lower is better)
        test_mae_baseline - test_mae_weighted    # MAE improvement (lower is better)
    ]
})

comparison['Improvement_%'] = [
    (test_r2_weighted - test_r2_baseline) / test_r2_baseline * 100,
    (test_rmse_baseline - test_rmse_weighted) / test_rmse_baseline * 100,
    (test_mae_baseline - test_mae_weighted) / test_mae_baseline * 100
]

print("\nOverall Performance Comparison:")
print(comparison.to_string(index=False))

# Long segment specific comparison
long_mask = y_test_segments == 'Long'
if long_mask.sum() > 0:
    baseline_long_mae = baseline_errors[long_mask]['abs_error'].mean()
    weighted_long_mae = weighted_errors[long_mask]['abs_error'].mean()
    long_improvement = (baseline_long_mae - weighted_long_mae) / baseline_long_mae * 100
    
    print(f"\nLong Segment Specific:")
    print(f"  Baseline MAE: {baseline_long_mae:.2f} minutes")
    print(f"  Weighted MAE: {weighted_long_mae:.2f} minutes")
    print(f"  Improvement: {long_improvement:.2f}%")

# ============================================================================
# Vizualizacija
# ============================================================================
print("\n" + "="*70)
print("KREIRANJE VIZUALIZACIJA")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Model Comparison: Baseline vs Weighted', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Overall performance comparison
ax1 = axes[0, 0]
metrics = ['R²', 'RMSE', 'MAE']
baseline_vals = [test_r2_baseline, test_rmse_baseline/1000, test_mae_baseline/1000]  # Scale for visibility
weighted_vals = [test_r2_weighted, test_rmse_weighted/1000, test_mae_weighted/1000]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, weighted_vals, width, label='Weighted', color='forestgreen', alpha=0.8)

ax1.set_ylabel('Score Value', fontsize=12, fontweight='bold')
ax1.set_title('Overall Performance Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, max(max(baseline_vals), max(weighted_vals)) * 1.2])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Error by segment comparison
ax2 = axes[0, 1]
segment_order = ['Very Short', 'Short', 'Medium', 'Long']
baseline_mae_by_seg = baseline_errors.groupby('segment')['abs_error'].mean()
weighted_mae_by_seg = weighted_errors.groupby('segment')['abs_error'].mean()

x = np.arange(len(segment_order))
bars1 = ax2.bar(x - width/2, [baseline_mae_by_seg.get(seg, 0) for seg in segment_order], 
                width, label='Baseline', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x + width/2, [weighted_mae_by_seg.get(seg, 0) for seg in segment_order], 
                width, label='Weighted', color='forestgreen', alpha=0.8)

ax2.set_xlabel('Segment', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Absolute Error (minutes)', fontsize=12, fontweight='bold')
ax2.set_title('MAE by Segment', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(segment_order, fontsize=10, rotation=45, ha='right')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Actual vs Predicted - Baseline
ax3 = axes[1, 0]
ax3.scatter(y_test.values, y_pred_baseline, alpha=0.6, color='steelblue', s=50)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual (minutes)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Predicted (minutes)', fontsize=12, fontweight='bold')
ax3.set_title(f'Baseline Model (R²={test_r2_baseline:.4f})', fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Actual vs Predicted - Weighted
ax4 = axes[1, 1]
ax4.scatter(y_test.values, y_pred_weighted, alpha=0.6, color='forestgreen', s=50)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax4.set_xlabel('Actual (minutes)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Predicted (minutes)', fontsize=12, fontweight='bold')
ax4.set_title(f'Weighted Model (R²={test_r2_weighted:.4f})', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('model_poboljsanja_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'model_poboljsanja_comparison.png'")

# Save results
comparison.to_csv('model_poboljsanja_comparison.csv', index=False)
print("✅ Saved 'model_poboljsanja_comparison.csv'")

baseline_segment_stats.to_csv('baseline_segment_stats.csv')
weighted_segment_stats.to_csv('weighted_segment_stats.csv')
print("✅ Saved segment statistics")

print("\n" + "="*70)
print("✅ ANALIZA ZAVRŠENA!")
print("="*70)
