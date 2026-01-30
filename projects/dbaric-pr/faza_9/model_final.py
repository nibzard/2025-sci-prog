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
print("FINAL MODEL - IMPLEMENTACIJA SVIH PREPORUKA IZ ANALIZE")
print("="*70)
print("\nImplementirane preporuke:")
print("  1. Post-processing calibration za Long segment")
print("  2. Quantile regression za heteroskedastične prediction intervals")
print("  3. Separate model za Long segment")
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

# Define Long segment threshold
long_threshold = y_train.quantile(0.75)
print(f"\nLong segment threshold: {long_threshold:.2f} minutes")

# ============================================================================
# STRATEGIJA 1: Main Model (za sve segmente)
# ============================================================================
print("\n" + "="*70)
print("STRATEGIJA 1: MAIN MODEL (za sve segmente)")
print("="*70)

model_main = xgb.XGBRegressor(
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

print("Training main model...")
model_main.fit(X_train, y_train_log)
y_pred_main_log = model_main.predict(X_test)
y_pred_main = np.expm1(y_pred_main_log)

# ============================================================================
# STRATEGIJA 2: Separate Model za Long Segment
# ============================================================================
print("\n" + "="*70)
print("STRATEGIJA 2: SEPARATE MODEL ZA LONG SEGMENT")
print("="*70)

# Identifikuj Long segment u training setu
train_long_mask = y_train > long_threshold
X_train_long = X_train[train_long_mask]
y_train_long_log = y_train_log[train_long_mask]

print(f"Long segment training samples: {len(X_train_long)} ({len(X_train_long)/len(X_train)*100:.1f}%)")

if len(X_train_long) > 5:  # Treba dovoljno podataka
    model_long = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,  # Manje regularizacije za Long segment
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    print("Training Long segment model...")
    model_long.fit(X_train_long, y_train_long_log)
    use_separate_long_model = True
else:
    print("⚠️  Not enough Long segment samples, skipping separate model")
    use_separate_long_model = False

# ============================================================================
# STRATEGIJA 3: Quantile Regression za Prediction Intervals
# ============================================================================
print("\n" + "="*70)
print("STRATEGIJA 3: QUANTILE REGRESSION ZA PREDICTION INTERVALS")
print("="*70)

# Treniraj quantile regression modele za različite kvantile
print("Training quantile regression models...")

# XGBoost quantile regression
model_q05 = xgb.XGBRegressor(
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
    verbosity=0,
    objective='reg:quantileerror',
    quantile_alpha=0.05
)

model_q50 = xgb.XGBRegressor(
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
    verbosity=0,
    objective='reg:quantileerror',
    quantile_alpha=0.50
)

model_q95 = xgb.XGBRegressor(
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
    verbosity=0,
    objective='reg:quantileerror',
    quantile_alpha=0.95
)

model_q05.fit(X_train, y_train_log)
model_q50.fit(X_train, y_train_log)
model_q95.fit(X_train, y_train_log)

# Predikcije za kvantile
y_pred_q05_log = model_q05.predict(X_test)
y_pred_q50_log = model_q50.predict(X_test)
y_pred_q95_log = model_q95.predict(X_test)

# Transform back to original scale
y_pred_q05 = np.expm1(y_pred_q05_log)
y_pred_q50 = np.expm1(y_pred_q50_log)
y_pred_q95 = np.expm1(y_pred_q95_log)

print("✅ Quantile regression models trained")

# ============================================================================
# KOMBINACIJA: Finalne predikcije
# ============================================================================
print("\n" + "="*70)
print("KOMBINACIJA: FINALNE PREDIKCIJE")
print("="*70)

# Identifikuj Long segment u test setu
test_long_mask = y_test > long_threshold

# Za Long segment: koristi separate model ako postoji, inače main model
y_pred_final = y_pred_main.copy()

if use_separate_long_model and test_long_mask.sum() > 0:
    print(f"Using separate Long segment model for {test_long_mask.sum()} test samples")
    y_pred_long_log = model_long.predict(X_test[test_long_mask])
    y_pred_final[test_long_mask] = np.expm1(y_pred_long_log)
else:
    print("Using main model for all segments")

# ============================================================================
# POBOLJŠANJE 1: Post-Processing Calibration za Long Segment
# ============================================================================
print("\n" + "="*70)
print("POBOLJŠANJE 1: POST-PROCESSING CALIBRATION")
print("="*70)

# Izračunaj optimalni calibration factor na osnovu training seta
train_long_mask_cal = y_train > long_threshold
if train_long_mask_cal.sum() > 0:
    y_train_pred_main_log = model_main.predict(X_train[train_long_mask_cal])
    y_train_pred_main = np.expm1(y_train_pred_main_log)
    y_train_actual_long = y_train[train_long_mask_cal].values
    
    # Izračunaj prosječni bias
    bias = (y_train_actual_long - y_train_pred_main).mean()
    ratio = y_train_actual_long.mean() / y_train_pred_main.mean()
    
    print(f"Training Long segment bias: {bias:.2f} minutes")
    print(f"Training Long segment ratio: {ratio:.4f}")
    
    # Koristi ratio za calibration
    calibration_factor = ratio
    print(f"Calibration factor: {calibration_factor:.4f}")
    
    # Primijeni calibration
    y_pred_calibrated = y_pred_final.copy()
    y_pred_calibrated[test_long_mask] = y_pred_final[test_long_mask] * calibration_factor
    
    print(f"✅ Applied calibration to {test_long_mask.sum()} Long segment predictions")
else:
    y_pred_calibrated = y_pred_final.copy()
    calibration_factor = 1.0
    print("⚠️  No Long segment samples for calibration")

# ============================================================================
# POBOLJŠANJE 2: Dynamic Prediction Intervals
# ============================================================================
print("\n" + "="*70)
print("POBOLJŠANJE 2: DYNAMIC PREDICTION INTERVALS")
print("="*70)

# Koristi quantile regression za dinamičke intervale
y_pred_lower = y_pred_q05.copy()
y_pred_upper = y_pred_q95.copy()

# Također primijeni calibration na intervale za Long segment
if test_long_mask.sum() > 0:
    y_pred_lower[test_long_mask] = y_pred_lower[test_long_mask] * calibration_factor
    y_pred_upper[test_long_mask] = y_pred_upper[test_long_mask] * calibration_factor

# Izračunaj interval width
interval_widths = y_pred_upper - y_pred_lower

print(f"Mean interval width: {interval_widths.mean():.2f} minutes")
print(f"Min interval width: {interval_widths.min():.2f} minutes")
print(f"Max interval width: {interval_widths.max():.2f} minutes")
print(f"Std interval width: {interval_widths.std():.2f} minutes")

# ============================================================================
# EVALUACIJA
# ============================================================================
print("\n" + "="*70)
print("EVALUACIJA FINALNOG MODELA")
print("="*70)

# Baseline (main model bez poboljšanja)
baseline_r2 = r2_score(y_test, y_pred_main)
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_main))
baseline_mae = mean_absolute_error(y_test, y_pred_main)

# Final model (sa svim poboljšanjima)
final_r2 = r2_score(y_test, y_pred_calibrated)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_calibrated))
final_mae = mean_absolute_error(y_test, y_pred_calibrated)

print("\nBaseline Model (Main only):")
print(f"  R²: {baseline_r2:.4f}")
print(f"  RMSE: {baseline_rmse:.2f} minutes")
print(f"  MAE: {baseline_mae:.2f} minutes")

print("\nFinal Model (with all improvements):")
print(f"  R²: {final_r2:.4f}")
print(f"  RMSE: {final_rmse:.2f} minutes")
print(f"  MAE: {final_mae:.2f} minutes")

improvement_r2 = final_r2 - baseline_r2
improvement_rmse = baseline_rmse - final_rmse
improvement_mae = baseline_mae - final_mae

print("\nImprovement:")
print(f"  R²: {improvement_r2:+.4f} ({improvement_r2/baseline_r2*100:+.2f}%)")
print(f"  RMSE: {improvement_rmse:+.2f} minutes ({improvement_rmse/baseline_rmse*100:+.2f}%)")
print(f"  MAE: {improvement_mae:+.2f} minutes ({improvement_mae/baseline_mae*100:+.2f}%)")

# Analiza po segmentima
print("\n" + "="*70)
print("ANALIZA PO SEGMENTIMA")
print("="*70)

q25 = y_test.quantile(0.25)
median = y_test.median()
q75 = y_test.quantile(0.75)

y_test_segments = pd.cut(y_test, 
                        bins=[0, q25, median, q75, float('inf')],
                        labels=['Very Short', 'Short', 'Medium', 'Long'])

final_errors = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred_calibrated,
    'error': y_test.values - y_pred_calibrated,
    'abs_error': np.abs(y_test.values - y_pred_calibrated),
    'segment': y_test_segments
})

segment_stats = final_errors.groupby('segment').agg({
    'error': ['mean', 'std'],
    'abs_error': ['mean', 'median']
}).round(2)

print("\nFinal Model - Error by Segment:")
print(segment_stats)

# Prediction interval coverage
within_interval = ((y_test.values >= y_pred_lower) & (y_test.values <= y_pred_upper)).sum()
coverage = (within_interval / len(y_test)) * 100

print(f"\nPrediction Interval Coverage (95%): {coverage:.2f}%")
print(f"Expected: 90% (5th to 95th percentile)")

# ============================================================================
# VIZUALIZACIJA
# ============================================================================
print("\n" + "="*70)
print("KREIRANJE VIZUALIZACIJA")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Final Model Performance - All Improvements Applied', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_test.values, y_pred_calibrated, alpha=0.6, color='forestgreen', s=50)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual (minutes)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted (minutes)', fontsize=12, fontweight='bold')
ax1.set_title(f'Actual vs Predicted (R²={final_r2:.4f})', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

score_text = f'R² = {final_r2:.4f}\nRMSE = {final_rmse:.2f} min\nMAE = {final_mae:.2f} min'
ax1.text(0.05, 0.95, score_text, transform=ax1.transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='lightgreen', alpha=0.8), fontweight='bold')

# Plot 2: Prediction Intervals
ax2 = axes[0, 1]
sorted_idx = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_idx]
y_pred_sorted = y_pred_calibrated[sorted_idx]
lower_sorted = y_pred_lower[sorted_idx]
upper_sorted = y_pred_upper[sorted_idx]

ax2.fill_between(range(len(y_test_sorted)), lower_sorted, upper_sorted, 
                 alpha=0.3, color='lightblue', label='95% Prediction Interval')
ax2.plot(range(len(y_test_sorted)), y_test_sorted, 'o', color='green', 
         markersize=6, label='Actual', alpha=0.7)
ax2.plot(range(len(y_test_sorted)), y_pred_sorted, 'o', color='red', 
         markersize=4, label='Predicted', alpha=0.7)
ax2.set_xlabel('Sample Index (sorted by actual)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Duration (minutes)', fontsize=12, fontweight='bold')
ax2.set_title(f'Predictions with Dynamic Intervals (Coverage: {coverage:.1f}%)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Error by Segment
ax3 = axes[1, 0]
segment_order = ['Very Short', 'Short', 'Medium', 'Long']
mae_by_seg = final_errors.groupby('segment')['abs_error'].mean()
bars = ax3.bar(segment_order, [mae_by_seg.get(seg, 0) for seg in segment_order], 
               color='steelblue', edgecolor='black', alpha=0.8)
ax3.set_xlabel('Segment', fontsize=12, fontweight='bold')
ax3.set_ylabel('Mean Absolute Error (minutes)', fontsize=12, fontweight='bold')
ax3.set_title('MAE by Segment', fontsize=13, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Interval Width vs Predicted Value
ax4 = axes[1, 1]
ax4.scatter(y_pred_calibrated, interval_widths, alpha=0.6, color='crimson', s=50)
ax4.set_xlabel('Predicted Value (minutes)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Interval Width (minutes)', fontsize=12, fontweight='bold')
ax4.set_title('Dynamic Interval Width vs Predicted Value', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)

# Add trend line
z = np.polyfit(y_pred_calibrated, interval_widths, 1)
p = np.poly1d(z)
ax4.plot(y_pred_calibrated, p(y_pred_calibrated), "r--", alpha=0.8, linewidth=2, label='Trend')

width_text = f'Mean Width: {interval_widths.mean():.2f} min\n'
width_text += f'Correlation: {np.corrcoef(y_pred_calibrated, interval_widths)[0,1]:.3f}'
ax4.text(0.05, 0.95, width_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.8), fontweight='bold')
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('model_final_performance.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'model_final_performance.png'")

# ============================================================================
# SAČUVAJ REZULTATE
# ============================================================================
print("\n" + "="*70)
print("SAČUVAVANJE REZULTATA")
print("="*70)

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted_baseline': y_pred_main,
    'predicted_final': y_pred_calibrated,
    'lower_95': y_pred_lower,
    'upper_95': y_pred_upper,
    'interval_width': interval_widths,
    'error': y_test.values - y_pred_calibrated,
    'abs_error': np.abs(y_test.values - y_pred_calibrated),
    'segment': y_test_segments,
    'within_interval': (y_test.values >= y_pred_lower) & (y_test.values <= y_pred_upper)
})

predictions_df.to_csv('model_final_predictions.csv', index=False)
print("✅ Saved 'model_final_predictions.csv'")

# Save metrics
metrics_df = pd.DataFrame({
    'model': ['Baseline', 'Final'],
    'r2': [baseline_r2, final_r2],
    'rmse': [baseline_rmse, final_rmse],
    'mae': [baseline_mae, final_mae]
})

metrics_df.to_csv('model_final_metrics.csv', index=False)
print("✅ Saved 'model_final_metrics.csv'")

# Save segment statistics
segment_stats.to_csv('model_final_segment_stats.csv')
print("✅ Saved 'model_final_segment_stats.csv'")

print("\n" + "="*70)
print("✅ FINALNI MODEL ZAVRŠEN!")
print("="*70)
print(f"\nImplementirane preporuke:")
print(f"  ✅ Post-processing calibration (factor: {calibration_factor:.4f})")
print(f"  ✅ Quantile regression za prediction intervals")
if use_separate_long_model:
    print(f"  ✅ Separate model za Long segment")
else:
    print(f"  ⚠️  Separate model za Long segment (nema dovoljno podataka)")
print(f"\nFinal Performance:")
print(f"  R²: {final_r2:.4f}")
print(f"  RMSE: {final_rmse:.2f} minutes")
print(f"  MAE: {final_mae:.2f} minutes")
print(f"  Prediction Interval Coverage: {coverage:.2f}%")
