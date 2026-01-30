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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print("="*70)
print("DETALJNA ANALIZA MODELA - ANALITIČKI GRAFOVI")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_csv('source.csv')
print(f"Dataset shape: {df.shape}")

# Target variable
target_col = 'effective_minutes'
print(f"\nTarget variable: {target_col}")

# Remove rows where target is missing
df = df[df[target_col].notna()].copy()
print(f"Samples after removing missing target: {len(df)}")

# Check for invalid target values
df = df[df[target_col] >= 0].copy()
print(f"Final samples: {len(df)}")

# Randomize data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Columns to exclude from features
exclude_cols = [
    'non_working_minutes',
    'pr_number', 'pr_id',
    'created_at', 'closed_at', 'merged_at', 'updated_at',
    'ready_for_review_time', 'workflow_start_time',
    'first_review_time', 'first_approval_time',
    'title', 'description', 'body',
    'author', 'merged_by_login',
    'task_id',
]

# Get feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]

# Prepare features and target
X = df[feature_cols].copy()
y = df[target_col].copy()

# Ensure indices are aligned
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Handle missing values
print("\nHandling missing values...")
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
print("\nFeature engineering...")
X_engineered = X_base.copy()

# Create interaction features
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

# Create polynomial features
important_features = ['time_to_first_approval_minutes', 'commits', 'review_count', 
                     'total_lines_changed', 'changed_files']
for feat in important_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_squared'] = X_engineered[feat] ** 2

# Create log transformations
skewed_features = ['additions', 'deletions', 'total_lines_changed', 'commits', 
                   'review_count', 'comments', 'review_comments']
for feat in skewed_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_log'] = np.log1p(X_engineered[feat])

# Create time-based features
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

# Train Best Model (XGBoost with log-transformed target)
print("\nTraining XGBoost model (Log-Transformed Target)...")
model = xgb.XGBRegressor(
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
model.fit(X_train, y_train_log)

# Make predictions
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Transform back to original scale
y_train_pred_log = model.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)

# Calculate residuals
residuals_log = y_test_log - y_pred_log
residuals_original = y_test - y_pred

# Calculate metrics
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.2f} minutes")
print(f"  Test MAE: {test_mae:.2f} minutes")

# ============================================================================
# 1. RESIDUAL PLOTS - LOG VS ORIGINAL SCALE
# ============================================================================
print("\n" + "="*70)
print("1. KREIRANJE RESIDUAL PLOTOVA (LOG VS ORIGINAL SKALA)")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Residual Analysis: Log Scale vs Original Scale', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Residuals on log scale vs predicted (log scale)
ax1 = axes[0, 0]
ax1.scatter(y_pred_log, residuals_log, alpha=0.6, color='steelblue', s=50)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted (log scale)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Residuals (log scale)', fontsize=12, fontweight='bold')
ax1.set_title('Residuals Plot - Log Scale', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

# Add statistics
res_log_mean = residuals_log.mean()
res_log_std = residuals_log.std()
res_log_text = f'Mean: {res_log_mean:.4f}\nStd: {res_log_std:.4f}\n'
res_log_text += f'Min: {residuals_log.min():.4f}\nMax: {residuals_log.max():.4f}'
ax1.text(0.05, 0.95, res_log_text, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='lightblue', alpha=0.8), fontweight='bold')

# Plot 2: Residuals on original scale vs predicted (original scale)
ax2 = axes[0, 1]
ax2.scatter(y_pred, residuals_original, alpha=0.6, color='forestgreen', s=50)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted (original scale, minutes)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals (original scale, minutes)', fontsize=12, fontweight='bold')
ax2.set_title('Residuals Plot - Original Scale', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

# Add statistics
res_orig_mean = residuals_original.mean()
res_orig_std = residuals_original.std()
res_orig_text = f'Mean: {res_orig_mean:.2f} min\nStd: {res_orig_std:.2f} min\n'
res_orig_text += f'Min: {residuals_original.min():.2f} min\nMax: {residuals_original.max():.2f} min'
ax2.text(0.05, 0.95, res_orig_text, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='lightgreen', alpha=0.8), fontweight='bold')

# Plot 3: Q-Q plot for residuals (log scale)
ax3 = axes[1, 0]
stats.probplot(residuals_log, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot - Residuals (Log Scale)', fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Q-Q plot for residuals (original scale)
ax4 = axes[1, 1]
stats.probplot(residuals_original, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot - Residuals (Original Scale)', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('residual_analysis_log_vs_original.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'residual_analysis_log_vs_original.png'")

# ============================================================================
# 2. ERROR BY SEGMENT (SHORT VS LONG DURATIONS)
# ============================================================================
print("\n" + "="*70)
print("2. ANALIZA GREŠAKA PO SEGMENTIMA (KRATKE VS DUGE TRAJANJA)")
print("="*70)

# Define segments based on actual duration
median_duration = y_test.median()
q25 = y_test.quantile(0.25)
q75 = y_test.quantile(0.75)

# Create segments
y_test_segments = pd.cut(y_test, 
                        bins=[0, q25, median_duration, q75, float('inf')],
                        labels=['Very Short', 'Short', 'Medium', 'Long'])

# Calculate errors by segment
segment_errors = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'error': residuals_original,
    'abs_error': np.abs(residuals_original),
    'segment': y_test_segments
})

segment_stats = segment_errors.groupby('segment').agg({
    'error': ['mean', 'std', 'count'],
    'abs_error': ['mean', 'median'],
    'actual': ['mean', 'median']
}).round(2)

print("\nError Statistics by Duration Segment:")
print(segment_stats)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Error Analysis by Duration Segments', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Mean Absolute Error by segment
ax1 = axes[0, 0]
segment_mae = segment_errors.groupby('segment')['abs_error'].mean()
segment_mae.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
ax1.set_xlabel('Duration Segment', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Absolute Error (minutes)', fontsize=12, fontweight='bold')
ax1.set_title('Mean Absolute Error by Segment', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Add value labels
for i, v in enumerate(segment_mae):
    ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Error distribution by segment (boxplot)
ax2 = axes[0, 1]
segment_order = ['Very Short', 'Short', 'Medium', 'Long']
segment_errors_filtered = segment_errors[segment_errors['segment'].notna()]
# Prepare data for boxplot
boxplot_data = [segment_errors_filtered[segment_errors_filtered['segment'] == seg]['error'].values 
                for seg in segment_order if seg in segment_errors_filtered['segment'].values]
boxplot_labels = [seg for seg in segment_order if seg in segment_errors_filtered['segment'].values]
bp = ax2.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax2.set_xlabel('Duration Segment', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error (minutes)', fontsize=12, fontweight='bold')
ax2.set_title('Error Distribution by Segment', fontsize=13, fontweight='bold')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(alpha=0.3)

# Plot 3: Actual vs Predicted by segment
ax3 = axes[1, 0]
colors = {'Very Short': 'green', 'Short': 'blue', 'Medium': 'orange', 'Long': 'red'}
for seg in segment_order:
    seg_data = segment_errors_filtered[segment_errors_filtered['segment'] == seg]
    if len(seg_data) > 0:
        ax3.scatter(seg_data['actual'], seg_data['predicted'], 
                   label=seg, alpha=0.6, s=60, color=colors.get(seg, 'gray'))
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'k--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Actual (minutes)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Predicted (minutes)', fontsize=12, fontweight='bold')
ax3.set_title('Actual vs Predicted by Segment', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Relative error by segment
ax4 = axes[1, 1]
segment_errors_filtered['relative_error'] = (segment_errors_filtered['error'] / 
                                              (segment_errors_filtered['actual'] + 1)) * 100
relative_error_by_segment = segment_errors_filtered.groupby('segment')['relative_error'].mean()
relative_error_by_segment.plot(kind='bar', ax=ax4, color='crimson', edgecolor='black')
ax4.set_xlabel('Duration Segment', fontsize=12, fontweight='bold')
ax4.set_ylabel('Mean Relative Error (%)', fontsize=12, fontweight='bold')
ax4.set_title('Mean Relative Error by Segment', fontsize=13, fontweight='bold')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.grid(axis='y', alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# Add value labels
for i, v in enumerate(relative_error_by_segment):
    ax4.text(i, v, f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('error_by_segment.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'error_by_segment.png'")

# ============================================================================
# 3. PREDICTION INTERVALS
# ============================================================================
print("\n" + "="*70)
print("3. PREDICTION INTERVALS (XGBOOST APPROXIMATION)")
print("="*70)

# Use XGBoost's built-in prediction intervals approximation
# We'll use quantile regression approach by training multiple models
# or use the standard deviation of residuals

# Method 1: Use residual standard deviation for prediction intervals
residual_std = residuals_original.std()
prediction_interval_lower = y_pred - 1.96 * residual_std  # 95% confidence interval
prediction_interval_upper = y_pred + 1.96 * residual_std

# Method 2: Use quantile regression (simpler approach - use percentiles of residuals)
percentile_95 = np.percentile(np.abs(residuals_original), 95)
percentile_90 = np.percentile(np.abs(residuals_original), 90)
percentile_75 = np.percentile(np.abs(residuals_original), 75)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Prediction Intervals Analysis', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Prediction intervals using standard deviation
ax1 = axes[0, 0]
sorted_idx = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]
lower_sorted = prediction_interval_lower[sorted_idx]
upper_sorted = prediction_interval_upper[sorted_idx]

ax1.fill_between(range(len(y_test_sorted)), lower_sorted, upper_sorted, 
                 alpha=0.3, color='lightblue', label='95% Prediction Interval')
ax1.plot(range(len(y_test_sorted)), y_test_sorted, 'o', color='green', 
         markersize=6, label='Actual', alpha=0.7)
ax1.plot(range(len(y_test_sorted)), y_pred_sorted, 'o', color='red', 
         markersize=4, label='Predicted', alpha=0.7)
ax1.set_xlabel('Sample Index (sorted by actual)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Duration (minutes)', fontsize=12, fontweight='bold')
ax1.set_title('Prediction Intervals (95% CI using Std Dev)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Coverage analysis - how many actual values fall within intervals
ax2 = axes[0, 1]
within_interval_95 = ((y_test.values >= prediction_interval_lower) & 
                      (y_test.values <= prediction_interval_upper)).sum()
coverage_95 = (within_interval_95 / len(y_test)) * 100

# Calculate for different confidence levels
coverage_data = []
for conf_level in [68, 80, 90, 95, 99]:
    z_score = {68: 1.0, 80: 1.28, 90: 1.645, 95: 1.96, 99: 2.58}[conf_level]
    lower = y_pred - z_score * residual_std
    upper = y_pred + z_score * residual_std
    within = ((y_test.values >= lower) & (y_test.values <= upper)).sum()
    coverage = (within / len(y_test)) * 100
    coverage_data.append({'Confidence Level': f'{conf_level}%', 'Coverage': coverage})

coverage_df = pd.DataFrame(coverage_data)
bars = ax2.bar(coverage_df['Confidence Level'], coverage_df['Coverage'], 
               color='steelblue', edgecolor='black')
ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Expected 95%')
ax2.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
ax2.set_ylabel('Actual Coverage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Interval Coverage', fontsize=13, fontweight='bold')
ax2.set_ylim([0, 100])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 3: Prediction intervals vs actual (scatter with intervals)
ax3 = axes[1, 0]
ax3.scatter(y_test.values, y_pred, alpha=0.6, color='steelblue', s=50, label='Predictions')
for i in range(len(y_test)):
    ax3.plot([y_test.values[i], y_test.values[i]], 
            [prediction_interval_lower[i], prediction_interval_upper[i]], 
            'gray', alpha=0.3, linewidth=1)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Actual (minutes)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Predicted (minutes)', fontsize=12, fontweight='bold')
ax3.set_title('Predictions with 95% Intervals', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Interval width analysis
ax4 = axes[1, 1]
interval_widths = prediction_interval_upper - prediction_interval_lower
ax4.scatter(y_pred, interval_widths, alpha=0.6, color='crimson', s=50)
ax4.set_xlabel('Predicted Value (minutes)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Interval Width (minutes)', fontsize=12, fontweight='bold')
ax4.set_title('Prediction Interval Width vs Predicted Value', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)

# Add statistics
width_mean = interval_widths.mean()
width_text = f'Mean Width: {width_mean:.2f} min\n'
width_text += f'Std Width: {interval_widths.std():.2f} min\n'
width_text += f'Min Width: {interval_widths.min():.2f} min\n'
width_text += f'Max Width: {interval_widths.max():.2f} min'
ax4.text(0.05, 0.95, width_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.8), fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('prediction_intervals.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'prediction_intervals.png'")

# Save detailed statistics
print("\n" + "="*70)
print("SAVING DETAILED STATISTICS")
print("="*70)

# Save segment analysis
segment_stats.to_csv('error_by_segment_stats.csv')
print("✅ Saved 'error_by_segment_stats.csv'")

# Save prediction intervals data
intervals_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'lower_95': prediction_interval_lower,
    'upper_95': prediction_interval_upper,
    'within_interval': (y_test.values >= prediction_interval_lower) & 
                      (y_test.values <= prediction_interval_upper),
    'interval_width': interval_widths
})
intervals_df.to_csv('prediction_intervals_data.csv', index=False)
print("✅ Saved 'prediction_intervals_data.csv'")

# Save coverage statistics
coverage_df.to_csv('prediction_interval_coverage.csv', index=False)
print("✅ Saved 'prediction_interval_coverage.csv'")

print("\n" + "="*70)
print("✅ ANALIZA ZAVRŠENA!")
print("="*70)
print(f"\nKreirani grafovi:")
print("  1. residual_analysis_log_vs_original.png - Residual plots na log i originalnoj skali")
print("  2. error_by_segment.png - Analiza grešaka po segmentima trajanja")
print("  3. prediction_intervals.png - Prediction intervals analiza")
print(f"\nKreirani CSV fajlovi:")
print("  1. error_by_segment_stats.csv - Statistike grešaka po segmentima")
print("  2. prediction_intervals_data.csv - Podaci o prediction intervalima")
print("  3. prediction_interval_coverage.csv - Coverage statistike")
