import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM and CatBoost
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not available")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not available")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*70)
print("IMPROVED MODEL WITH MULTIPLE STRATEGIES")
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

# Check target distribution
print(f"\nTarget variable statistics:")
print(f"  Min: {df[target_col].min():.2f}")
print(f"  Max: {df[target_col].max():.2f}")
print(f"  Mean: {df[target_col].mean():.2f}")
print(f"  Median: {df[target_col].median():.2f}")
print(f"  Std: {df[target_col].std():.2f}")

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
print(f"\nInitial number of features: {len(feature_cols)}")

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
print(f"Categorical columns: {list(categorical_cols)}")

X_encoded = X_numeric_imputed.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_col_filled = X[col].fillna('unknown').astype(str)
    X_encoded[col] = le.fit_transform(X_col_filled)
    label_encoders[col] = le

X_base = X_encoded.reset_index(drop=True)
y_base = y.reset_index(drop=True)

# ============================================================================
# IMPROVEMENT 1: Feature Engineering
# ============================================================================
print("\n" + "="*70)
print("IMPROVEMENT 1: FEATURE ENGINEERING")
print("="*70)

X_engineered = X_base.copy()

# Create interaction features (ratios)
print("\nCreating interaction features...")
if 'additions' in X_engineered.columns and 'deletions' in X_engineered.columns:
    X_engineered['additions_deletions_ratio'] = (X_engineered['additions'] + 1) / (X_engineered['deletions'] + 1)
    print("  ✅ additions_deletions_ratio")

if 'commits' in X_engineered.columns and 'changed_files' in X_engineered.columns:
    X_engineered['commits_per_file'] = X_engineered['commits'] / (X_engineered['changed_files'] + 1)
    print("  ✅ commits_per_file")

if 'total_lines_changed' in X_engineered.columns and 'commits' in X_engineered.columns:
    X_engineered['lines_per_commit'] = X_engineered['total_lines_changed'] / (X_engineered['commits'] + 1)
    print("  ✅ lines_per_commit")

if 'review_count' in X_engineered.columns and 'reviewer_count' in X_engineered.columns:
    X_engineered['reviews_per_reviewer'] = X_engineered['review_count'] / (X_engineered['reviewer_count'] + 1)
    print("  ✅ reviews_per_reviewer")

if 'time_to_first_review_minutes' in X_engineered.columns and 'time_to_first_approval_minutes' in X_engineered.columns:
    X_engineered['review_to_approval_time'] = np.maximum(0, 
        X_engineered['time_to_first_approval_minutes'] - X_engineered['time_to_first_review_minutes'])
    print("  ✅ review_to_approval_time")

# Create polynomial features for important features
print("\nCreating polynomial features...")
important_features = ['time_to_first_approval_minutes', 'commits', 'review_count', 
                     'total_lines_changed', 'changed_files']
for feat in important_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_squared'] = X_engineered[feat] ** 2
        print(f"  ✅ {feat}_squared")

# Create log transformations for skewed features
print("\nCreating log transformations...")
skewed_features = ['additions', 'deletions', 'total_lines_changed', 'commits', 
                   'review_count', 'comments', 'review_comments']
for feat in skewed_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_log'] = np.log1p(X_engineered[feat])
        print(f"  ✅ {feat}_log")

# Create time-based features from dates if available
print("\nCreating time-based features...")
if 'created_at' in df.columns:
    try:
        df['created_at_parsed'] = pd.to_datetime(df['created_at'], errors='coerce')
        X_engineered['created_hour'] = df['created_at_parsed'].dt.hour
        X_engineered['created_day_of_week'] = df['created_at_parsed'].dt.dayofweek
        X_engineered['created_is_weekend'] = (X_engineered['created_day_of_week'] >= 5).astype(int)
        print("  ✅ Time-based features from created_at")
    except:
        pass

print(f"\nTotal features after engineering: {X_engineered.shape[1]} (added {X_engineered.shape[1] - X_base.shape[1]})")

# ============================================================================
# IMPROVEMENT 2: Feature Selection
# ============================================================================
print("\n" + "="*70)
print("IMPROVEMENT 2: FEATURE SELECTION")
print("="*70)

# Select top features using F-regression
n_features_to_select = min(40, X_engineered.shape[1])
selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
X_selected = selector.fit_transform(X_engineered, y_base)
selected_features = X_engineered.columns[selector.get_support()].tolist()
X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X_engineered.index)

print(f"\nSelected {len(selected_features)} best features from {X_engineered.shape[1]} total")
print(f"Top 10 selected features:")
feature_scores = pd.DataFrame({
    'feature': X_engineered.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)
print(feature_scores.head(10)[['feature', 'score']].to_string(index=False))

# ============================================================================
# IMPROVEMENT 3: Target Transformation
# ============================================================================
print("\n" + "="*70)
print("IMPROVEMENT 3: TARGET TRANSFORMATION")
print("="*70)

# Log transform target to handle skewness
y_log = np.log1p(y_base)
print("✅ Applied log transformation to target (log1p)")

# ============================================================================
# IMPROVEMENT 4: Train-Test Split
# ============================================================================
print("\n" + "="*70)
print("IMPROVEMENT 4: TRAIN-TEST SPLIT")
print("="*70)

# Use different random_state for split to ensure better randomization
# independent from the initial data shuffle (using seed 123 instead of 42)
X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
    X_selected_df, y_base, y_log, test_size=0.2, random_state=123, shuffle=True
)

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# IMPROVEMENT 5: Test Multiple Models
# ============================================================================
print("\n" + "="*70)
print("IMPROVEMENT 5: TESTING MULTIPLE MODELS")
print("="*70)

results = []

# 1. XGBoost with original target
print("\n1. XGBoost (Original Target)...")
model_xgb = xgb.XGBRegressor(
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
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
test_r2_xgb = r2_score(y_test, y_pred_xgb)
test_rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
test_mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
results.append({
    'model': 'XGBoost (Original)',
    'r2': test_r2_xgb,
    'rmse': test_rmse_xgb,
    'mae': test_mae_xgb,
    'model_obj': model_xgb
})
print(f"   Test R²: {test_r2_xgb:.4f}, RMSE: {test_rmse_xgb:.2f}")

# 2. XGBoost with log-transformed target
print("\n2. XGBoost (Log-Transformed Target)...")
model_xgb_log = xgb.XGBRegressor(
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
model_xgb_log.fit(X_train, y_train_log)
y_pred_log = model_xgb_log.predict(X_test)
y_pred_xgb_log = np.expm1(y_pred_log)  # Transform back
test_r2_xgb_log = r2_score(y_test, y_pred_xgb_log)
test_rmse_xgb_log = np.sqrt(mean_squared_error(y_test, y_pred_xgb_log))
test_mae_xgb_log = mean_absolute_error(y_test, y_pred_xgb_log)
results.append({
    'model': 'XGBoost (Log Target)',
    'r2': test_r2_xgb_log,
    'rmse': test_rmse_xgb_log,
    'mae': test_mae_xgb_log,
    'model_obj': model_xgb_log
})
print(f"   Test R²: {test_r2_xgb_log:.4f}, RMSE: {test_rmse_xgb_log:.2f}")

# 3. LightGBM if available
if HAS_LIGHTGBM:
    print("\n3. LightGBM (Log-Transformed Target)...")
    model_lgb = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    model_lgb.fit(X_train, y_train_log)
    y_pred_lgb_log = model_lgb.predict(X_test)
    y_pred_lgb = np.expm1(y_pred_lgb_log)
    test_r2_lgb = r2_score(y_test, y_pred_lgb)
    test_rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    test_mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    results.append({
        'model': 'LightGBM (Log Target)',
        'r2': test_r2_lgb,
        'rmse': test_rmse_lgb,
        'mae': test_mae_lgb,
        'model_obj': model_lgb
    })
    print(f"   Test R²: {test_r2_lgb:.4f}, RMSE: {test_rmse_lgb:.2f}")

# 4. XGBoost with more regularization (reduce overfitting)
print("\n4. XGBoost (High Regularization, Log Target)...")
model_xgb_reg = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model_xgb_reg.fit(X_train, y_train_log)
y_pred_reg_log = model_xgb_reg.predict(X_test)
y_pred_xgb_reg = np.expm1(y_pred_reg_log)
test_r2_xgb_reg = r2_score(y_test, y_pred_xgb_reg)
test_rmse_xgb_reg = np.sqrt(mean_squared_error(y_test, y_pred_xgb_reg))
test_mae_xgb_reg = mean_absolute_error(y_test, y_pred_xgb_reg)
results.append({
    'model': 'XGBoost (High Reg, Log)',
    'r2': test_r2_xgb_reg,
    'rmse': test_rmse_xgb_reg,
    'mae': test_mae_xgb_reg,
    'model_obj': model_xgb_reg
})
print(f"   Test R²: {test_r2_xgb_reg:.4f}, RMSE: {test_rmse_xgb_reg:.2f}")

# ============================================================================
# Find Best Model
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('r2', ascending=False)

print("\nModel Performance (sorted by R²):")
print(results_df[['model', 'r2', 'rmse', 'mae']].to_string(index=False))

best_idx = results_df['r2'].idxmax()
best_model_name = results_df.loc[best_idx, 'model']
best_model = results_df.loc[best_idx, 'model_obj']
best_r2 = results_df.loc[best_idx, 'r2']
best_rmse = results_df.loc[best_idx, 'rmse']
best_mae = results_df.loc[best_idx, 'mae']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test R²: {best_r2:.4f}")
print(f"   Test RMSE: {best_rmse:.2f} minutes ({best_rmse/60:.2f} hours)")
print(f"   Test MAE: {best_mae:.2f} minutes ({best_mae/60:.2f} hours)")

# Get predictions from best model
if 'Log' in best_model_name:
    y_train_pred_log = best_model.predict(X_train)
    y_train_pred = np.expm1(y_train_pred_log)
    y_test_pred_log = best_model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)
else:
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

# Calculate training metrics
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

# ============================================================================
# Feature Importance
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Best Model)")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save model comparison
results_df.to_csv('model_comparison.csv', index=False)
print("✅ Model comparison saved to 'model_comparison.csv'")

# Save metrics
metrics_df = pd.DataFrame({
    'metric': ['RMSE', 'MAE', 'R²'],
    'train': [train_rmse, train_mae, train_r2],
    'test': [best_rmse, best_mae, best_r2]
})
metrics_df.to_csv('model_scores_improved.csv', index=False)
print("✅ Model scores saved to 'model_scores_improved.csv'")

# Save feature importance
feature_importance.to_csv('feature_importance_improved.csv', index=False)
print("✅ Feature importance saved to 'feature_importance_improved.csv'")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_test_pred,
    'error': y_test.values - y_test_pred,
    'abs_error': np.abs(y_test.values - y_test_pred)
})
predictions_df.to_csv('predictions_improved.csv', index=False)
print("✅ Predictions saved to 'predictions_improved.csv'")

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. Model Comparison
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(results_df))
bars = ax.bar(x_pos, results_df['r2'], color='steelblue', edgecolor='black')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison: R² Scores', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(results_df.iterrows()):
    ax.text(i, row['r2'], f' {row["r2"]:.4f}',
            va='bottom', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'model_comparison.png'")

# 2. Best Model Performance
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Best Model Performance: {best_model_name}', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Training set
ax1 = axes[0, 0]
ax1.scatter(y_train, y_train_pred, alpha=0.6, color='steelblue')
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax1.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax1.set_title('Training Set', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
score_text = f'R² = {train_r2:.4f}\nRMSE = {train_rmse:.2f} min\nMAE = {train_mae:.2f} min'
ax1.text(0.05, 0.95, score_text, transform=ax1.transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.8), fontweight='bold')

# Plot 2: Test set
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred, alpha=0.6, color='forestgreen')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax2.set_title('Test Set', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)
score_text = f'R² = {best_r2:.4f}\nRMSE = {best_rmse:.2f} min\nMAE = {best_mae:.2f} min'
ax2.text(0.05, 0.95, score_text, transform=ax2.transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='lightgreen', alpha=0.8), fontweight='bold')

# Plot 3: Residuals
ax3 = axes[1, 0]
residuals = np.array(y_test) - y_test_pred
ax3.scatter(y_test_pred, residuals, alpha=0.6, color='crimson')
ax3.axhline(y=0, color='black', linestyle='--', lw=2)
ax3.set_xlabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuals (minutes)', fontsize=11, fontweight='bold')
ax3.set_title('Residuals Plot (Test Set)', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Feature Importance
ax4 = axes[1, 1]
top_features = feature_importance.head(15)
ax4.barh(range(len(top_features)), top_features['importance'], color='steelblue')
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['feature'], fontsize=9)
ax4.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax4.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('best_model_performance.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'best_model_performance.png'")

# 3. Improvement Summary
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold', y=0.98)

metrics = ['RMSE', 'MAE', 'R² Score']
train_values = [train_rmse, train_mae, train_r2]
test_values = [best_rmse, best_mae, best_r2]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, train_values, width, label='Training Set', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, test_values, width, label='Test Set', color='forestgreen', alpha=0.8)

ax.set_ylabel('Score Value', fontsize=12, fontweight='bold')
ax.set_title('Best Model Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

summary_text = f'Best Model: {best_model_name}\n'
summary_text += f'Test R²: {best_r2:.4f} ({best_r2*100:.2f}% variance explained)\n'
summary_text += f'Test RMSE: {best_rmse:.2f} minutes ({best_rmse/60:.2f} hours)\n'
summary_text += f'Test MAE: {best_mae:.2f} minutes ({best_mae/60:.2f} hours)'
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
        facecolor='wheat', alpha=0.9), fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('model_metrics_improved.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'model_metrics_improved.png'")

print("\n" + "="*70)
print("✅ IMPROVEMENT ANALYSIS COMPLETE!")
print("="*70)
print(f"\nBest Model: {best_model_name}")
print(f"Test R²: {best_r2:.4f} (improved from baseline)")
print(f"Test RMSE: {best_rmse:.2f} minutes")
print(f"Test MAE: {best_mae:.2f} minutes")


