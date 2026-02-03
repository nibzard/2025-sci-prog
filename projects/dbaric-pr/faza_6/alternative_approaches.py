import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try importing LightGBM and CatBoost (optional)
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not available")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*70)
print("TESTING ALTERNATIVE ALGORITHMS & APPROACHES")
print("="*70)

# Load and prepare data (using best feature engineering from before)
print("\nLoading and preparing data...")
df = pd.read_csv('source.csv')
target_col = 'effective_minutes'
df = df[df[target_col].notna()].copy()

exclude_cols = [
    'non_working_minutes',
    'pr_number', 'pr_id',
    'created_at', 'closed_at', 'merged_at', 'updated_at',
    'ready_for_review_time', 'workflow_start_time',
    'first_review_time', 'first_approval_time',
    'title', 'description', 'body',
    'reviewers',
    'author_login', 'merged_by_login',
    'repo_language',
    'task_id',
]

feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
X = df[feature_cols].copy()
y = df[target_col].copy()

# Feature engineering (best from previous analysis)
imputer = SimpleImputer(strategy='median')
X_numeric = X.select_dtypes(include=[np.number])
X_numeric_imputed = pd.DataFrame(
    imputer.fit_transform(X_numeric),
    columns=X_numeric.columns,
    index=X_numeric.index
)

categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
X_encoded = X_numeric_imputed.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_col_filled = X[col].fillna('unknown').astype(str)
    X_encoded[col] = le.fit_transform(X_col_filled)
    label_encoders[col] = le

X_base = X_encoded.copy()

# Add engineered features
X_engineered = X_base.copy()

# Interaction features
if 'additions' in X_engineered.columns and 'deletions' in X_engineered.columns:
    X_engineered['additions_deletions_ratio'] = (X_engineered['additions'] + 1) / (X_engineered['deletions'] + 1)

if 'commits' in X_engineered.columns and 'changed_files' in X_engineered.columns:
    X_engineered['commits_per_file'] = X_engineered['commits'] / (X_engineered['changed_files'] + 1)

if 'total_lines_changed' in X_engineered.columns and 'commits' in X_engineered.columns:
    X_engineered['lines_per_commit'] = X_engineered['total_lines_changed'] / (X_engineered['commits'] + 1)

if 'review_count' in X_engineered.columns and 'reviewer_count' in X_engineered.columns:
    X_engineered['reviews_per_reviewer'] = X_engineered['review_count'] / (X_engineered['reviewer_count'] + 1)

if 'time_to_first_review_minutes' in X_engineered.columns and 'time_to_first_approval_minutes' in X_engineered.columns:
    X_engineered['review_to_approval_time'] = X_engineered['time_to_first_approval_minutes'] - X_engineered['time_to_first_review_minutes']

# Polynomial features
for feat in ['time_to_first_approval_minutes', 'commits', 'review_count']:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_squared'] = X_engineered[feat] ** 2

# Log transformations
for feat in ['additions', 'deletions', 'total_lines_changed', 'commits']:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_log'] = np.log1p(X_engineered[feat])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test: {X_test.shape[0]} samples")

# ============================================================================
# Test different algorithms
# ============================================================================
print("\n" + "="*70)
print("TESTING DIFFERENT ALGORITHMS")
print("="*70)

results = []

# 1. XGBoost (baseline - best from before)
print("\n1. XGBoost (Baseline)...")
model_xgb = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
results.append({
    'algorithm': 'XGBoost (Baseline)',
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'model': model_xgb
})
print(f"   Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")

# 2. LightGBM
if HAS_LIGHTGBM:
    print("\n2. LightGBM...")
    model_lgb = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    model_lgb.fit(X_train, y_train)
    y_pred = model_lgb.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    results.append({
        'algorithm': 'LightGBM',
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'model': model_lgb
    })
    print(f"   Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
else:
    print("\n2. LightGBM - Not available (install: pip install lightgbm)")

# 3. CatBoost
if HAS_CATBOOST:
    print("\n3. CatBoost...")
    # CatBoost can handle categoricals directly
    cat_features = [i for i, col in enumerate(X_engineered.columns) if col in categorical_cols]
    model_cat = cb.CatBoostRegressor(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        cat_features=cat_features if cat_features else None
    )
    model_cat.fit(X_train, y_train)
    y_pred = model_cat.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    results.append({
        'algorithm': 'CatBoost',
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'model': model_cat
    })
    print(f"   Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
else:
    print("\n3. CatBoost - Not available (install: pip install catboost)")

# 4. Random Forest
print("\n4. Random Forest...")
model_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
results.append({
    'algorithm': 'Random Forest',
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'model': model_rf
})
print(f"   Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")

# 5. Gradient Boosting (sklearn)
print("\n5. Gradient Boosting (sklearn)...")
model_gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model_gb.fit(X_train, y_train)
y_pred = model_gb.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
results.append({
    'algorithm': 'Gradient Boosting',
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'model': model_gb
})
print(f"   Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")

# 6. Ridge Regression (with regularization)
print("\n6. Ridge Regression...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_ridge = Ridge(alpha=10.0, random_state=42)
model_ridge.fit(X_train_scaled, y_train)
y_pred = model_ridge.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
results.append({
    'algorithm': 'Ridge Regression',
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'model': model_ridge
})
print(f"   Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")

# ============================================================================
# Try target transformation
# ============================================================================
print("\n" + "="*70)
print("TESTING TARGET TRANSFORMATION")
print("="*70)

# Log transform target
y_train_log = np.log1p(y_train)
y_test_log = y_test.values

print("\n7. XGBoost with Log-Transformed Target...")
model_xgb_log = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model_xgb_log.fit(X_train, y_train_log)
y_pred_log = model_xgb_log.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Transform back
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
results.append({
    'algorithm': 'XGBoost (Log Target)',
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'model': model_xgb_log
})
print(f"   Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")

# ============================================================================
# Results Summary
# ============================================================================
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_r2', ascending=False)

print("\nAlgorithm Performance (sorted by Test R²):")
print(results_df[['algorithm', 'test_r2', 'test_rmse', 'test_mae']].to_string(index=False))

best_idx = results_df['test_r2'].idxmax()
best_algorithm = results_df.loc[best_idx, 'algorithm']
best_r2 = results_df.loc[best_idx, 'test_r2']

print(f"\n🏆 Best Algorithm: {best_algorithm}")
print(f"   Test R²: {best_r2:.4f}")
print(f"   Test RMSE: {results_df.loc[best_idx, 'test_rmse']:.2f} minutes")
print(f"   Test MAE: {results_df.loc[best_idx, 'test_mae']:.2f} minutes")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold', y=0.995)

# R² comparison
ax1 = axes[0]
x_pos = np.arange(len(results_df))
colors = ['green' if r == best_r2 else 'steelblue' for r in results_df['test_r2']]
ax1.barh(x_pos, results_df['test_r2'], color=colors, alpha=0.7)
ax1.set_yticks(x_pos)
ax1.set_yticklabels(results_df['algorithm'], fontsize=9)
ax1.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
ax1.set_title('Test R² by Algorithm', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# RMSE comparison
ax2 = axes[1]
ax2.barh(x_pos, results_df['test_rmse'], color=colors, alpha=0.7)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(results_df['algorithm'], fontsize=9)
ax2.set_xlabel('Test RMSE (minutes)', fontsize=11, fontweight='bold')
ax2.set_title('Test RMSE by Algorithm', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved 'algorithm_comparison.png'")

# Save results
results_df[['algorithm', 'test_r2', 'test_rmse', 'test_mae']].to_csv(
    'algorithm_comparison_results.csv', index=False)
print("✅ Saved 'algorithm_comparison_results.csv'")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)


