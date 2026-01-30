import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*70)
print("OPTIMIZING RANDOM FOREST MODEL")
print("="*70)

# Load and prepare data (same as before)
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

# Feature engineering
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
X_engineered = X_base.copy()

# Add engineered features
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

for feat in ['time_to_first_approval_minutes', 'commits', 'review_count']:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_squared'] = X_engineered[feat] ** 2

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
# Test different Random Forest configurations
# ============================================================================
print("\n" + "="*70)
print("TESTING RANDOM FOREST CONFIGURATIONS")
print("="*70)

configs = [
    {
        'name': 'Baseline RF',
        'params': {
            'n_estimators': 200,
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
        }
    },
    {
        'name': 'More Trees',
        'params': {
            'n_estimators': 500,
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
        }
    },
    {
        'name': 'Deeper Trees',
        'params': {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
        }
    },
    {
        'name': 'More Regularized',
        'params': {
            'n_estimators': 300,
            'max_depth': 6,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
        }
    },
    {
        'name': 'Balanced',
        'params': {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_split': 8,
            'min_samples_leaf': 3,
        }
    },
    {
        'name': 'Many Trees, Shallow',
        'params': {
            'n_estimators': 1000,
            'max_depth': 5,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
        }
    },
]

results = []

for config in configs:
    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **config['params']
    )
    
    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores)
    
    # Train and evaluate
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results.append({
        'config': config['name'],
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'overfit_gap': train_r2 - test_r2,
        'model': model
    })
    
    print(f"\n{config['name']}:")
    print(f"  CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.2f} minutes")
    print(f"  Overfit gap: {train_r2 - test_r2:.4f}")

# Find best
results_df = pd.DataFrame(results)
best_idx = results_df['test_r2'].idxmax()
best_model = results[best_idx]['model']
best_config = results[best_idx]['config']

print("\n" + "="*70)
print("BEST RANDOM FOREST CONFIGURATION")
print("="*70)
print(f"\nConfiguration: {best_config}")
print(f"Test R²: {results[best_idx]['test_r2']:.4f}")
print(f"Test RMSE: {results[best_idx]['test_rmse']:.2f} minutes ({results[best_idx]['test_rmse']/60:.2f} hours)")
print(f"Test MAE: {results[best_idx]['test_mae']:.2f} minutes ({results[best_idx]['test_mae']/60:.2f} hours)")
print(f"Overfit gap: {results[best_idx]['overfit_gap']:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Random Forest Optimization Results', fontsize=16, fontweight='bold', y=0.995)

# R² comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(results_df))
ax1.bar(x_pos, results_df['test_r2'], color='steelblue', alpha=0.7)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['config'], rotation=45, ha='right')
ax1.set_ylabel('Test R²', fontsize=11, fontweight='bold')
ax1.set_title('Test R² by Configuration', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# RMSE comparison
ax2 = axes[0, 1]
ax2.bar(x_pos, results_df['test_rmse'], color='crimson', alpha=0.7)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df['config'], rotation=45, ha='right')
ax2.set_ylabel('Test RMSE (minutes)', fontsize=11, fontweight='bold')
ax2.set_title('Test RMSE by Configuration', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Overfit gap
ax3 = axes[1, 0]
ax3.bar(x_pos, results_df['overfit_gap'], color='orange', alpha=0.7)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(results_df['config'], rotation=45, ha='right')
ax3.set_ylabel('Overfit Gap', fontsize=11, fontweight='bold')
ax3.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Feature importance
ax4 = axes[1, 1]
top_features = feature_importance.head(15)
ax4.barh(range(len(top_features)), top_features['importance'], color='forestgreen', alpha=0.7)
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['feature'], fontsize=9)
ax4.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax4.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('random_forest_optimization.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved 'random_forest_optimization.png'")

# Best model predictions
y_test_pred_best = best_model.predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Best Random Forest Model ({best_config})', fontsize=16, fontweight='bold', y=0.995)

ax1 = axes[0]
ax1.scatter(y_test, y_test_pred_best, alpha=0.6, color='forestgreen', s=50)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax1.set_title(f'Test Set Predictions\nR² = {results[best_idx]["test_r2"]:.4f}', 
              fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

ax2 = axes[1]
residuals = y_test - y_test_pred_best
ax2.scatter(y_test_pred_best, residuals, alpha=0.6, color='crimson', s=50)
ax2.axhline(y=0, color='black', linestyle='--', lw=2)
ax2.set_xlabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals (minutes)', fontsize=11, fontweight='bold')
ax2.set_title('Residuals Plot', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('best_random_forest_performance.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'best_random_forest_performance.png'")

# Save results
results_df[['config', 'cv_rmse_mean', 'cv_rmse_std', 'train_r2', 'test_r2', 
            'test_rmse', 'test_mae', 'overfit_gap']].to_csv(
    'random_forest_optimization_results.csv', index=False)
print("✅ Saved 'random_forest_optimization_results.csv'")

feature_importance.to_csv('random_forest_feature_importance.csv', index=False)
print("✅ Saved 'random_forest_feature_importance.csv'")

print("\n" + "="*70)
print("✅ OPTIMIZATION COMPLETE!")
print("="*70)


