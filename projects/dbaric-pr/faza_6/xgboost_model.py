import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*60)
print("XGBOOST REGRESSION MODEL")
print("="*60)

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

# Columns to exclude from features
exclude_cols = [
    'non_working_minutes',  # Alternative/complementary metric
    'pr_number', 'pr_id',  # IDs
    'created_at', 'closed_at', 'merged_at', 'updated_at',  # Raw dates
    'ready_for_review_time', 'workflow_start_time',  # Raw dates
    'first_review_time', 'first_approval_time',  # Raw dates
    'title', 'description', 'body',  # Raw text (we have processed versions)
    'reviewers',  # Complex string
    'author_login', 'merged_by_login',  # Names (can encode if needed)
    'repo_language',  # Mostly same value
    'task_id',  # ID
]

# Get feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
print(f"\nNumber of features: {len(feature_cols)}")

# Prepare features and target
X = df[feature_cols].copy()
y = df[target_col].copy()

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
    # Fill NaN with 'unknown' for encoding
    X_col_filled = X[col].fillna('unknown').astype(str)
    X_encoded[col] = le.fit_transform(X_col_filled)
    label_encoders[col] = le

# Final feature matrix
X_final = X_encoded
feature_names = X_final.columns.tolist()

print(f"\nFinal feature matrix shape: {X_final.shape}")

# Train-test split (80-20)
print("\n" + "="*60)
print("TRAIN-TEST SPLIT (80-20)")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")

# Create and train XGBoost model
print("\n" + "="*60)
print("TRAINING XGBOOST MODEL")
print("="*60)

model = xgb.XGBRegressor(
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

print("Training model...")
model.fit(X_train, y_train)
print("✅ Model trained successfully")

# Make predictions
print("\n" + "="*60)
print("MAKING PREDICTIONS")
print("="*60)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
print("\n" + "="*60)
print("MODEL SCORES")
print("="*60)

# Training set metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Test set metrics
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTRAINING SET METRICS:")
print(f"  RMSE (Root Mean Squared Error): {train_rmse:.2f} minutes")
print(f"  MAE (Mean Absolute Error): {train_mae:.2f} minutes")
print(f"  R² Score: {train_r2:.4f}")

print("\nTEST SET METRICS:")
print(f"  RMSE (Root Mean Squared Error): {test_rmse:.2f} minutes")
print(f"  MAE (Mean Absolute Error): {test_mae:.2f} minutes")
print(f"  R² Score: {test_r2:.4f}")

# Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Top 10)")
print("="*60)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Save results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save metrics
metrics_df = pd.DataFrame({
    'metric': ['RMSE', 'MAE', 'R²'],
    'train': [train_rmse, train_mae, train_r2],
    'test': [test_rmse, test_mae, test_r2]
})
metrics_df.to_csv('model_scores.csv', index=False)
print("✅ Model scores saved to 'model_scores.csv'")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✅ Feature importance saved to 'feature_importance.csv'")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_test_pred,
    'error': y_test.values - y_test_pred,
    'abs_error': np.abs(y_test.values - y_test_pred)
})
predictions_df.to_csv('predictions.csv', index=False)
print("✅ Predictions saved to 'predictions.csv'")

# Create visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# 1. Actual vs Predicted scatter plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost Model Performance', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Training set - Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_train, y_train_pred, alpha=0.6, color='steelblue')
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax1.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax1.set_title(f'Training Set\nR² = {train_r2:.4f}, RMSE = {train_rmse:.2f}', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Plot 2: Test set - Actual vs Predicted
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred, alpha=0.6, color='forestgreen')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax2.set_title(f'Test Set\nR² = {test_r2:.4f}, RMSE = {test_rmse:.2f}', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Plot 3: Residuals plot (Test set)
ax3 = axes[1, 0]
residuals = y_test - y_test_pred
ax3.scatter(y_test_pred, residuals, alpha=0.6, color='crimson')
ax3.axhline(y=0, color='black', linestyle='--', lw=2)
ax3.set_xlabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuals (minutes)', fontsize=11, fontweight='bold')
ax3.set_title('Residuals Plot (Test Set)', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Feature Importance (Top 15)
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
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'model_performance.png'")

# ============================================================================
# ADDITIONAL FEATURE EFFECT VISUALIZATIONS
# ============================================================================
print("\nCreating additional feature effect visualizations...")

# Get top 10 features for detailed analysis
top_10_features = feature_importance.head(10)['feature'].tolist()

# 1. Feature vs Target Scatter Plots (Top 10 features)
print("  - Creating feature vs target scatter plots...")
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Feature Effect on Duration (Top 10 Features)', fontsize=16, fontweight='bold', y=0.995)
axes = axes.flatten()

for idx, feat_name in enumerate(top_10_features):
    if idx >= 12:  # Only plot first 12 (3x4 grid)
        break
    ax = axes[idx]
    
    # Get original feature values (before encoding for categorical)
    if feat_name in categorical_cols:
        # For categorical, use original values
        feat_values = df.loc[X_final.index, feat_name].values
    else:
        feat_values = X_final[feat_name].values
    
    target_values = y.values
    
    ax.scatter(feat_values, target_values, alpha=0.5, s=30, color='steelblue')
    ax.set_xlabel(feat_name, fontsize=9, fontweight='bold')
    ax.set_ylabel('Effective Minutes', fontsize=9, fontweight='bold')
    ax.set_title(f'{feat_name}', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add trend line if numeric
    if feat_name not in categorical_cols:
        z = np.polyfit(feat_values, target_values, 1)
        p = np.poly1d(z)
        ax.plot(feat_values, p(feat_values), "r--", alpha=0.8, linewidth=2)

# Hide unused subplots
for idx in range(len(top_10_features), 12):
    axes[idx].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('feature_effect_scatter.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved 'feature_effect_scatter.png'")

# 2. Partial Dependence Plots (Top 8 numeric features)
print("  - Creating partial dependence plots...")
numeric_top_features = [f for f in top_10_features if f not in categorical_cols][:8]

if len(numeric_top_features) > 0:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Partial Dependence Plots: How Features Affect Predicted Duration', 
                 fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, feat_name in enumerate(numeric_top_features):
        ax = axes[idx]
        
        # Get feature range
        feat_values = X_final[feat_name].values
        feat_min, feat_max = feat_values.min(), feat_values.max()
        
        # Create grid of values
        grid_values = np.linspace(feat_min, feat_max, 50)
        predictions = []
        
        # For each grid value, predict with all other features at their mean
        X_temp = X_final.copy()
        for grid_val in grid_values:
            X_temp[feat_name] = grid_val
            pred = model.predict(X_temp)
            predictions.append(pred.mean())
        
        ax.plot(grid_values, predictions, linewidth=2, color='steelblue')
        ax.fill_between(grid_values, predictions, alpha=0.3, color='steelblue')
        ax.set_xlabel(feat_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted Duration (minutes)', fontsize=10, fontweight='bold')
        ax.set_title(f'{feat_name}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(numeric_top_features), 8):
        axes[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('partial_dependence_plots.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved 'partial_dependence_plots.png'")

# 3. Categorical Feature Effects (Box plots)
print("  - Creating categorical feature effect plots...")
categorical_top = [f for f in top_10_features if f in categorical_cols]

if len(categorical_top) > 0:
    n_cats = len(categorical_top)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    fig.suptitle('Duration Distribution by Categorical Features', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if n_cats == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feat_name in enumerate(categorical_top):
        ax = axes[idx]
        
        # Get original categorical values
        feat_original = df.loc[X_final.index, feat_name].values
        target_values = y.values
        
        # Create box plot
        unique_values = np.unique(feat_original)
        data_to_plot = [target_values[feat_original == val] for val in unique_values]
        
        bp = ax.boxplot(data_to_plot, labels=[str(v) for v in unique_values], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xlabel(feat_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Effective Minutes', fontsize=11, fontweight='bold')
        ax.set_title(f'{feat_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(categorical_top), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('categorical_feature_effects.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved 'categorical_feature_effects.png'")

# 4. Correlation Heatmap with Target
print("  - Creating correlation heatmap...")
# Get numeric features only for correlation
numeric_features = [f for f in feature_names if f not in categorical_cols]
corr_data = pd.concat([X_final[numeric_features], y], axis=1)
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=True, yticklabels=True, ax=ax, annot_kws={'size': 8})
ax.set_title('Correlation Matrix: Features vs Effective Duration', 
             fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved 'correlation_heatmap.png'")

# 5. Feature Importance Comparison
print("  - Creating feature importance comparison...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=0.995)

# Top 15 features
ax1 = axes[0]
top_15 = feature_importance.head(15)
ax1.barh(range(len(top_15)), top_15['importance'], color='steelblue', edgecolor='black')
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['feature'], fontsize=10)
ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Most Important Features', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Bottom 15 features
ax2 = axes[1]
bottom_15 = feature_importance.tail(15)
ax2.barh(range(len(bottom_15)), bottom_15['importance'], color='lightcoral', edgecolor='black')
ax2.set_yticks(range(len(bottom_15)))
ax2.set_yticklabels(bottom_15['feature'], fontsize=10)
ax2.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax2.set_title('Bottom 15 Least Important Features', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved 'feature_importance_comparison.png'")

# 6. Feature Effect Summary (Top 6 features with detailed analysis)
print("  - Creating detailed feature effect analysis...")
top_6_features = feature_importance.head(6)['feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Detailed Feature Effect Analysis (Top 6 Features)', 
             fontsize=16, fontweight='bold', y=0.995)
axes = axes.flatten()

for idx, feat_name in enumerate(top_6_features):
    ax = axes[idx]
    
    if feat_name in categorical_cols:
        # Categorical: box plot
        feat_original = df.loc[X_final.index, feat_name].values
        unique_values = np.unique(feat_original)
        data_to_plot = [y.values[feat_original == val] for val in unique_values]
        
        bp = ax.boxplot(data_to_plot, labels=[str(v) for v in unique_values], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('Effective Minutes', fontsize=10, fontweight='bold')
    else:
        # Numeric: scatter with trend
        feat_values = X_final[feat_name].values
        target_values = y.values
        
        ax.scatter(feat_values, target_values, alpha=0.4, s=20, color='steelblue')
        
        # Add trend line
        z = np.polyfit(feat_values, target_values, 1)
        p = np.poly1d(z)
        ax.plot(sorted(feat_values), p(sorted(feat_values)), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax.set_ylabel('Effective Minutes', fontsize=10, fontweight='bold')
        ax.legend()
    
    ax.set_xlabel(feat_name, fontsize=10, fontweight='bold')
    ax.set_title(f'{feat_name}\n(Importance: {feature_importance[feature_importance["feature"]==feat_name]["importance"].values[0]:.4f})', 
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('detailed_feature_effects.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved 'detailed_feature_effects.png'")

print("\n✅ All additional visualizations created!")

# Summary report
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nModel: XGBoost Regressor")
print(f"Target: {target_col}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {len(feature_names)}")
print(f"\nTest Set Performance:")
print(f"  R² Score: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
print(f"  RMSE: {test_rmse:.2f} minutes ({test_rmse/60:.2f} hours)")
print(f"  MAE: {test_mae:.2f} minutes ({test_mae/60:.2f} hours)")
print(f"\nTop 3 Most Important Features:")
for i, row in feature_importance.head(3).iterrows():
    print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)

