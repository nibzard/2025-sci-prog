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

# Check for invalid target values
invalid_targets = df[df[target_col] < 0]
if len(invalid_targets) > 0:
    print(f"⚠️  Warning: Found {len(invalid_targets)} rows with negative target values")
    print("   Removing invalid target values...")
    df = df[df[target_col] >= 0].copy()
    print(f"   Samples after removing invalid targets: {len(df)}")

# Check target distribution
print(f"\nTarget variable statistics:")
print(f"  Min: {df[target_col].min():.2f}")
print(f"  Max: {df[target_col].max():.2f}")
print(f"  Mean: {df[target_col].mean():.2f}")
print(f"  Median: {df[target_col].median():.2f}")
print(f"  Std: {df[target_col].std():.2f}")

# Randomize data before splitting
print("\nRandomizing data...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("✅ Data randomized")

# Columns to exclude from features
exclude_cols = [
    'non_working_minutes',  # Alternative/complementary metric
    'pr_number', 'pr_id',  # IDs
    'created_at', 'closed_at', 'merged_at', 'updated_at',  # Raw dates
    'ready_for_review_time', 'workflow_start_time',  # Raw dates
    'first_review_time', 'first_approval_time',  # Raw dates
    'title', 'description', 'body',  # Raw text (we have processed versions)
    'author', 'merged_by_login',  # Names (can encode if needed)
    'task_id',  # ID
]

# Get feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
print(f"\nNumber of features: {len(feature_cols)}")

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
    # Fill NaN with 'unknown' for encoding
    X_col_filled = X[col].fillna('unknown').astype(str)
    X_encoded[col] = le.fit_transform(X_col_filled)
    label_encoders[col] = le

# Final feature matrix - ensure index alignment
X_final = X_encoded.reset_index(drop=True)
y_final = y.reset_index(drop=True)

# Verify alignment
assert len(X_final) == len(y_final), "Feature and target lengths don't match!"
assert X_final.index.equals(y_final.index), "Feature and target indices don't match!"

feature_names = X_final.columns.tolist()

print(f"\nFinal feature matrix shape: {X_final.shape}")
print(f"Target vector length: {len(y_final)}")
print("✅ Index alignment verified")

# Train-test split (80-20) with shuffle
print("\n" + "="*60)
print("TRAIN-TEST SPLIT (80-20) - RANDOMIZED")
print("="*60)
# Use different random_state for split to ensure better randomization
# independent from the initial data shuffle (using seed 123 instead of 42)
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=123, shuffle=True
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

# Debug: Check prediction ranges
print(f"\nTarget statistics:")
print(f"  Train y: min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")
print(f"  Test y: min={y_test.min():.2f}, max={y_test.max():.2f}, mean={y_test.mean():.2f}")
print(f"\nPrediction statistics:")
print(f"  Train pred: min={y_train_pred.min():.2f}, max={y_train_pred.max():.2f}, mean={y_train_pred.mean():.2f}")
print(f"  Test pred: min={y_test_pred.min():.2f}, max={y_test_pred.max():.2f}, mean={y_test_pred.mean():.2f}")

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

# Save predictions - ensure we use arrays for consistency
y_test_array = np.array(y_test)
predictions_df = pd.DataFrame({
    'actual': y_test_array,
    'predicted': y_test_pred,
    'error': y_test_array - y_test_pred,
    'abs_error': np.abs(y_test_array - y_test_pred)
})
predictions_df.to_csv('predictions.csv', index=False)
print("✅ Predictions saved to 'predictions.csv'")

# Create visualizations with scores on images
print("\n" + "="*60)
print("CREATING VISUALIZATIONS WITH SCORES")
print("="*60)

# 1. Main Performance Dashboard with scores displayed
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost Model Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Training set - Actual vs Predicted with scores
ax1 = axes[0, 0]
ax1.scatter(y_train, y_train_pred, alpha=0.6, color='steelblue')
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax1.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax1.set_title('Training Set Performance', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Add scores as text on the plot
score_text = f'R² = {train_r2:.4f}\nRMSE = {train_rmse:.2f} min\nMAE = {train_mae:.2f} min'
ax1.text(0.05, 0.95, score_text, transform=ax1.transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.8), fontweight='bold')

# Plot 2: Test set - Actual vs Predicted with scores
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred, alpha=0.6, color='forestgreen')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual (minutes)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax2.set_title('Test Set Performance', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Add scores as text on the plot
score_text = f'R² = {test_r2:.4f}\nRMSE = {test_rmse:.2f} min\nMAE = {test_mae:.2f} min'
ax2.text(0.05, 0.95, score_text, transform=ax2.transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='lightgreen', alpha=0.8), fontweight='bold')

# Plot 3: Residuals plot (Test set) with statistics
ax3 = axes[1, 0]
residuals = np.array(y_test) - y_test_pred
ax3.scatter(y_test_pred, residuals, alpha=0.6, color='crimson')
ax3.axhline(y=0, color='black', linestyle='--', lw=2)
ax3.set_xlabel('Predicted (minutes)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuals (minutes)', fontsize=11, fontweight='bold')
ax3.set_title('Residuals Plot (Test Set)', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Add residual statistics
residual_mean = residuals.mean()
residual_std = residuals.std()
residual_text = f'Mean: {residual_mean:.2f} min\nStd: {residual_std:.2f} min'
ax3.text(0.05, 0.95, residual_text, transform=ax3.transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='lightcoral', alpha=0.8), fontweight='bold')

# Plot 4: Feature Importance (Top 15) with importance scores
ax4 = axes[1, 1]
top_features = feature_importance.head(15)
ax4.barh(range(len(top_features)), top_features['importance'], color='steelblue')
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['feature'], fontsize=9)
ax4.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax4.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# Add top feature score
top_feat_score = top_features.iloc[0]['importance']
top_feat_name = top_features.iloc[0]['feature']
top_text = f'Top: {top_feat_name}\nScore: {top_feat_score:.4f}'
ax4.text(0.95, 0.05, top_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'model_performance.png'")

# 2. Scores Summary Visualization
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('Model Performance Scores Summary', fontsize=16, fontweight='bold', y=0.98)

metrics = ['RMSE', 'MAE', 'R² Score']
train_values = [train_rmse, train_mae, train_r2]
test_values = [test_rmse, test_mae, test_r2]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, train_values, width, label='Training Set', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, test_values, width, label='Test Set', color='forestgreen', alpha=0.8)

ax.set_ylabel('Score Value', fontsize=12, fontweight='bold')
ax.set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add overall summary text
summary_text = f'Test R²: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)\n'
summary_text += f'Test RMSE: {test_rmse:.2f} minutes ({test_rmse/60:.2f} hours)\n'
summary_text += f'Test MAE: {test_mae:.2f} minutes ({test_mae/60:.2f} hours)'
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
        facecolor='wheat', alpha=0.9), fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('model_scores.png', dpi=300, bbox_inches='tight')
print("✅ Scores visualization saved as 'model_scores.png'")

# 3. Feature Importance with scores
fig, ax = plt.subplots(figsize=(14, 10))
fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=0.98)

top_20 = feature_importance.head(20)
bars = ax.barh(range(len(top_20)), top_20['importance'], color='steelblue', edgecolor='black')
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'], fontsize=10)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add importance scores on bars
for i, (idx, row) in enumerate(top_20.iterrows()):
    ax.text(row['importance'], i, f' {row["importance"]:.4f}',
            va='center', fontsize=9, fontweight='bold')

# Add summary text
top3_text = 'Top 3 Features:\n'
for i, (idx, row) in enumerate(top_20.head(3).iterrows()):
    top3_text += f'{i+1}. {row["feature"]}: {row["importance"]:.4f}\n'
ax.text(0.98, 0.02, top3_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Feature importance visualization saved as 'feature_importance.png'")

# 4. Prediction Accuracy Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Prediction Accuracy Analysis', fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Error distribution
ax1 = axes[0]
errors = y_test_array - y_test_pred
ax1.hist(errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax1.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
ax1.set_xlabel('Prediction Error (minutes)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Error Distribution (Test Set)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Add statistics
error_stats = f'Mean Error: {errors.mean():.2f} min\n'
error_stats += f'Std Error: {errors.std():.2f} min\n'
error_stats += f'Median Error: {np.median(errors):.2f} min'
ax1.text(0.98, 0.98, error_stats, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontweight='bold')

# Plot 2: Absolute error distribution
ax2 = axes[1]
abs_errors = np.abs(errors)
ax2.hist(abs_errors, bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
ax2.axvline(x=abs_errors.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {abs_errors.mean():.2f}')
ax2.set_xlabel('Absolute Prediction Error (minutes)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Absolute Error Distribution (Test Set)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Add statistics
abs_error_stats = f'Mean Abs Error: {abs_errors.mean():.2f} min\n'
abs_error_stats += f'Median Abs Error: {np.median(abs_errors):.2f} min\n'
abs_error_stats += f'Max Abs Error: {abs_errors.max():.2f} min'
ax2.text(0.98, 0.98, abs_error_stats, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('prediction_accuracy.png', dpi=300, bbox_inches='tight')
print("✅ Prediction accuracy visualization saved as 'prediction_accuracy.png'")

print("\n✅ All visualizations with scores created!")

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
