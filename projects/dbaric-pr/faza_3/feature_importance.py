import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

# Load data
print("Loading data...")
df = pd.read_csv('source.csv')

# Select target variable (effective_hours - time to merge excluding non-working hours)
target_col = 'effective_hours'
print(f"Target variable: {target_col}")

# Remove rows where target is missing
df = df[df[target_col].notna()].copy()

# Columns to exclude from features
exclude_cols = [
    'duration_hours', 'duration_days',  # Alternative targets
    'effective_days',  # Alternative target
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

print(f"Number of features: {len(feature_cols)}")
print(f"Number of samples: {len(df)}")

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

# Method 1: Random Forest Feature Importance
print("\n" + "="*60)
print("Method 1: Random Forest Feature Importance")
print("="*60)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_final, y)
rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features (Random Forest):")
print(rf_importance.head(10).to_string(index=False))
print("\nBottom 10 least important features (Random Forest):")
print(rf_importance.tail(10).to_string(index=False))

# Method 2: Mutual Information Regression
print("\n" + "="*60)
print("Method 2: Mutual Information Regression")
print("="*60)
mi_scores = mutual_info_regression(X_final, y, random_state=42)
mi_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': mi_scores
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features (Mutual Information):")
print(mi_importance.head(10).to_string(index=False))
print("\nBottom 10 least important features (Mutual Information):")
print(mi_importance.tail(10).to_string(index=False))

# Method 3: F-regression
print("\n" + "="*60)
print("Method 3: F-regression")
print("="*60)
f_scores, p_values = f_regression(X_final, y)
f_importance = pd.DataFrame({
    'feature': feature_names,
    'f_score': f_scores,
    'p_value': p_values
}).sort_values('f_score', ascending=False)

print("\nTop 10 most important features (F-regression):")
print(f_importance.head(10).to_string(index=False))
print("\nBottom 10 least important features (F-regression):")
print(f_importance.tail(10).to_string(index=False))

# Normalize scores for comparison (0-1 scale)
rf_importance['importance_norm'] = (rf_importance['importance'] - rf_importance['importance'].min()) / (rf_importance['importance'].max() - rf_importance['importance'].min())
mi_importance['importance_norm'] = (mi_importance['importance'] - mi_importance['importance'].min()) / (mi_importance['importance'].max() - mi_importance['importance'].min())
f_importance['importance_norm'] = (f_importance['f_score'] - f_importance['f_score'].min()) / (f_importance['f_score'].max() - f_importance['f_score'].min())

# Create comprehensive visualization
print("\n" + "="*60)
print("Creating visualization...")
print("="*60)

# Create a larger figure with 6 subplots: 3 for most important, 3 for least important
fig, axes = plt.subplots(3, 2, figsize=(24, 20))
fig.suptitle('Feature Importance Analysis: Most and Least Important Features', 
             fontsize=18, fontweight='bold', y=0.995)

top_n = 15
bottom_n = 15

# Plot 1: Random Forest (Top)
ax1 = axes[0, 0]
rf_top = rf_importance.head(top_n)
ax1.barh(range(len(rf_top)), rf_top['importance'], color='steelblue')
ax1.set_yticks(range(len(rf_top)))
ax1.set_yticklabels(rf_top['feature'], fontsize=9)
ax1.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax1.set_title(f'Random Forest - Most Important (Top {top_n})', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Random Forest (Bottom)
ax2 = axes[0, 1]
rf_bottom = rf_importance.tail(bottom_n)
ax2.barh(range(len(rf_bottom)), rf_bottom['importance'], color='lightcoral')
ax2.set_yticks(range(len(rf_bottom)))
ax2.set_yticklabels(rf_bottom['feature'], fontsize=9)
ax2.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax2.set_title(f'Random Forest - Least Important (Bottom {bottom_n})', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Mutual Information (Top)
ax3 = axes[1, 0]
mi_top = mi_importance.head(top_n)
ax3.barh(range(len(mi_top)), mi_top['importance'], color='forestgreen')
ax3.set_yticks(range(len(mi_top)))
ax3.set_yticklabels(mi_top['feature'], fontsize=9)
ax3.set_xlabel('Mutual Information Score', fontsize=11, fontweight='bold')
ax3.set_title(f'Mutual Information - Most Important (Top {top_n})', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Mutual Information (Bottom)
ax4 = axes[1, 1]
mi_bottom = mi_importance.tail(bottom_n)
ax4.barh(range(len(mi_bottom)), mi_bottom['importance'], color='lightcoral')
ax4.set_yticks(range(len(mi_bottom)))
ax4.set_yticklabels(mi_bottom['feature'], fontsize=9)
ax4.set_xlabel('Mutual Information Score', fontsize=11, fontweight='bold')
ax4.set_title(f'Mutual Information - Least Important (Bottom {bottom_n})', fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# Plot 5: F-regression (Top)
ax5 = axes[2, 0]
f_top = f_importance.head(top_n)
ax5.barh(range(len(f_top)), f_top['f_score'], color='crimson')
ax5.set_yticks(range(len(f_top)))
ax5.set_yticklabels(f_top['feature'], fontsize=9)
ax5.set_xlabel('F-score', fontsize=11, fontweight='bold')
ax5.set_title(f'F-regression - Most Important (Top {top_n})', fontsize=12, fontweight='bold')
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)

# Plot 6: F-regression (Bottom)
ax6 = axes[2, 1]
f_bottom = f_importance.tail(bottom_n)
ax6.barh(range(len(f_bottom)), f_bottom['f_score'], color='lightcoral')
ax6.set_yticks(range(len(f_bottom)))
ax6.set_yticklabels(f_bottom['feature'], fontsize=9)
ax6.set_xlabel('F-score', fontsize=11, fontweight='bold')
ax6.set_title(f'F-regression - Least Important (Bottom {bottom_n})', fontsize=12, fontweight='bold')
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'feature_importance_analysis.png'")

# Save results to CSV
print("\nSaving results to CSV...")
results_combined = pd.DataFrame({
    'feature': feature_names,
    'rf_importance': rf_importance.sort_values('feature')['importance'].values,
    'rf_rank': rf_importance.sort_values('feature')['importance'].rank(ascending=False).values,
    'mi_importance': mi_importance.sort_values('feature')['importance'].values,
    'mi_rank': mi_importance.sort_values('feature')['importance'].rank(ascending=False).values,
    'f_score': f_importance.sort_values('feature')['f_score'].values,
    'f_p_value': f_importance.sort_values('feature')['p_value'].values,
    'f_rank': f_importance.sort_values('feature')['f_score'].rank(ascending=False).values,
})

results_combined = results_combined.sort_values('rf_importance', ascending=False)
results_combined.to_csv('feature_importance_results.csv', index=False)
print("✅ Results saved to 'feature_importance_results.csv'")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
print(f"\nSummary:")
print(f"- Total features analyzed: {len(feature_names)}")
print(f"- Samples: {len(X_final)}")
print(f"- Target: {target_col}")
print(f"\nFiles created:")
print(f"  - feature_importance_analysis.png (visualization)")
print(f"  - feature_importance_results.csv (detailed results)")

