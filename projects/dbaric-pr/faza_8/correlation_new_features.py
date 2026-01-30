import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

print("="*70)
print("CORRELATION ANALYSIS FOR NEW ENGINEERED FEATURES")
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
print("\nPreprocessing data...")
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

# ============================================================================
# CREATE NEW ENGINEERED FEATURES
# ============================================================================
print("\n" + "="*70)
print("CREATING NEW ENGINEERED FEATURES")
print("="*70)

X_engineered = X_base.copy()
new_features = []

# Create interaction features (ratios)
print("\nCreating interaction features...")
if 'additions' in X_engineered.columns and 'deletions' in X_engineered.columns:
    X_engineered['additions_deletions_ratio'] = (X_engineered['additions'] + 1) / (X_engineered['deletions'] + 1)
    new_features.append('additions_deletions_ratio')
    print("  ✅ additions_deletions_ratio")

if 'commits' in X_engineered.columns and 'changed_files' in X_engineered.columns:
    X_engineered['commits_per_file'] = X_engineered['commits'] / (X_engineered['changed_files'] + 1)
    new_features.append('commits_per_file')
    print("  ✅ commits_per_file")

if 'total_lines_changed' in X_engineered.columns and 'commits' in X_engineered.columns:
    X_engineered['lines_per_commit'] = X_engineered['total_lines_changed'] / (X_engineered['commits'] + 1)
    new_features.append('lines_per_commit')
    print("  ✅ lines_per_commit")

if 'review_count' in X_engineered.columns and 'reviewer_count' in X_engineered.columns:
    X_engineered['reviews_per_reviewer'] = X_engineered['review_count'] / (X_engineered['reviewer_count'] + 1)
    new_features.append('reviews_per_reviewer')
    print("  ✅ reviews_per_reviewer")

if 'time_to_first_review_minutes' in X_engineered.columns and 'time_to_first_approval_minutes' in X_engineered.columns:
    X_engineered['review_to_approval_time'] = np.maximum(0, 
        X_engineered['time_to_first_approval_minutes'] - X_engineered['time_to_first_review_minutes'])
    new_features.append('review_to_approval_time')
    print("  ✅ review_to_approval_time")

# Create polynomial features for important features
print("\nCreating polynomial features...")
important_features = ['time_to_first_approval_minutes', 'commits', 'review_count', 
                     'total_lines_changed', 'changed_files']
for feat in important_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_squared'] = X_engineered[feat] ** 2
        new_features.append(f'{feat}_squared')
        print(f"  ✅ {feat}_squared")

# Create log transformations for skewed features
print("\nCreating log transformations...")
skewed_features = ['additions', 'deletions', 'total_lines_changed', 'commits', 
                   'review_count', 'comments', 'review_comments']
for feat in skewed_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_log'] = np.log1p(X_engineered[feat])
        new_features.append(f'{feat}_log')
        print(f"  ✅ {feat}_log")

# Create time-based features from dates if available
print("\nCreating time-based features...")
if 'created_at' in df.columns:
    try:
        df['created_at_parsed'] = pd.to_datetime(df['created_at'], errors='coerce')
        X_engineered['created_hour'] = df['created_at_parsed'].dt.hour
        X_engineered['created_day_of_week'] = df['created_at_parsed'].dt.dayofweek
        X_engineered['created_is_weekend'] = (X_engineered['created_day_of_week'] >= 5).astype(int)
        new_features.extend(['created_hour', 'created_day_of_week', 'created_is_weekend'])
        print("  ✅ Time-based features from created_at")
    except:
        pass

print(f"\nTotal new features created: {len(new_features)}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

# Create dataframe with new features and target
corr_data = pd.concat([X_engineered[new_features], y_base], axis=1)
corr_data.columns = list(new_features) + [target_col]

# Calculate correlation matrix
corr_matrix = corr_data.corr()

# Get correlations with target
target_correlations = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)

print("\nCorrelations with target (effective_minutes):")
print(target_correlations.to_string())

# Save correlations to CSV
target_corr_df = pd.DataFrame({
    'feature': target_correlations.index,
    'correlation': target_correlations.values,
    'abs_correlation': np.abs(target_correlations.values)
}).sort_values('abs_correlation', ascending=False)

target_corr_df.to_csv('new_features_correlations.csv', index=False)
print("\n✅ Correlations saved to 'new_features_correlations.csv'")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. Full Correlation Matrix Heatmap
print("\n1. Creating full correlation matrix heatmap...")
fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=True, yticklabels=True, ax=ax, annot_kws={'size': 9})
ax.set_title('Correlation Matrix: New Engineered Features vs Target', 
             fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('correlation_new_features_full.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'correlation_new_features_full.png'")

# 2. Correlation Bar Chart (Top features with target)
print("\n2. Creating correlation bar chart...")
fig, ax = plt.subplots(figsize=(14, 10))
top_corr = target_correlations.head(20)
colors = ['steelblue' if x > 0 else 'crimson' for x in top_corr.values]
bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors, edgecolor='black')
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr.index, fontsize=10)
ax.set_xlabel('Correlation with Effective Minutes', fontsize=12, fontweight='bold')
ax.set_title('Top 20 New Features: Correlation with Target', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Add value labels
for i, (idx, val) in enumerate(top_corr.items()):
    ax.text(val, i, f' {val:.3f}', va='center', 
            ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', label='Positive Correlation'),
                   Patch(facecolor='crimson', label='Negative Correlation')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('correlation_new_features_bar.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'correlation_new_features_bar.png'")

# 3. Focused Heatmap (Top correlated features only)
print("\n3. Creating focused correlation heatmap (top features)...")
top_n = min(15, len(new_features))
top_features_for_corr = target_corr_df.head(top_n)['feature'].tolist()
top_features_for_corr.append(target_col)

corr_top = corr_data[top_features_for_corr].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_top, dtype=bool))
sns.heatmap(corr_top, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=True, yticklabels=True, ax=ax, annot_kws={'size': 10})
ax.set_title(f'Correlation Matrix: Top {top_n} New Features vs Target', 
             fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('correlation_new_features_top.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'correlation_new_features_top.png'")

# 4. Scatter plots for top correlated features
print("\n4. Creating scatter plots for top correlated features...")
top_6_features = target_corr_df.head(6)['feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Top 6 New Features: Relationship with Target', fontsize=16, fontweight='bold', y=0.995)
axes = axes.flatten()

for idx, feat_name in enumerate(top_6_features):
    ax = axes[idx]
    feat_values = X_engineered[feat_name].values
    target_values = y_base.values
    
    ax.scatter(feat_values, target_values, alpha=0.5, s=30, color='steelblue')
    ax.set_xlabel(feat_name, fontsize=11, fontweight='bold')
    ax.set_ylabel('Effective Minutes', fontsize=11, fontweight='bold')
    ax.set_title(f'{feat_name}\n(Correlation: {target_correlations[feat_name]:.3f})', 
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(feat_values, target_values, 1)
    p = np.poly1d(z)
    ax.plot(sorted(feat_values), p(sorted(feat_values)), "r--", alpha=0.8, linewidth=2)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('correlation_new_features_scatter.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'correlation_new_features_scatter.png'")

# 5. Summary Statistics
print("\n5. Creating summary statistics visualization...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('New Features: Correlation Summary', fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Positive vs Negative correlations
ax1 = axes[0]
positive_corr = target_correlations[target_correlations > 0]
negative_corr = target_correlations[target_correlations < 0]

ax1.barh(['Positive', 'Negative'], 
         [len(positive_corr), len(negative_corr)],
         color=['steelblue', 'crimson'], edgecolor='black')
ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax1.set_title('Correlation Sign Distribution', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add value labels
ax1.text(len(positive_corr), 0, f' {len(positive_corr)}', va='center', 
         ha='left', fontsize=12, fontweight='bold')
ax1.text(len(negative_corr), 1, f' {len(negative_corr)}', va='center', 
         ha='left', fontsize=12, fontweight='bold')

# Plot 2: Correlation strength distribution
ax2 = axes[1]
corr_bins = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5+']
corr_counts = [
    len(target_correlations[(np.abs(target_correlations) >= 0.0) & (np.abs(target_correlations) < 0.1)]),
    len(target_correlations[(np.abs(target_correlations) >= 0.1) & (np.abs(target_correlations) < 0.2)]),
    len(target_correlations[(np.abs(target_correlations) >= 0.2) & (np.abs(target_correlations) < 0.3)]),
    len(target_correlations[(np.abs(target_correlations) >= 0.3) & (np.abs(target_correlations) < 0.4)]),
    len(target_correlations[(np.abs(target_correlations) >= 0.4) & (np.abs(target_correlations) < 0.5)]),
    len(target_correlations[np.abs(target_correlations) >= 0.5])
]

bars = ax2.bar(corr_bins, corr_counts, color='steelblue', edgecolor='black')
ax2.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax2.set_xlabel('Absolute Correlation Range', fontsize=12, fontweight='bold')
ax2.set_title('Correlation Strength Distribution', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

# Add value labels
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('correlation_new_features_summary.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'correlation_new_features_summary.png'")

print("\n" + "="*70)
print("✅ CORRELATION ANALYSIS COMPLETE!")
print("="*70)
print(f"\nTotal new features analyzed: {len(new_features)}")
print(f"Strongest positive correlation: {target_correlations.idxmax()} ({target_correlations.max():.3f})")
print(f"Strongest negative correlation: {target_correlations.idxmin()} ({target_correlations.min():.3f})")
print(f"Average absolute correlation: {np.abs(target_correlations).mean():.3f}")


