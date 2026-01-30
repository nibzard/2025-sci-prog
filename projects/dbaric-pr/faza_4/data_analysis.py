import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

# Load data
print("Loading data...")
df = pd.read_csv('source_5.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"Rows: {len(df)}")

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric columns: {len(numeric_cols)}")

# Remove ID columns and other non-meaningful columns for correlation
exclude_from_corr = ['pr_number', 'pr_id', 'author_id']
correlation_cols = [col for col in numeric_cols if col not in exclude_from_corr]

print(f"\nColumns for correlation analysis: {len(correlation_cols)}")

# ============================================================================
# 1. CORRELATION MATRIX
# ============================================================================
print("\n" + "="*60)
print("1. CORRELATION MATRIX ANALYSIS")
print("="*60)

# Calculate correlation matrix
corr_matrix = df[correlation_cols].corr()

# Save correlation matrix to CSV
corr_matrix.to_csv('correlation_matrix.csv')
print("✅ Correlation matrix saved to 'correlation_matrix.csv'")

# Find strongest correlations (excluding self-correlations)
print("\nTop 20 strongest positive correlations:")
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append({
            'feature_1': corr_matrix.columns[i],
            'feature_2': corr_matrix.columns[j],
            'correlation': corr_matrix.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs)
corr_df = corr_df.sort_values('correlation', ascending=False, key=abs)
print(corr_df.head(20).to_string(index=False))

# Save top correlations
corr_df.to_csv('top_correlations.csv', index=False)
print("\n✅ Top correlations saved to 'top_correlations.csv'")

# Visualize correlation matrix
print("\nCreating correlation matrix visualization...")

# Create a large figure for the full correlation matrix
fig, ax = plt.subplots(figsize=(30, 28))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=False, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=True, yticklabels=True, ax=ax)
ax.set_title('Correlation Matrix - All Numeric Features', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('correlation_matrix_full.png', dpi=300, bbox_inches='tight')
print("✅ Full correlation matrix saved as 'correlation_matrix_full.png'")

# Create a focused correlation matrix for top correlated features
top_corr_features = corr_df.head(30)
top_features = list(set(top_corr_features['feature_1'].tolist() + top_corr_features['feature_2'].tolist()))
top_corr_matrix = df[top_features].corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(top_corr_matrix, dtype=bool))
sns.heatmap(top_corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=True, yticklabels=True, ax=ax)
ax.set_title('Correlation Matrix - Top Correlated Features', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('correlation_matrix_top.png', dpi=300, bbox_inches='tight')
print("✅ Top correlations matrix saved as 'correlation_matrix_top.png'")

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*60)
print("2. DESCRIPTIVE STATISTICS")
print("="*60)

desc_stats = df[correlation_cols].describe()
desc_stats.to_csv('descriptive_statistics.csv')
print("✅ Descriptive statistics saved to 'descriptive_statistics.csv'")

# Additional statistics
additional_stats = pd.DataFrame({
    'column': correlation_cols,
    'mean': [df[col].mean() for col in correlation_cols],
    'median': [df[col].median() for col in correlation_cols],
    'std': [df[col].std() for col in correlation_cols],
    'min': [df[col].min() for col in correlation_cols],
    'max': [df[col].max() for col in correlation_cols],
    'skewness': [stats.skew(df[col].dropna()) for col in correlation_cols],
    'kurtosis': [stats.kurtosis(df[col].dropna()) for col in correlation_cols],
    'missing_count': [df[col].isna().sum() for col in correlation_cols],
    'missing_pct': [df[col].isna().sum() / len(df) * 100 for col in correlation_cols],
    'zero_count': [(df[col] == 0).sum() for col in correlation_cols],
    'unique_count': [df[col].nunique() for col in correlation_cols]
})

additional_stats = additional_stats.sort_values('mean', ascending=False)
additional_stats.to_csv('additional_statistics.csv', index=False)
print("✅ Additional statistics saved to 'additional_statistics.csv'")

print("\nTop 10 features by mean value:")
print(additional_stats.head(10)[['column', 'mean', 'median', 'std']].to_string(index=False))

# ============================================================================
# 3. DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("3. DISTRIBUTION ANALYSIS")
print("="*60)

# Select key features for distribution plots
key_features = [
    'effective_minutes', 'non_working_minutes', 'additions', 'deletions', 
    'changed_files', 'commits', 'review_count', 'time_to_first_review_minutes',
    'time_to_first_approval_minutes', 'title_length', 'description_length', 
    'reviewer_count'
]

# Filter to only include features that exist
key_features = [f for f in key_features if f in df.columns]

n_features = len(key_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    data = df[feature].dropna()
    
    ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_title(f'{feature}\n(mean={data.mean():.2f}, median={data.median():.2f})', 
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Hide empty subplots
for idx in range(n_features, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Distribution of Key Features', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
print("✅ Distribution plots saved as 'distributions.png'")

# ============================================================================
# 4. BOX PLOTS FOR OUTLIER DETECTION
# ============================================================================
print("\n" + "="*60)
print("4. OUTLIER DETECTION (Box Plots)")
print("="*60)

n_features = len(key_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    data = df[feature].dropna()
    
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
    ax.set_ylabel('Value', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Hide empty subplots
for idx in range(n_features, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
print("✅ Box plots saved as 'boxplots.png'")

# ============================================================================
# 5. PAIRWISE RELATIONSHIPS
# ============================================================================
print("\n" + "="*60)
print("5. PAIRWISE RELATIONSHIPS")
print("="*60)

# Select most important pairs based on correlation
top_pairs = corr_df.head(12)

fig, axes = plt.subplots(3, 4, figsize=(24, 18))
axes = axes.flatten()

for idx, row in enumerate(top_pairs.iterrows()):
    ax = axes[idx]
    pair = row[1]
    feat1 = pair['feature_1']
    feat2 = pair['feature_2']
    corr_val = pair['correlation']
    
    data1 = df[feat1].dropna()
    data2 = df[feat2].dropna()
    
    # Get common indices
    common_idx = df[[feat1, feat2]].dropna().index
    if len(common_idx) > 0:
        ax.scatter(df.loc[common_idx, feat1], df.loc[common_idx, feat2], 
                  alpha=0.5, s=20, color='steelblue')
        ax.set_xlabel(feat1, fontsize=9)
        ax.set_ylabel(feat2, fontsize=9)
        ax.set_title(f'Corr: {corr_val:.3f}', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

plt.suptitle('Top 12 Feature Pairs by Correlation', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('pairwise_relationships.png', dpi=300, bbox_inches='tight')
print("✅ Pairwise relationships saved as 'pairwise_relationships.png'")

# ============================================================================
# 6. CATEGORICAL ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("6. CATEGORICAL ANALYSIS")
print("="*60)

categorical_cols = ['is_bug_fix', 'is_new_feature', 'is_update', 'is_refactor', 
                    'is_backend', 'is_frontend', 'repo_language']

categorical_cols = [col for col in categorical_cols if col in df.columns]

if categorical_cols:
    n_cats = len(categorical_cols)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        value_counts = df[col].value_counts()
        
        ax.bar(range(len(value_counts)), value_counts.values, color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_title(f'{col}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(value_counts.values):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
    
    # Hide empty subplots
    for idx in range(n_cats, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Categorical Feature Distributions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Categorical analysis saved as 'categorical_analysis.png'")

# ============================================================================
# 7. MISSING DATA ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("7. MISSING DATA ANALYSIS")
print("="*60)

missing_data = pd.DataFrame({
    'column': df.columns,
    'missing_count': [df[col].isna().sum() for col in df.columns],
    'missing_pct': [df[col].isna().sum() / len(df) * 100 for col in df.columns]
})

missing_data = missing_data.sort_values('missing_pct', ascending=False)
missing_data.to_csv('missing_data_analysis.csv', index=False)
print("✅ Missing data analysis saved to 'missing_data_analysis.csv'")

print("\nColumns with missing data:")
print(missing_data[missing_data['missing_count'] > 0].head(20).to_string(index=False))

# Visualize missing data
if missing_data[missing_data['missing_count'] > 0].shape[0] > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    missing_cols = missing_data[missing_data['missing_count'] > 0].head(20)
    ax.barh(range(len(missing_cols)), missing_cols['missing_pct'], color='coral')
    ax.set_yticks(range(len(missing_cols)))
    ax.set_yticklabels(missing_cols['column'], fontsize=9)
    ax.set_xlabel('Missing Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Top 20 Columns with Missing Data', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('missing_data.png', dpi=300, bbox_inches='tight')
    print("✅ Missing data visualization saved as 'missing_data.png'")

# ============================================================================
# 8. SUMMARY REPORT
# ============================================================================
print("\n" + "="*60)
print("8. GENERATING SUMMARY REPORT")
print("="*60)

summary = f"""
DATA ANALYSIS SUMMARY REPORT
{'='*60}

Dataset Information:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}
- Numeric columns: {len(numeric_cols)}
- Categorical columns: {len(categorical_cols) if categorical_cols else 0}

Correlation Analysis:
- Total feature pairs analyzed: {len(corr_df)}
- Strongest positive correlation: {corr_df.iloc[0]['feature_1']} vs {corr_df.iloc[0]['feature_2']} ({corr_df.iloc[0]['correlation']:.3f})
- Strongest negative correlation: {corr_df.iloc[-1]['feature_1']} vs {corr_df.iloc[-1]['feature_2']} ({corr_df.iloc[-1]['correlation']:.3f})

Missing Data:
- Columns with missing data: {len(missing_data[missing_data['missing_count'] > 0])}
- Total missing values: {missing_data['missing_count'].sum()}

Key Features Analyzed:
{', '.join(key_features[:10])}
...

Generated Files:
1. correlation_matrix.csv - Full correlation matrix
2. top_correlations.csv - Top correlations between feature pairs
3. descriptive_statistics.csv - Basic descriptive statistics
4. additional_statistics.csv - Extended statistics (skewness, kurtosis, etc.)
5. missing_data_analysis.csv - Missing data analysis
6. correlation_matrix_full.png - Full correlation heatmap
7. correlation_matrix_top.png - Top correlations heatmap
8. distributions.png - Distribution plots for key features
9. boxplots.png - Box plots for outlier detection
10. pairwise_relationships.png - Scatter plots of top correlated pairs
11. categorical_analysis.png - Categorical feature distributions
12. missing_data.png - Missing data visualization

{'='*60}
"""

with open('analysis_summary.txt', 'w') as f:
    f.write(summary)

print(summary)
print("✅ Summary report saved to 'analysis_summary.txt'")

print("\n" + "="*60)
print("✅ DATA ANALYSIS COMPLETE!")
print("="*60)

