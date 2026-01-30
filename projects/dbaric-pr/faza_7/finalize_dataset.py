import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 10

print("="*70)
print("FINALIZING DATASET")
print("="*70)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('source_final.csv')
print(f"   Initial shape: {df.shape}")

# ============================================================================
# REMOVE reviewers COLUMN
# ============================================================================
print("\n2. Removing reviewers column...")
if 'reviewers' in df.columns:
    df = df.drop(columns=['reviewers'])
    print("   ✅ Removed reviewers column (features already extracted)")

print(f"   After removal: {df.shape}")

# ============================================================================
# CATEGORICAL ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("3. CATEGORICAL ANALYSIS")
print("="*70)

categorical_cols = ['pr_type', 'position']

# Add reviewer binary columns to categorical analysis
reviewer_cols = [col for col in df.columns if col.startswith('has_reviewer_')]
categorical_cols.extend(reviewer_cols)

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
    
    plt.suptitle('Categorical Feature Distributions - FINAL', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('categorical_analysis_final.png', dpi=300, bbox_inches='tight')
    print("✅ Categorical analysis saved as 'categorical_analysis_final.png'")
    
    # Print statistics
    print("\n   Categorical feature distributions:")
    for col in ['pr_type', 'position']:
        if col in df.columns:
            print(f"\n   {col}:")
            print(df[col].value_counts().to_string())

# ============================================================================
# DATASET SUMMARY
# ============================================================================
print("\n" + "="*70)
print("4. DATASET SUMMARY")
print("="*70)

print(f"\n   Dataset shape: {df.shape}")
print(f"   Total rows: {len(df)}")
print(f"   Total columns: {len(df.columns)}")

print(f"\n   Column categories:")
print(f"   - PR type: pr_type")
print(f"   - Position: position")
print(f"   - Reviewer features: {len(reviewer_cols)} binary columns")
print(f"   - Team size features: reviewer_team_small, reviewer_team_medium, reviewer_team_large")
print(f"   - Author feature: author_is_reviewer")

print(f"\n   PR type distribution:")
print(df['pr_type'].value_counts().to_string())

print(f"\n   Position distribution:")
print(df['position'].value_counts().to_string())

# ============================================================================
# SAVE FINAL DATASET
# ============================================================================
print("\n" + "="*70)
print("5. SAVING FINAL DATASET")
print("="*70)

output_file = 'source_final.csv'
df.to_csv(output_file, index=False)
print(f"✅ Final dataset saved to '{output_file}'")
print(f"   Final shape: {df.shape}")

# List all columns
print(f"\n   All columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n" + "="*70)
print("✅ DATASET FINALIZATION COMPLETE!")
print("="*70)


