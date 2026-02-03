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
print("DATA QUALITY IMPROVEMENT SCRIPT")
print("="*70)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('source.csv')
print(f"   Initial dataset shape: {df.shape}")
print(f"   Initial columns: {len(df.columns)}")

# ============================================================================
# CATEGORICAL ANALYSIS - BEFORE
# ============================================================================
print("\n" + "="*70)
print("2. CATEGORICAL ANALYSIS - BEFORE")
print("="*70)

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
    
    plt.suptitle('Categorical Feature Distributions - BEFORE', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('categorical_analysis_before.png', dpi=300, bbox_inches='tight')
    print("✅ Categorical analysis (BEFORE) saved as 'categorical_analysis_before.png'")
    
    # Print statistics
    print("\n   Categorical feature distributions (BEFORE):")
    for col in categorical_cols:
        print(f"\n   {col}:")
        print(df[col].value_counts().to_string())

# ============================================================================
# STEP A: REMOVE REDUNDANT COLUMNS
# ============================================================================
print("\n" + "="*70)
print("3. STEP A: REMOVING REDUNDANT COLUMNS")
print("="*70)

# Identify redundant columns based on previous analysis
redundant_cols = [
    'title_length',  # redundant with title_word_count (if exists) or can be derived
    'description_length',  # redundant with description_word_count
    'title_croatian_words',  # if we only need one language indicator
    'description_croatian_words',  # if we only need one language indicator
    'description_croatian_pct',  # if we only need one language indicator
    # Keep description_word_count as it's more meaningful than length
]

# Check which redundant columns actually exist
redundant_cols = [col for col in redundant_cols if col in df.columns]
print(f"   Removing {len(redundant_cols)} redundant columns: {redundant_cols}")

df_cleaned = df.drop(columns=redundant_cols)
print(f"   After removing redundant columns: {df_cleaned.shape}")

# ============================================================================
# STEP B: ADD is_fullstack AND BALANCE BY TECHNOLOGY STACK
# ============================================================================
print("\n" + "="*70)
print("4. STEP B: ADDING is_fullstack AND BALANCING BY TECHNOLOGY STACK")
print("="*70)

# Add is_fullstack column
# When both is_frontend and is_backend are True, set is_fullstack=True
# and set is_frontend=False, is_backend=False
df_cleaned['is_fullstack'] = (df_cleaned['is_frontend'] == True) & (df_cleaned['is_backend'] == True)

# When is_fullstack is True, set is_frontend and is_backend to False
df_cleaned.loc[df_cleaned['is_fullstack'] == True, 'is_frontend'] = False
df_cleaned.loc[df_cleaned['is_fullstack'] == True, 'is_backend'] = False

print(f"   Technology stack distribution before balancing:")
print(f"   - is_frontend=True: {(df_cleaned['is_frontend'] == True).sum()}")
print(f"   - is_backend=True: {(df_cleaned['is_backend'] == True).sum()}")
print(f"   - is_fullstack=True: {(df_cleaned['is_fullstack'] == True).sum()}")
print(f"   - None (False for all): {((df_cleaned['is_frontend'] == False) & (df_cleaned['is_backend'] == False) & (df_cleaned['is_fullstack'] == False)).sum()}")

# Balance by technology stack
# Get counts for each category
frontend_count = (df_cleaned['is_frontend'] == True).sum()
backend_count = (df_cleaned['is_backend'] == True).sum()
fullstack_count = (df_cleaned['is_fullstack'] == True).sum()
none_count = ((df_cleaned['is_frontend'] == False) & 
              (df_cleaned['is_backend'] == False) & 
              (df_cleaned['is_fullstack'] == False)).sum()

# Balance frontend and backend to be equal (requirement: "Ensure frontend and backend PRs are equally represented")
frontend_backend_target = min(frontend_count, backend_count)
print(f"\n   Balancing frontend and backend to: {frontend_backend_target} each")
print(f"   Keeping fullstack: {fullstack_count}")
print(f"   Keeping none: {none_count}")

# Sample equally from frontend and backend
balanced_dfs = []

if frontend_count > 0:
    frontend_df = df_cleaned[df_cleaned['is_frontend'] == True].sample(n=frontend_backend_target, random_state=42)
    balanced_dfs.append(frontend_df)

if backend_count > 0:
    backend_df = df_cleaned[df_cleaned['is_backend'] == True].sample(n=frontend_backend_target, random_state=42)
    balanced_dfs.append(backend_df)

# Keep all fullstack (they're already balanced as a separate category)
if fullstack_count > 0:
    balanced_dfs.append(df_cleaned[df_cleaned['is_fullstack'] == True])

# Keep all none (or sample if too many)
if none_count > 0:
    # Keep all if reasonable, otherwise sample
    max_none = min(none_count, frontend_backend_target * 2)  # Don't let none dominate
    none_df = df_cleaned[(df_cleaned['is_frontend'] == False) & 
                         (df_cleaned['is_backend'] == False) & 
                         (df_cleaned['is_fullstack'] == False)].sample(n=min(max_none, none_count), random_state=42)
    balanced_dfs.append(none_df)

df_tech_balanced = pd.concat(balanced_dfs, ignore_index=True)
df_tech_balanced = df_tech_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n   Technology stack distribution after balancing:")
print(f"   - is_frontend=True: {(df_tech_balanced['is_frontend'] == True).sum()}")
print(f"   - is_backend=True: {(df_tech_balanced['is_backend'] == True).sum()}")
print(f"   - is_fullstack=True: {(df_tech_balanced['is_fullstack'] == True).sum()}")
print(f"   - None (False for all): {((df_tech_balanced['is_frontend'] == False) & (df_tech_balanced['is_backend'] == False) & (df_tech_balanced['is_fullstack'] == False)).sum()}")
print(f"   Total rows after tech balancing: {len(df_tech_balanced)}")

# ============================================================================
# STEP C: BALANCE BY PR TYPE
# ============================================================================
print("\n" + "="*70)
print("5. STEP C: BALANCING BY PR TYPE")
print("="*70)

# Check which PR type columns exist
pr_type_cols = ['is_bug_fix', 'is_new_feature', 'is_update']
pr_type_cols = [col for col in pr_type_cols if col in df_tech_balanced.columns]

print(f"   PR type columns: {pr_type_cols}")

# Create a combined PR type column for easier balancing
# Priority: is_bug_fix > is_new_feature > is_update
df_tech_balanced['pr_type'] = 'other'
if 'is_bug_fix' in pr_type_cols:
    df_tech_balanced.loc[df_tech_balanced['is_bug_fix'] == True, 'pr_type'] = 'fix'
if 'is_new_feature' in pr_type_cols:
    df_tech_balanced.loc[(df_tech_balanced['is_new_feature'] == True) & 
                         (df_tech_balanced['pr_type'] == 'other'), 'pr_type'] = 'feature'
if 'is_update' in pr_type_cols:
    df_tech_balanced.loc[(df_tech_balanced['is_update'] == True) & 
                         (df_tech_balanced['pr_type'] == 'other'), 'pr_type'] = 'update'

print(f"\n   PR type distribution before balancing:")
pr_type_counts = df_tech_balanced['pr_type'].value_counts()
print(pr_type_counts.to_string())

# Balance by PR type - only balance fix, feature, and update (requirement: "is_fix, is_update, and is_feature are as equally represented as possible")
pr_types_to_balance = ['fix', 'feature', 'update']
pr_counts_to_balance = {pt: pr_type_counts.get(pt, 0) for pt in pr_types_to_balance}
min_pr_count = min([c for c in pr_counts_to_balance.values() if c > 0])
print(f"\n   Balancing fix, feature, and update to: {min_pr_count} each")
print(f"   Keeping other: {pr_type_counts.get('other', 0)}")

balanced_pr_dfs = []
for pr_type in pr_type_counts.index:
    if pr_type in pr_types_to_balance:
        # Balance these three types
        pr_df = df_tech_balanced[df_tech_balanced['pr_type'] == pr_type].sample(
            n=min(min_pr_count, pr_type_counts[pr_type]), 
            random_state=42
        )
    else:
        # Keep all of 'other' type
        pr_df = df_tech_balanced[df_tech_balanced['pr_type'] == pr_type]
    balanced_pr_dfs.append(pr_df)

df_final = pd.concat(balanced_pr_dfs, ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove the temporary pr_type column
df_final = df_final.drop(columns=['pr_type'])

print(f"\n   PR type distribution after balancing:")
if 'is_bug_fix' in pr_type_cols:
    print(f"   - is_bug_fix=True: {(df_final['is_bug_fix'] == True).sum()}")
if 'is_new_feature' in pr_type_cols:
    print(f"   - is_new_feature=True: {(df_final['is_new_feature'] == True).sum()}")
if 'is_update' in pr_type_cols:
    print(f"   - is_update=True: {(df_final['is_update'] == True).sum()}")
print(f"   Total rows after PR type balancing: {len(df_final)}")

# ============================================================================
# CATEGORICAL ANALYSIS - AFTER
# ============================================================================
print("\n" + "="*70)
print("6. CATEGORICAL ANALYSIS - AFTER")
print("="*70)

# Update categorical columns to include is_fullstack
categorical_cols_after = ['is_bug_fix', 'is_new_feature', 'is_update', 'is_refactor', 
                          'is_backend', 'is_frontend', 'is_fullstack', 'repo_language']
categorical_cols_after = [col for col in categorical_cols_after if col in df_final.columns]

if categorical_cols_after:
    n_cats = len(categorical_cols_after)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols_after):
        ax = axes[idx]
        value_counts = df_final[col].value_counts()
        
        ax.bar(range(len(value_counts)), value_counts.values, color='coral', edgecolor='black')
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
    
    plt.suptitle('Categorical Feature Distributions - AFTER', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('categorical_analysis_after.png', dpi=300, bbox_inches='tight')
    print("✅ Categorical analysis (AFTER) saved as 'categorical_analysis_after.png'")
    
    # Print statistics
    print("\n   Categorical feature distributions (AFTER):")
    for col in categorical_cols_after:
        print(f"\n   {col}:")
        print(df_final[col].value_counts().to_string())

# ============================================================================
# SAVE FINAL DATASET
# ============================================================================
print("\n" + "="*70)
print("7. SAVING FINAL DATASET")
print("="*70)

output_file = 'source_balanced.csv'
df_final.to_csv(output_file, index=False)
print(f"✅ Final balanced dataset saved to '{output_file}'")
print(f"   Final shape: {df_final.shape}")
print(f"   Final columns: {len(df_final.columns)}")

# ============================================================================
# CREATE FINAL DATASET: 30 FRONTEND, 30 BACKEND, 30 FULLSTACK
# ============================================================================
print("\n" + "="*70)
print("8. CREATING FINAL DATASET (30 FRONTEND, 30 BACKEND, 30 FULLSTACK)")
print("="*70)

# Check available counts
frontend_available = (df_final['is_frontend'] == True).sum()
backend_available = (df_final['is_backend'] == True).sum()
fullstack_available = (df_final['is_fullstack'] == True).sum()

print(f"   Available samples:")
print(f"   - Frontend: {frontend_available}")
print(f"   - Backend: {backend_available}")
print(f"   - Fullstack: {fullstack_available}")

target_count = 30

# Sample 30 from each category
final_dfs = []

if frontend_available >= target_count:
    frontend_final = df_final[df_final['is_frontend'] == True].sample(n=target_count, random_state=42)
    final_dfs.append(frontend_final)
    print(f"\n   ✅ Sampled {target_count} frontend PRs")
else:
    print(f"\n   ⚠️  Only {frontend_available} frontend PRs available (need {target_count})")
    if frontend_available > 0:
        final_dfs.append(df_final[df_final['is_frontend'] == True])

if backend_available >= target_count:
    backend_final = df_final[df_final['is_backend'] == True].sample(n=target_count, random_state=42)
    final_dfs.append(backend_final)
    print(f"   ✅ Sampled {target_count} backend PRs")
else:
    print(f"   ⚠️  Only {backend_available} backend PRs available (need {target_count})")
    if backend_available > 0:
        final_dfs.append(df_final[df_final['is_backend'] == True])

if fullstack_available >= target_count:
    fullstack_final = df_final[df_final['is_fullstack'] == True].sample(n=target_count, random_state=42)
    final_dfs.append(fullstack_final)
    print(f"   ✅ Sampled {target_count} fullstack PRs")
else:
    print(f"   ⚠️  Only {fullstack_available} fullstack PRs available (need {target_count})")
    if fullstack_available > 0:
        final_dfs.append(df_final[df_final['is_fullstack'] == True])

df_final_30 = pd.concat(final_dfs, ignore_index=True)
df_final_30 = df_final_30.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n   Final dataset distribution:")
print(f"   - Frontend: {(df_final_30['is_frontend'] == True).sum()}")
print(f"   - Backend: {(df_final_30['is_backend'] == True).sum()}")
print(f"   - Fullstack: {(df_final_30['is_fullstack'] == True).sum()}")
print(f"   Total rows: {len(df_final_30)}")

# Save final dataset
final_output_file = 'source_final.csv'
df_final_30.to_csv(final_output_file, index=False)
print(f"\n✅ Final dataset (30 each) saved to '{final_output_file}'")
print(f"   Final shape: {df_final_30.shape}")

# ============================================================================
# CREATE FINAL DATASET BY PR TYPE: 34 BUG FIX, 34 FEATURE, 34 UPDATE, 34 REFACTOR
# ============================================================================
print("\n" + "="*70)
print("9. CREATING FINAL DATASET BY PR TYPE (34 EACH)")
print("="*70)

# Assign each PR to a single type (priority: bug_fix > feature > update > refactor)
df_final['pr_type_assigned'] = 'other'
df_final.loc[df_final['is_bug_fix'] == True, 'pr_type_assigned'] = 'bug_fix'
df_final.loc[(df_final['is_new_feature'] == True) & (df_final['pr_type_assigned'] == 'other'), 'pr_type_assigned'] = 'feature'
df_final.loc[(df_final['is_update'] == True) & (df_final['pr_type_assigned'] == 'other'), 'pr_type_assigned'] = 'update'
df_final.loc[(df_final['is_refactor'] == True) & (df_final['pr_type_assigned'] == 'other'), 'pr_type_assigned'] = 'refactor'

# Check available counts
pr_type_counts = df_final['pr_type_assigned'].value_counts()
print(f"   Available samples by assigned PR type:")
for pr_type in ['bug_fix', 'feature', 'update', 'refactor']:
    count = pr_type_counts.get(pr_type, 0)
    print(f"   - {pr_type}: {count}")

target_count = 34

# Sample 34 from each PR type category
final_pr_type_dfs = []
used_indices = set()

for pr_type in ['bug_fix', 'feature', 'update', 'refactor']:
    available = df_final[df_final['pr_type_assigned'] == pr_type]
    # Exclude already selected PRs
    available = available[~available.index.isin(used_indices)]
    
    if len(available) >= target_count:
        sampled = available.sample(n=target_count, random_state=42)
        final_pr_type_dfs.append(sampled)
        used_indices.update(sampled.index)
        print(f"\n   ✅ Sampled {target_count} {pr_type} PRs")
    else:
        if len(available) > 0:
            final_pr_type_dfs.append(available)
            used_indices.update(available.index)
            print(f"\n   ⚠️  Only {len(available)} {pr_type} PRs available (need {target_count})")
        else:
            print(f"\n   ⚠️  No {pr_type} PRs available")

df_final_pr_type = pd.concat(final_pr_type_dfs, ignore_index=True)
df_final_pr_type = df_final_pr_type.drop(columns=['pr_type_assigned'])
df_final_pr_type = df_final_pr_type.sample(frac=1, random_state=42).reset_index(drop=True)

# Count by assigned type (not boolean columns which can overlap)
df_final_pr_type['pr_type_count'] = 'other'
df_final_pr_type.loc[df_final_pr_type['is_bug_fix'] == True, 'pr_type_count'] = 'bug_fix'
df_final_pr_type.loc[(df_final_pr_type['is_new_feature'] == True) & (df_final_pr_type['pr_type_count'] == 'other'), 'pr_type_count'] = 'feature'
df_final_pr_type.loc[(df_final_pr_type['is_update'] == True) & (df_final_pr_type['pr_type_count'] == 'other'), 'pr_type_count'] = 'update'
df_final_pr_type.loc[(df_final_pr_type['is_refactor'] == True) & (df_final_pr_type['pr_type_count'] == 'other'), 'pr_type_count'] = 'refactor'

pr_type_dist = df_final_pr_type['pr_type_count'].value_counts()
df_final_pr_type = df_final_pr_type.drop(columns=['pr_type_count'])

print(f"\n   Final dataset distribution by PR type:")
for pr_type in ['bug_fix', 'feature', 'update', 'refactor']:
    count = pr_type_dist.get(pr_type, 0)
    print(f"   - {pr_type}: {count}")
print(f"   Total rows: {len(df_final_pr_type)}")

# Save final dataset by PR type
final_pr_type_output_file = 'source_final.csv'
df_final_pr_type.to_csv(final_pr_type_output_file, index=False)
print(f"\n✅ Final dataset by PR type (34 each) saved to '{final_pr_type_output_file}'")
print(f"   Final shape: {df_final_pr_type.shape}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Initial dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Balanced dataset: {df_final.shape[0]} rows, {df_final.shape[1]} columns")
print(f"Final dataset by tech stack (30 each): {df_final_30.shape[0]} rows, {df_final_30.shape[1]} columns")
print(f"Final dataset by PR type (34 each): {df_final_pr_type.shape[0]} rows, {df_final_pr_type.shape[1]} columns")
print(f"Rows removed from initial: {df.shape[0] - df_final_pr_type.shape[0]}")
print(f"Columns removed: {df.shape[1] - df_final.shape[1]}")
print("\n✅ DATA QUALITY IMPROVEMENT COMPLETE!")
print("="*70)

