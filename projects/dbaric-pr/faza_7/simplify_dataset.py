import pandas as pd
import numpy as np

print("="*70)
print("SIMPLIFYING DATASET")
print("="*70)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('source_final.csv', dtype={'author_id': str})  # Ensure author_id is string
print(f"   Initial shape: {df.shape}")

# ============================================================================
# CREATE SINGLE PR_TYPE COLUMN
# ============================================================================
print("\n2. Creating single 'pr_type' column...")

# Assign priority: bug_fix > feature > update > refactor
df['pr_type'] = 'other'
df.loc[df['is_bug_fix'] == True, 'pr_type'] = 'fix'
df.loc[(df['is_new_feature'] == True) & (df['pr_type'] == 'other'), 'pr_type'] = 'feature'
df.loc[(df['is_update'] == True) & (df['pr_type'] == 'other'), 'pr_type'] = 'update'
df.loc[(df['is_refactor'] == True) & (df['pr_type'] == 'other'), 'pr_type'] = 'refactor'

print(f"   PR type distribution:")
print(df['pr_type'].value_counts().to_string())

# ============================================================================
# CREATE SINGLE POSITION COLUMN
# ============================================================================
print("\n3. Creating single 'position' column...")

# Assign priority: fullstack > frontend > backend
df['position'] = 'other'
df.loc[df['is_fullstack'] == True, 'position'] = 'fullstack'
df.loc[(df['is_frontend'] == True) & (df['position'] == 'other'), 'position'] = 'frontend'
df.loc[(df['is_backend'] == True) & (df['position'] == 'other'), 'position'] = 'backend'

print(f"   Position distribution:")
print(df['position'].value_counts().to_string())

# ============================================================================
# REMOVE REDUNDANT COLUMNS
# ============================================================================
print("\n4. Removing redundant columns...")

# Remove boolean columns for PR type and position
columns_to_remove = [
    'is_bug_fix',
    'is_new_feature',
    'is_update',
    'is_refactor',
    'is_backend',
    'is_frontend',
    'is_fullstack',
    'repo_language'  # All values are 'TypeScript', so redundant
]

columns_to_remove = [col for col in columns_to_remove if col in df.columns]
print(f"   Removing {len(columns_to_remove)} columns: {columns_to_remove}")

df_simplified = df.drop(columns=columns_to_remove)

print(f"   After removal: {df_simplified.shape}")

# ============================================================================
# VERIFY AUTHOR_ID IS STRING
# ============================================================================
print("\n5. Verifying data types...")
print(f"   author_id dtype: {df_simplified['author_id'].dtype}")
if df_simplified['author_id'].dtype == 'object':
    print("   ✅ author_id is correctly treated as string")
else:
    print("   ⚠️  Converting author_id to string...")
    df_simplified['author_id'] = df_simplified['author_id'].astype(str)

# ============================================================================
# SAVE SIMPLIFIED DATASET
# ============================================================================
print("\n6. Saving simplified dataset...")

output_file = 'source_final.csv'
df_simplified.to_csv(output_file, index=False)
print(f"   ✅ Simplified dataset saved to '{output_file}'")
print(f"   Final shape: {df_simplified.shape}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Initial columns: {len(df.columns)}")
print(f"Final columns: {len(df_simplified.columns)}")
print(f"Columns removed: {len(columns_to_remove)}")
print(f"\nNew columns added:")
print(f"  - pr_type: {df_simplified['pr_type'].value_counts().to_dict()}")
print(f"  - position: {df_simplified['position'].value_counts().to_dict()}")
print("\n✅ DATASET SIMPLIFICATION COMPLETE!")
print("="*70)


