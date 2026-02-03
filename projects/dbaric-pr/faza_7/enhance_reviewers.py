import pandas as pd
import numpy as np

print("="*70)
print("ENHANCING DATASET: REMOVING REDUNDANT COLUMNS & EXTRACTING REVIEWER FEATURES")
print("="*70)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('source_final.csv', dtype={'author_id': str})
print(f"   Initial shape: {df.shape}")

# ============================================================================
# REMOVE author_id (redundant with author_login)
# ============================================================================
print("\n2. Removing redundant author_id column...")
if 'author_id' in df.columns:
    df = df.drop(columns=['author_id'])
    print("   ✅ Removed author_id")

# ============================================================================
# RENAME author_login to author
# ============================================================================
print("\n3. Renaming author_login to author...")
if 'author_login' in df.columns:
    df = df.rename(columns={'author_login': 'author'})
    print("   ✅ Renamed author_login to author")

# ============================================================================
# EXTRACT REVIEWER FEATURES
# ============================================================================
print("\n4. Extracting reviewer features...")

# Get all unique reviewers
all_reviewers = set()
for reviewers_str in df['reviewers'].dropna():
    if pd.notna(reviewers_str) and str(reviewers_str) != '':
        reviewers_list = str(reviewers_str).split('|')
        all_reviewers.update(reviewers_list)

all_reviewers = sorted(list(all_reviewers))
print(f"   Found {len(all_reviewers)} unique reviewers: {all_reviewers}")

# Create binary features for each reviewer
for reviewer in all_reviewers:
    col_name = f'has_reviewer_{reviewer}'
    df[col_name] = df['reviewers'].apply(
        lambda x: reviewer in str(x).split('|') if pd.notna(x) and str(x) != '' else False
    )
    print(f"   ✅ Created {col_name}")

# Create reviewer diversity score (number of unique reviewers)
# This is already captured by reviewer_count, but we can add a feature
# for reviewer team size categories
df['reviewer_team_size'] = df['reviewer_count']
df['reviewer_team_small'] = (df['reviewer_count'] <= 2).astype(int)
df['reviewer_team_medium'] = ((df['reviewer_count'] > 2) & (df['reviewer_count'] <= 4)).astype(int)
df['reviewer_team_large'] = (df['reviewer_count'] > 4).astype(int)

print("   ✅ Created reviewer team size features")

# Create a feature for whether the PR author is also a reviewer (self-review)
df['author_is_reviewer'] = df.apply(
    lambda row: row['author'] in str(row['reviewers']).split('|') 
    if pd.notna(row['reviewers']) and str(row['reviewers']) != '' and pd.notna(row['author']) 
    else False, 
    axis=1
).astype(int)
print("   ✅ Created author_is_reviewer feature")

# ============================================================================
# OPTIONALLY REMOVE ORIGINAL reviewers COLUMN
# ============================================================================
print("\n5. Checking if reviewers column should be kept...")
print("   Keeping 'reviewers' column for reference (can be removed later if needed)")

# ============================================================================
# SAVE ENHANCED DATASET
# ============================================================================
print("\n6. Saving enhanced dataset...")

output_file = 'source_final.csv'
df.to_csv(output_file, index=False)
print(f"   ✅ Enhanced dataset saved to '{output_file}'")
print(f"   Final shape: {df.shape}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Initial columns: {len(df.columns) - len(all_reviewers) - 3}")  # Subtract new columns
print(f"Final columns: {len(df.columns)}")
print(f"New reviewer features: {len(all_reviewers) + 4}")  # binary features + team size features
print(f"\nReviewer binary features created:")
for reviewer in all_reviewers:
    count = df[f'has_reviewer_{reviewer}'].sum()
    print(f"  - has_reviewer_{reviewer}: {count} PRs")
print(f"\nOther reviewer features:")
print(f"  - reviewer_team_small: {df['reviewer_team_small'].sum()} PRs")
print(f"  - reviewer_team_medium: {df['reviewer_team_medium'].sum()} PRs")
print(f"  - reviewer_team_large: {df['reviewer_team_large'].sum()} PRs")
print(f"  - author_is_reviewer: {df['author_is_reviewer'].sum()} PRs")
print("\n✅ DATASET ENHANCEMENT COMPLETE!")
print("="*70)


