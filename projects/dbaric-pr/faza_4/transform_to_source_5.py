import pandas as pd
import numpy as np

print("Loading source_4.csv...")
df = pd.read_csv('source_4.csv')

print(f"Original shape: {df.shape}")
print(f"Original columns: {len(df.columns)}")

# Columns to remove
cols_to_remove = []

# 1. Remove comment_count_timeline (perfectly correlated with comments, correlation: 1.0)
if 'comment_count_timeline' in df.columns:
    cols_to_remove.append('comment_count_timeline')
    print("\n✅ Will remove: comment_count_timeline (perfectly correlated with comments, correlation: 1.0)")

# 2. Remove title_croatian_pct (duplicate/derived from title_croatian_words, correlation: 0.942)
if 'title_croatian_pct' in df.columns:
    cols_to_remove.append('title_croatian_pct')
    print("✅ Will remove: title_croatian_pct (duplicate of title_croatian_words, correlation: 0.942)")

# Remove the columns
for col in cols_to_remove:
    if col in df.columns:
        df = df.drop(columns=[col])

print(f"\nNew shape: {df.shape}")
print(f"New columns: {len(df.columns)}")
print(f"Columns removed: {len(cols_to_remove)}")

print("\n" + "="*60)
print("SUMMARY OF CHANGES")
print("="*60)
print("\nRemoved columns:")
print("  - comment_count_timeline (perfectly correlated with comments)")
print("  - title_croatian_pct (duplicate of title_croatian_words)")

print("\nKept columns:")
print("  - comments (original comment count)")
print("  - title_croatian_words (word count metric)")

print(f"\nSaving to source_5.csv...")
df.to_csv('source_5.csv', index=False)
print("✅ Saved to source_5.csv")


