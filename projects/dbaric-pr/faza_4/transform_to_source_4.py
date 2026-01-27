import pandas as pd
import numpy as np

print("Loading source_3.csv...")
df = pd.read_csv('source_3.csv')

print(f"Original shape: {df.shape}")
print(f"Original columns: {len(df.columns)}")

# Columns to remove
cols_to_remove = []

# 1. Remove title_word_count (keep title_length - they're highly correlated 0.862)
if 'title_word_count' in df.columns:
    cols_to_remove.append('title_word_count')
    print("\n✅ Will remove: title_word_count (duplicate of title_length, correlation: 0.862)")

# 2. Remove timeline_event_count (highly correlated with commits and commit_count_timeline)
if 'timeline_event_count' in df.columns:
    cols_to_remove.append('timeline_event_count')
    print("✅ Will remove: timeline_event_count (highly correlated with commits, correlation: 0.964)")

# 3. Remove commit_count_timeline (perfectly correlated with commits, correlation: 1.0)
if 'commit_count_timeline' in df.columns:
    cols_to_remove.append('commit_count_timeline')
    print("✅ Will remove: commit_count_timeline (perfectly correlated with commits, correlation: 1.0)")

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
print("  - title_word_count (duplicate of title_length)")
print("  - timeline_event_count (highly correlated with commits)")
print("  - commit_count_timeline (perfectly correlated with commits)")

print("\nKept columns:")
print("  - title_length (length metric)")
print("  - commits (original commit count)")

print(f"\nSaving to source_4.csv...")
df.to_csv('source_4.csv', index=False)
print("✅ Saved to source_4.csv")


