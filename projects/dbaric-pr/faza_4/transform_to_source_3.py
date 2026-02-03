import pandas as pd
import numpy as np

print("Loading source_2.csv...")
df = pd.read_csv('source_2.csv')

print(f"Original shape: {df.shape}")
print(f"Original columns: {len(df.columns)}")

# 1. Remove duplicate length columns
# Keep description_length, remove body_length (they're duplicates)
if 'body_length' in df.columns:
    df = df.drop(columns=['body_length'])
    print("\n✅ Removed: body_length (duplicate of description_length)")

# For title, we have title_length and title_word_count - they might be different
# But user said "same for title", so let's check if they're highly correlated
# Actually, let's keep title_length and remove title_word_count if it exists as duplicate
# But I think title_length is character count and title_word_count is word count, so they're different
# Let me keep both for now unless user specifies

# 2. Language columns - keep only Croatian metrics, remove English
# Keep Croatian percentage (or count), remove English ones
language_cols_to_remove = [
    'description_english_words',
    'description_english_pct',
    'title_english_words',
    'title_english_pct'
]

for col in language_cols_to_remove:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"✅ Removed: {col}")

# Keep Croatian columns:
# - description_croatian_words
# - description_croatian_pct
# - title_croatian_words
# - title_croatian_pct

# 3. Remove duration_minutes (total time), keep only effective_minutes and non_working_minutes
if 'duration_minutes' in df.columns:
    df = df.drop(columns=['duration_minutes'])
    print("\n✅ Removed: duration_minutes (keeping only effective_minutes and non_working_minutes)")

print(f"\nNew shape: {df.shape}")
print(f"New columns: {len(df.columns)}")
print(f"Columns removed: {len(pd.read_csv('source_2.csv').columns) - len(df.columns)}")

print("\n" + "="*60)
print("SUMMARY OF CHANGES")
print("="*60)
print("\nRemoved columns:")
print("  - body_length (duplicate of description_length)")
print("  - description_english_words")
print("  - description_english_pct")
print("  - title_english_words")
print("  - title_english_pct")
print("  - duration_minutes (total time)")

print("\nKept columns:")
print("  - description_length (length metric)")
print("  - title_length (length metric)")
print("  - description_croatian_words, description_croatian_pct (Croatian language metrics)")
print("  - title_croatian_words, title_croatian_pct (Croatian language metrics)")
print("  - effective_minutes, non_working_minutes (time metrics)")

print(f"\nSaving to source_3.csv...")
df.to_csv('source_3.csv', index=False)
print("✅ Saved to source_3.csv")


