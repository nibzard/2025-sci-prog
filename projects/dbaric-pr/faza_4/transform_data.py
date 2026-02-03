import pandas as pd
import numpy as np

print("Loading source.csv...")
df = pd.read_csv('source.csv')

print(f"Original shape: {df.shape}")

# Find all columns with "hours" in the name
hours_cols = [col for col in df.columns if 'hours' in col.lower()]
print(f"\nColumns with 'hours': {hours_cols}")

# Convert hours to minutes and rename
for col in hours_cols:
    new_col_name = col.replace('_hours', '_minutes').replace('hours', 'minutes')
    df[new_col_name] = df[col] * 60
    df = df.drop(columns=[col])
    print(f"  Converted: {col} -> {new_col_name} (multiplied by 60)")

# Find all columns with "days" in the name
days_cols = [col for col in df.columns if 'days' in col.lower()]
print(f"\nColumns with 'days': {days_cols}")

# Delete days columns
for col in days_cols:
    df = df.drop(columns=[col])
    print(f"  Deleted: {col}")

print(f"\nNew shape: {df.shape}")
print(f"\nSaving to source_2.csv...")
df.to_csv('source_2.csv', index=False)
print("✅ Saved to source_2.csv")

# Show summary of changes
print("\n" + "="*60)
print("SUMMARY OF CHANGES")
print("="*60)
print(f"Original columns: {len(pd.read_csv('source.csv').columns)}")
print(f"New columns: {len(df.columns)}")
print(f"Columns removed: {len(pd.read_csv('source.csv').columns) - len(df.columns)}")
print(f"\nNew columns with 'minutes':")
minutes_cols = [col for col in df.columns if 'minutes' in col.lower()]
for col in minutes_cols:
    print(f"  - {col}")


