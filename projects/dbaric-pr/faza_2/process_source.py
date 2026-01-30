import pandas as pd
from datetime import datetime, timedelta
import pytz

# Croatian holidays for 2024-2025 (common holidays)
CROATIAN_HOLIDAYS = [
    # 2024
    datetime(2024, 1, 1),   # New Year's Day
    datetime(2024, 1, 6),   # Epiphany
    datetime(2024, 3, 31),  # Easter Monday
    datetime(2024, 5, 1),   # Labour Day
    datetime(2024, 5, 30),  # Statehood Day
    datetime(2024, 6, 20),  # Corpus Christi
    datetime(2024, 6, 22),  # Anti-Fascist Struggle Day
    datetime(2024, 8, 5),   # Victory and Homeland Thanksgiving Day
    datetime(2024, 8, 15),   # Assumption of Mary
    datetime(2024, 10, 8),  # Independence Day
    datetime(2024, 11, 1),   # All Saints' Day
    datetime(2024, 11, 18), # Remembrance Day
    datetime(2024, 12, 25), # Christmas
    datetime(2024, 12, 26), # St. Stephen's Day
    # 2025
    datetime(2025, 1, 1),   # New Year's Day
    datetime(2025, 1, 6),   # Epiphany
    datetime(2025, 4, 21),  # Easter Monday
    datetime(2025, 5, 1),   # Labour Day
    datetime(2025, 5, 30),  # Statehood Day
    datetime(2025, 6, 19),  # Corpus Christi
    datetime(2025, 6, 22),  # Anti-Fascist Struggle Day
    datetime(2025, 8, 5),   # Victory and Homeland Thanksgiving Day
    datetime(2025, 8, 15),  # Assumption of Mary
    datetime(2025, 10, 8),  # Independence Day
    datetime(2025, 11, 1),  # All Saints' Day
    datetime(2025, 11, 18), # Remembrance Day
    datetime(2025, 12, 25), # Christmas
    datetime(2025, 12, 26), # St. Stephen's Day
]

def is_weekend(date):
    """Check if date is a weekend (Saturday or Sunday)"""
    return date.weekday() >= 5

def is_holiday(date):
    """Check if date is a Croatian holiday"""
    return date.date() in [h.date() for h in CROATIAN_HOLIDAYS]

def is_non_working_day(date):
    """Check if date is a non-working day (weekend or holiday)"""
    return is_weekend(date) or is_holiday(date)

def calculate_non_working_time(created_at_str, closed_at_str):
    """
    Calculate hours and days spent in non-working days (weekends and holidays)
    for Croatia timezone.
    """
    if pd.isna(created_at_str) or pd.isna(closed_at_str):
        return 0.0, 0.0
    
    # Parse timestamps (they're in UTC)
    created_at = pd.to_datetime(created_at_str)
    closed_at = pd.to_datetime(closed_at_str)
    
    # Convert to Croatia timezone (CET/CEST)
    croatia_tz = pytz.timezone('Europe/Zagreb')
    created_at_croatia = created_at.tz_localize('UTC').astimezone(croatia_tz) if created_at.tz is None else created_at.astimezone(croatia_tz)
    closed_at_croatia = closed_at.tz_localize('UTC').astimezone(croatia_tz) if closed_at.tz is None else closed_at.astimezone(croatia_tz)
    
    # Calculate total duration
    total_duration = closed_at_croatia - created_at_croatia
    total_hours = total_duration.total_seconds() / 3600.0
    
    # Count non-working hours
    non_working_hours = 0.0
    current = created_at_croatia
    
    # Iterate through each hour in the duration
    while current < closed_at_croatia:
        next_hour = min(current + timedelta(hours=1), closed_at_croatia)
        
        # Check if this hour falls on a non-working day
        if is_non_working_day(current):
            # Calculate how many hours in this period are non-working
            hour_duration = (next_hour - current).total_seconds() / 3600.0
            non_working_hours += hour_duration
        
        current = next_hour
    
    non_working_days = non_working_hours / 24.0
    
    return non_working_hours, non_working_days

def get_base_column_name(col_name):
    """Get base column name by removing _v2 and _cleaned suffixes"""
    name = col_name
    if name.endswith('_v2'):
        name = name[:-3]
    if name.endswith('_cleaned'):
        name = name[:-8]
    return name

def ensure_boolean_prefix(col_name):
    """Ensure boolean columns have 'is_' prefix if they don't already"""
    # Known boolean columns that should have is_ prefix
    boolean_columns = ['draft', 'merged']
    
    if col_name in boolean_columns and not col_name.startswith('is_'):
        return f'is_{col_name}'
    
    return col_name

def get_column_base_name(col_name):
    """Get base column name, removing pandas .1/.2 suffixes and _v2/_cleaned suffixes"""
    name = col_name
    # Remove pandas duplicate suffixes (.1, .2, etc.)
    if '.' in name:
        parts = name.rsplit('.', 1)
        if parts[1].isdigit():
            name = parts[0]
    # Remove _v2 and _cleaned
    name = get_base_column_name(name)
    return name

def prioritize_columns(columns):
    """
    When columns have the same base name (after removing all suffixes),
    keep the _v2 version if it exists, otherwise _cleaned, otherwise the one with higher suffix.
    """
    base_to_cols = {}
    
    # Group columns by their final base name
    for col in columns:
        base = get_column_base_name(col)
        if base not in base_to_cols:
            base_to_cols[base] = []
        base_to_cols[base].append(col)
    
    # For each base name, choose the best column
    selected_cols = []
    for base, cols in base_to_cols.items():
        if len(cols) == 1:
            selected_cols.append(cols[0])
        else:
            # Priority: _v2 > _cleaned > higher pandas suffix (.1 > .0)
            v2_cols = [c for c in cols if '_v2' in c or c.endswith('_v2')]
            cleaned_cols = [c for c in cols if '_cleaned' in c or c.endswith('_cleaned')]
            
            if v2_cols:
                # Among _v2 columns, prefer the one with highest pandas suffix
                selected_cols.append(max(v2_cols, key=lambda x: get_pandas_suffix_num(x)))
            elif cleaned_cols:
                selected_cols.append(max(cleaned_cols, key=lambda x: get_pandas_suffix_num(x)))
            else:
                # Keep the one with highest pandas suffix (most recent duplicate)
                selected_cols.append(max(cols, key=lambda x: get_pandas_suffix_num(x)))
    
    return selected_cols

def get_pandas_suffix_num(col_name):
    """Get numeric suffix from pandas (.1 -> 1, .2 -> 2, no suffix -> 0)"""
    if '.' in col_name:
        parts = col_name.rsplit('.', 1)
        if parts[1].isdigit():
            return int(parts[1])
    return 0

# Read the CSV
df = pd.read_csv('source.csv')

# Filter out rows that are not merged/closed
df = df[(df['state'] == 'closed') & (df['merged'] == True)]

# Calculate non-working time for each PR
non_working_hours_list = []
non_working_days_list = []

for idx, row in df.iterrows():
    non_working_hours, non_working_days = calculate_non_working_time(
        row['created_at'], 
        row['closed_at']
    )
    non_working_hours_list.append(non_working_hours)
    non_working_days_list.append(non_working_days)

# Add new columns
df['non_working_hours'] = non_working_hours_list
df['non_working_days'] = non_working_days_list

# Calculate effective hours and days
df['effective_hours'] = df['duration_hours'] - df['non_working_hours']
df['effective_days'] = df['duration_days'] - df['non_working_days']

# Select columns to keep (prioritize _v2, then _cleaned, then original)
columns_to_keep = prioritize_columns(df.columns)
df = df[columns_to_keep]

# Clean column names: remove pandas suffixes, then _v2 and _cleaned prefixes
def clean_final_column_name(col_name):
    """Remove pandas .1/.2 suffixes and _v2/_cleaned suffixes"""
    name = col_name
    # Remove pandas duplicate suffixes (.1, .2, etc.)
    if '.' in name:
        parts = name.rsplit('.', 1)
        if parts[1].isdigit():
            name = parts[0]
    # Remove _v2 and _cleaned
    name = get_base_column_name(name)
    return name

df.columns = [clean_final_column_name(col) for col in df.columns]

# Ensure boolean columns have is_ prefix
df.columns = [ensure_boolean_prefix(col) for col in df.columns]

# Save to source_v2.csv
df.to_csv('source_v2.csv', index=False)

print(f"Processed {len(df)} rows. Output saved to source_v2.csv")

