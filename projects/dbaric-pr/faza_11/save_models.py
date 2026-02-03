#!/usr/bin/env python3
"""
Script to save trained models from faza_10 for use in predictions.
Run this once to save the models, then use predict_pr.py for predictions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import pickle
import os
import sys

# Add parent directory to path to import from faza_10
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'faza_10'))

print("="*70)
print("SAVING TRAINED MODELS")
print("="*70)

# Load data
print("\nLoading data...")
source_path = os.path.join(os.path.dirname(__file__), '..', 'faza_10', 'source.csv')
df = pd.read_csv(source_path)
target_col = 'effective_minutes'
df = df[df[target_col].notna()].copy()
df = df[df[target_col] >= 0].copy()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Columns to exclude
exclude_cols = [
    'non_working_minutes', 'pr_number', 'pr_id',
    'created_at', 'closed_at', 'merged_at', 'updated_at',
    'ready_for_review_time', 'workflow_start_time',
    'first_review_time', 'first_approval_time',
    'title', 'description', 'body', 'author', 'merged_by_login', 'task_id',
]

feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
X = df[feature_cols].copy()
y = df[target_col].copy()
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_numeric = X.select_dtypes(include=[np.number])
X_numeric_imputed = pd.DataFrame(
    imputer.fit_transform(X_numeric),
    columns=X_numeric.columns,
    index=X_numeric.index
)

# Save the exact numeric columns that imputer was trained on
numeric_columns = X_numeric.columns.tolist()

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
X_encoded = X_numeric_imputed.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_col_filled = X[col].fillna('unknown').astype(str)
    X_encoded[col] = le.fit_transform(X_col_filled)
    label_encoders[col] = le

X_base = X_encoded.reset_index(drop=True)
y_base = y.reset_index(drop=True)

# Feature Engineering
X_engineered = X_base.copy()

if 'additions' in X_engineered.columns and 'deletions' in X_engineered.columns:
    X_engineered['additions_deletions_ratio'] = (X_engineered['additions'] + 1) / (X_engineered['deletions'] + 1)
if 'commits' in X_engineered.columns and 'changed_files' in X_engineered.columns:
    X_engineered['commits_per_file'] = X_engineered['commits'] / (X_engineered['changed_files'] + 1)
if 'total_lines_changed' in X_engineered.columns and 'commits' in X_engineered.columns:
    X_engineered['lines_per_commit'] = X_engineered['total_lines_changed'] / (X_engineered['commits'] + 1)
if 'review_count' in X_engineered.columns and 'reviewer_count' in X_engineered.columns:
    X_engineered['reviews_per_reviewer'] = X_engineered['review_count'] / (X_engineered['reviewer_count'] + 1)
if 'time_to_first_review_minutes' in X_engineered.columns and 'time_to_first_approval_minutes' in X_engineered.columns:
    X_engineered['review_to_approval_time'] = np.maximum(0, 
        X_engineered['time_to_first_approval_minutes'] - X_engineered['time_to_first_review_minutes'])

important_features = ['time_to_first_approval_minutes', 'commits', 'review_count', 
                     'total_lines_changed', 'changed_files']
for feat in important_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_squared'] = X_engineered[feat] ** 2

skewed_features = ['additions', 'deletions', 'total_lines_changed', 'commits', 
                   'review_count', 'comments', 'review_comments']
for feat in skewed_features:
    if feat in X_engineered.columns:
        X_engineered[f'{feat}_log'] = np.log1p(X_engineered[feat])

if 'created_at' in df.columns:
    df['created_at_parsed'] = pd.to_datetime(df['created_at'], errors='coerce')
    X_engineered['created_hour'] = df['created_at_parsed'].dt.hour
    X_engineered['created_day_of_week'] = df['created_at_parsed'].dt.dayofweek
    X_engineered['created_is_weekend'] = (X_engineered['created_day_of_week'] >= 5).astype(int)

# Feature Selection
n_features_to_select = min(40, X_engineered.shape[1])
selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
X_selected = selector.fit_transform(X_engineered, y_base)
selected_features = X_engineered.columns[selector.get_support()].tolist()
X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X_engineered.index)

# Target Transformation
y_log = np.log1p(y_base)

# Train-Test Split
X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
    X_selected_df, y_base, y_log, test_size=0.2, random_state=123, shuffle=True
)

# Define Long segment threshold (75th percentile)
long_threshold = y_train.quantile(0.75)
print(f"\nLong PR threshold: {long_threshold:.2f} minutes")

# Train Normal Model
print("\nTraining Normal Model...")
train_normal_mask = y_train <= long_threshold
X_train_normal = X_train[train_normal_mask]
y_train_normal = y_train[train_normal_mask]
y_train_normal_log = y_train_log[train_normal_mask]

model_normal = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model_normal.fit(X_train_normal, y_train_normal_log)
print("✅ Normal Model trained")

# Train Long Model
print("\nTraining Long Model...")
train_long_mask = y_train > long_threshold
X_train_long = X_train[train_long_mask]
y_train_long = y_train[train_long_mask]
y_train_long_log = y_train_log[train_long_mask]

model_long = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    reg_alpha=0.15,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model_long.fit(X_train_long, y_train_long_log)
print("✅ Long Model trained")

# Save models and preprocessing objects
output_dir = os.path.dirname(__file__)
print(f"\nSaving models to {output_dir}...")

with open(os.path.join(output_dir, 'model_normal.pkl'), 'wb') as f:
    pickle.dump(model_normal, f)
print("✅ Saved model_normal.pkl")

with open(os.path.join(output_dir, 'model_long.pkl'), 'wb') as f:
    pickle.dump(model_long, f)
print("✅ Saved model_long.pkl")

with open(os.path.join(output_dir, 'imputer.pkl'), 'wb') as f:
    pickle.dump(imputer, f)
print("✅ Saved imputer.pkl")

with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)
print("✅ Saved label_encoders.pkl")

with open(os.path.join(output_dir, 'feature_selector.pkl'), 'wb') as f:
    pickle.dump(selector, f)
print("✅ Saved feature_selector.pkl")

with open(os.path.join(output_dir, 'selected_features.pkl'), 'wb') as f:
    pickle.dump(selected_features, f)
print("✅ Saved selected_features.pkl")

with open(os.path.join(output_dir, 'long_threshold.pkl'), 'wb') as f:
    pickle.dump(long_threshold, f)
print("✅ Saved long_threshold.pkl")

# Save feature columns for reference
with open(os.path.join(output_dir, 'feature_columns.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)
print("✅ Saved feature_columns.pkl")

# Save numeric columns that imputer was trained on
with open(os.path.join(output_dir, 'numeric_columns.pkl'), 'wb') as f:
    pickle.dump(numeric_columns, f)
print("✅ Saved numeric_columns.pkl")

print("\n" + "="*70)
print("✅ ALL MODELS SAVED SUCCESSFULLY!")
print("="*70)
print(f"\nModels saved in: {output_dir}")
print(f"Long PR threshold: {long_threshold:.2f} minutes")
print("\nYou can now use predict_pr.py to make predictions!")
