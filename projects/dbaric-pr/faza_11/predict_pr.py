#!/usr/bin/env python3
"""
Plug-and-play PR prediction script.
Takes GitHub token and PR URL, fetches data, and predicts PR duration.

Usage:
    python predict_pr.py <github_token> <pr_url>

Example:
    python predict_pr.py ghp_xxx https://github.com/owner/repo/pull/123
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Import our modules
# Try relative imports first (when imported as package), fall back to absolute imports (when run directly)
try:
    from .fetch_pr_data import fetch_pr_data
    from .extract_features import extract_pr_features, parse_iso_date, calculate_duration_hours
except ImportError:
    # Fall back to absolute imports when running directly
    faza_11_dir = os.path.dirname(os.path.abspath(__file__))
    if faza_11_dir not in sys.path:
        sys.path.insert(0, faza_11_dir)
    from fetch_pr_data import fetch_pr_data
    from extract_features import extract_pr_features, parse_iso_date, calculate_duration_hours


def process_features_for_prediction(features_dict: dict, 
                                   imputer, 
                                   label_encoders, 
                                   feature_columns,
                                   numeric_columns,
                                   selected_features) -> pd.DataFrame:
    """
    Process extracted features to match model input format.
    Applies the same preprocessing pipeline as training.
    """
    # Create DataFrame with single row
    df = pd.DataFrame([features_dict])
    
    # Ensure all expected feature columns exist BEFORE selecting
    for col in feature_columns:
        if col not in df.columns:
            # Use appropriate default based on column type
            if 'has_reviewer' in col or 'team' in col or col.endswith('_is_reviewer'):
                df[col] = 0  # Binary features default to 0
            elif col in ['task_id', 'pr_type', 'position']:
                df[col] = None  # Optional fields default to None
            elif col == 'non_working_minutes':
                df[col] = 0  # Will be calculated if PR is closed
            else:
                df[col] = 0  # Numeric features default to 0
    
    # Select only feature columns (now all should exist)
    X = df[feature_columns].copy()
    
    # Handle missing values for numeric columns
    # CRITICAL: Use the exact same numeric columns in the exact same order
    # that the imputer was trained on
    # First ensure all numeric columns exist in X
    for col in numeric_columns:
        if col not in X.columns:
            X[col] = 0  # Default to 0 for missing numeric columns
    
    # Now select numeric columns in exact order
    X_numeric = X[numeric_columns].copy()
    
    # Fill NaN values with 0 (imputer will handle them, but this ensures no issues)
    X_numeric = X_numeric.fillna(0)
    
    # Ensure all values are numeric
    for col in X_numeric.columns:
        X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce').fillna(0)
    
    X_numeric_imputed = pd.DataFrame(
        imputer.transform(X_numeric),
        columns=numeric_columns,  # Use exact column names
        index=X_numeric.index
    )
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    X_encoded = X_numeric_imputed.copy()
    
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            X_col_filled = X[col].fillna('unknown').astype(str)
            # Handle unseen values
            X_col_values = X_col_filled.values
            encoded_values = []
            for val in X_col_values:
                if val in le.classes_:
                    encoded_values.append(le.transform([val])[0])
                else:
                    # Use most common class (index 0) for unseen values
                    encoded_values.append(0)
            X_encoded[col] = encoded_values
        else:
            # If encoder doesn't exist, fill with 0
            X_encoded[col] = 0
    
    X_base = X_encoded.reset_index(drop=True)
    
    # Feature Engineering (same as training)
    X_engineered = X_base.copy()
    
    # Interaction features
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
    
    # Polynomial features
    important_features = ['time_to_first_approval_minutes', 'commits', 'review_count', 
                         'total_lines_changed', 'changed_files']
    for feat in important_features:
        if feat in X_engineered.columns:
            X_engineered[f'{feat}_squared'] = X_engineered[feat] ** 2
    
    # Log transformations
    skewed_features = ['additions', 'deletions', 'total_lines_changed', 'commits', 
                       'review_count', 'comments', 'review_comments']
    for feat in skewed_features:
        if feat in X_engineered.columns:
            X_engineered[f'{feat}_log'] = np.log1p(X_engineered[feat])
    
    # Time-based features
    if 'created_at' in features_dict and features_dict['created_at']:
        created_at_parsed = pd.to_datetime(features_dict['created_at'], errors='coerce')
        if pd.notna(created_at_parsed):
            X_engineered['created_hour'] = created_at_parsed.hour
            X_engineered['created_day_of_week'] = created_at_parsed.dayofweek
            X_engineered['created_is_weekend'] = (X_engineered['created_day_of_week'] >= 5).astype(int)
    else:
        # Default values if created_at not available
        X_engineered['created_hour'] = 12  # Default to noon
        X_engineered['created_day_of_week'] = 1  # Default to Monday
        X_engineered['created_is_weekend'] = 0
    
    # Feature Selection - ensure all selected features exist
    for feat in selected_features:
        if feat not in X_engineered.columns:
            X_engineered[feat] = 0  # Default value for missing engineered features
    
    # Select only the features used by the model
    X_selected = X_engineered[selected_features]
    
    return X_selected


def format_duration(minutes: float) -> str:
    """Format duration in minutes to human-readable string."""
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    elif minutes < 1440:  # Less than a day
        hours = minutes / 60
        return f"{hours:.1f} hours ({minutes:.0f} minutes)"
    else:
        days = minutes / 1440
        hours = (minutes % 1440) / 60
        return f"{days:.1f} days ({hours:.0f} hours, {minutes:.0f} minutes)"


def predict_pr_duration(github_token: str = None, pr_url: str = None, verbose: bool = True) -> dict:
    """
    Predict PR duration using ensemble model.
    
    Args:
        github_token: GitHub token (if None, reads from GITHUB_TOKEN env var or .env)
        pr_url: PR URL to predict
        verbose: Whether to print progress messages
    
    Returns:
        Dictionary with prediction results
    """
    # Get token from parameter, environment variable, or .env file
    if github_token is None:
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token is None:
            raise ValueError("GitHub token not provided. Set GITHUB_TOKEN env var or pass as parameter.")
    
    if pr_url is None:
        raise ValueError("PR URL not provided.")
    
    # Determine script directory - handle both direct execution and import from notebook
    try:
        # Try using __file__ first (works when run directly)
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # When imported from notebook, use inspect to find module location
        import inspect
        script_dir = os.path.dirname(os.path.abspath(inspect.getfile(predict_pr_duration)))
    
    if verbose:
        print("="*70)
        print("PR PREDICTION - PLUG AND PLAY")
        print("="*70)
    
    # Check if models exist
    model_normal_path = os.path.join(script_dir, 'model_normal.pkl')
    model_long_path = os.path.join(script_dir, 'model_long.pkl')
    
    if not os.path.exists(model_normal_path) or not os.path.exists(model_long_path):
        raise FileNotFoundError("Models not found! Please run 'python save_models.py' first.")
    
    # Load models and preprocessing objects
    if verbose:
        print("\n1. Loading models and preprocessing objects...")
    
    with open(model_normal_path, 'rb') as f:
        model_normal = pickle.load(f)
    if verbose:
        print("   ✅ Loaded Normal Model")
    
    with open(model_long_path, 'rb') as f:
        model_long = pickle.load(f)
    if verbose:
        print("   ✅ Loaded Long Model")
    
    with open(os.path.join(script_dir, 'imputer.pkl'), 'rb') as f:
        imputer = pickle.load(f)
    if verbose:
        print("   ✅ Loaded Imputer")
    
    with open(os.path.join(script_dir, 'label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
    if verbose:
        print("   ✅ Loaded Label Encoders")
    
    with open(os.path.join(script_dir, 'feature_selector.pkl'), 'rb') as f:
        feature_selector = pickle.load(f)
    if verbose:
        print("   ✅ Loaded Feature Selector")
    
    with open(os.path.join(script_dir, 'selected_features.pkl'), 'rb') as f:
        selected_features = pickle.load(f)
    if verbose:
        print("   ✅ Loaded Selected Features")
    
    with open(os.path.join(script_dir, 'long_threshold.pkl'), 'rb') as f:
        long_threshold = pickle.load(f)
    if verbose:
        print(f"   ✅ Loaded Long Threshold: {long_threshold:.2f} minutes")
    
    with open(os.path.join(script_dir, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    if verbose:
        print(f"   ✅ Loaded Feature Columns ({len(feature_columns)} columns)")
    
    # Try to load numeric columns, if not found derive from imputer
    numeric_columns_path = os.path.join(script_dir, 'numeric_columns.pkl')
    if os.path.exists(numeric_columns_path):
        with open(numeric_columns_path, 'rb') as f:
            numeric_columns = pickle.load(f)
        if verbose:
            print(f"   ✅ Loaded Numeric Columns ({len(numeric_columns)} columns)")
    else:
        # Backward compatibility: derive numeric columns from imputer feature names
        if verbose:
            print("   ⚠️  numeric_columns.pkl not found, deriving from imputer...")
        if hasattr(imputer, 'feature_names_in_'):
            numeric_columns = list(imputer.feature_names_in_)
        else:
            dummy_df = pd.DataFrame({col: [0] for col in feature_columns})
            numeric_columns = dummy_df.select_dtypes(include=[np.number]).columns.tolist()
        if verbose:
            print(f"   ✅ Derived Numeric Columns ({len(numeric_columns)} columns)")
    
    # Fetch PR data
    if verbose:
        print("\n2. Fetching PR data from GitHub...")
    try:
        pr_data = fetch_pr_data(github_token, pr_url)
        if verbose:
            print("   ✅ PR data fetched successfully")
    except Exception as e:
        if verbose:
            print(f"   ❌ Error fetching PR data: {e}")
        raise
    
    # Extract features
    if verbose:
        print("\n3. Extracting features...")
    try:
        features = extract_pr_features(pr_data)
        if verbose:
            print("   ✅ Features extracted")
    except Exception as e:
        if verbose:
            print(f"   ❌ Error extracting features: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    # Process features
    if verbose:
        print("\n4. Processing features for model input...")
    try:
        X_processed = process_features_for_prediction(
            features, imputer, label_encoders, feature_columns, numeric_columns, selected_features
        )
        if verbose:
            print(f"   ✅ Features processed (shape: {X_processed.shape})")
    except Exception as e:
        if verbose:
            print(f"   ❌ Error processing features: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    # Make prediction
    if verbose:
        print("\n5. Making prediction...")
        print(f"   Input shape: {X_processed.shape}")
        print(f"   Sample feature values:")
        for col in X_processed.columns[:10]:  # Show first 10 features
            print(f"     {col}: {X_processed[col].values[0]:.4f}")
    
    # First, get a preliminary prediction to decide which model to use
    y_pred_log_normal = model_normal.predict(X_processed)[0]
    y_pred_normal = np.expm1(y_pred_log_normal)
    
    if verbose:
        print(f"\n   Normal model log prediction: {y_pred_log_normal:.4f}")
        print(f"   Normal model prediction (minutes): {y_pred_normal:.2f}")
        print(f"   Long threshold: {long_threshold:.2f}")
    
    # Decide which model to use based on threshold
    if y_pred_normal <= long_threshold:
        model_used = "Normal Model"
        y_pred_log = y_pred_log_normal
        y_pred = y_pred_normal
    else:
        model_used = "Long Model"
        y_pred_log = model_long.predict(X_processed)[0]
        y_pred = np.expm1(y_pred_log)
        if verbose:
            print(f"   Long model log prediction: {y_pred_log:.4f}")
            print(f"   Long model prediction (minutes): {y_pred:.2f}")
    
    if verbose:
        print(f"   ✅ Prediction complete (using {model_used})")
        print(f"   Final prediction: {y_pred:.2f} minutes")
    
    # Prepare results
    results = {
        'pr_url': pr_url,
        'pr_number': features.get('pr_number'),
        'title': features.get('title'),
        'predicted_minutes': float(y_pred),
        'predicted_hours': float(y_pred / 60),
        'predicted_duration': format_duration(y_pred),
        'model_used': model_used,
        'features': {k: v for k, v in features.items() if k not in ['title', 'description', 'body']}
    }
    
    # Add actual duration if PR is closed
    if features.get('state') == 'closed' and features.get('merged'):
        created_at = parse_iso_date(features.get('created_at'))
        merged_at = parse_iso_date(features.get('merged_at'))
        if created_at and merged_at:
            duration_hours = calculate_duration_hours(created_at, merged_at)
            if duration_hours:
                actual_minutes = duration_hours * 60
                error = abs(y_pred - actual_minutes)
                error_pct = (error / actual_minutes * 100) if actual_minutes > 0 else 0
                results['actual_minutes'] = float(actual_minutes)
                results['actual_hours'] = float(actual_minutes / 60)
                results['actual_duration'] = format_duration(actual_minutes)
                results['prediction_error_minutes'] = float(error)
                results['prediction_error_percent'] = float(error_pct)
    
    return results


def main():
    # Get token from command line, environment variable, or .env file
    if len(sys.argv) < 2:
        print("Usage: python predict_pr.py <pr_url> [github_token]")
        print("\nExample:")
        print("  python predict_pr.py https://github.com/owner/repo/pull/123")
        print("  python predict_pr.py https://github.com/owner/repo/pull/123 ghp_xxx")
        print("\nNote: If token not provided, reads from GITHUB_TOKEN env var or .env file")
        sys.exit(1)
    
    pr_url = sys.argv[1]
    github_token = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Try to get token from environment or .env
    if github_token is None:
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token is None:
            print("❌ Error: GitHub token not provided!")
            print("   Set GITHUB_TOKEN env var, add to .env file, or pass as argument")
            sys.exit(1)
    
    # Use the predict_pr_duration function
    try:
        results = predict_pr_duration(github_token, pr_url, verbose=True)
        
        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"\nPR Information:")
        print(f"  Title: {results.get('title', 'N/A')}")
        print(f"  Number: {results.get('pr_number', 'N/A')}")
        
        print(f"\nPrediction:")
        print(f"  Model Used: {results['model_used']}")
        print(f"  Predicted Duration: {results['predicted_duration']}")
        print(f"  Predicted Minutes: {results['predicted_minutes']:.1f}")
        print(f"  Predicted Hours: {results['predicted_hours']:.1f}")
        
        if 'actual_minutes' in results:
            print(f"\nActual Duration (for comparison):")
            print(f"  Actual Duration: {results['actual_duration']}")
            print(f"  Actual Minutes: {results['actual_minutes']:.1f}")
            print(f"  Prediction Error: {results['prediction_error_minutes']:.1f} min ({results['prediction_error_percent']:.1f}%)")
        
        print("\n" + "="*70)
        print("✅ PREDICTION COMPLETE")
        print("="*70)
        
        # Save results to JSON
        import json
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, 'prediction_result.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
