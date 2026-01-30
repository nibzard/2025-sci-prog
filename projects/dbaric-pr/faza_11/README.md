# PR Prediction - Plug and Play Solution

A plug-and-play solution to predict PR duration (time to merge) using GitHub token and PR URL.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Save Trained Models

First, you need to save the trained models from `faza_10`. Run:

```bash
python save_models.py
```

This will create the following files:
- `model_normal.pkl` - Model for normal PRs (≤ threshold)
- `model_long.pkl` - Model for long PRs (> threshold)
- `imputer.pkl` - Missing value imputer
- `label_encoders.pkl` - Categorical encoders
- `feature_selector.pkl` - Feature selector
- `selected_features.pkl` - List of selected features
- `long_threshold.pkl` - Threshold for long PRs
- `feature_columns.pkl` - List of feature columns

## Usage

### Basic Usage

```bash
python predict_pr.py <github_token> <pr_url>
```

### Example

```bash
python predict_pr.py ghp_xxxxxxxxxxxx https://github.com/owner/repo/pull/123
```

### Getting a GitHub Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` scope
3. Copy the token and use it in the command

## How It Works

1. **Fetch PR Data**: Fetches PR data, reviews, requested reviewers, and timeline events from GitHub API
2. **Extract Features**: Extracts all relevant features (code metrics, review stats, timing, etc.)
3. **Process Features**: Applies the same preprocessing pipeline as training:
   - Missing value imputation
   - Categorical encoding
   - Feature engineering (ratios, squared terms, log transforms)
   - Feature selection
4. **Make Prediction**: Uses ensemble model (Normal or Long) based on predicted duration
5. **Output Results**: Displays prediction and saves to `prediction_result.json`

## Output

The script outputs:
- PR information (title, number, author, etc.)
- Key features (additions, deletions, commits, reviewers, etc.)
- Predicted duration in human-readable format
- If PR is already closed, shows actual duration and prediction error

Results are also saved to `prediction_result.json`.

## Model Details

The solution uses an **Ensemble Model** with two specialized models:

- **Normal Model**: For PRs with predicted duration ≤ threshold (typically ~2880 minutes)
- **Long Model**: For PRs with predicted duration > threshold

The model automatically selects the appropriate model based on the initial prediction.

## Files

- `save_models.py` - Saves trained models from faza_10
- `fetch_pr_data.py` - Fetches PR data from GitHub API
- `extract_features.py` - Extracts features from PR data
- `predict_pr.py` - Main prediction script
- `requirements.txt` - Python dependencies

## Troubleshooting

### Models not found
Run `python save_models.py` first to save the trained models.

### GitHub API rate limits
The script includes rate limiting delays. If you hit rate limits, wait a few minutes and try again.

### Missing features
If you encounter errors about missing features, ensure the PR has been fetched completely. Some features may be missing for very new PRs.

## Notes

- The model predicts **effective minutes** (excluding non-working hours/weekends/holidays)
- For draft PRs, the workflow start time is when the PR was marked as "ready_for_review"
- The model was trained on merged PRs, so predictions are most accurate for PRs that will be merged
