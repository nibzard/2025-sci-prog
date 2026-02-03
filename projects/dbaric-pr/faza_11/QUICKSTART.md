# Quick Start Guide

## Step 1: Install Dependencies

```bash
cd faza_11
pip install -r requirements.txt
```

## Step 2: Save Models (One-time setup)

```bash
python save_models.py
```

This reads the training data from `../faza_10/source.csv` and saves all necessary model files.

## Step 3: Make Predictions

```bash
python predict_pr.py <your_github_token> <pr_url>
```

### Example

```bash
python predict_pr.py ghp_xxxxxxxxxxxx https://github.com/owner/repo/pull/123
```

## What You Need

1. **GitHub Token**: 
   - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token with `repo` scope
   - Copy the token (starts with `ghp_`)

2. **PR URL**: 
   - Full GitHub PR URL, e.g., `https://github.com/owner/repo/pull/123`

## Output

The script will:
- Fetch PR data from GitHub
- Extract and process features
- Make prediction using the ensemble model
- Display results in the terminal
- Save results to `prediction_result.json`

## Troubleshooting

**Error: Models not found**
→ Run `python save_models.py` first

**Error: Invalid PR URL**
→ Make sure URL format is: `https://github.com/owner/repo/pull/123`

**Error: GitHub API rate limit**
→ Wait a few minutes and try again
