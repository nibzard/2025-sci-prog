# PR Features CSV Documentation

## Overview

The `prs_features.csv` file contains flattened features extracted from GitHub PR data for machine learning analysis. The target variable is **duration_hours** (time from PR open to close/merge).

## Target Variable

- **duration_hours**: Time in hours from workflow start to close/merge
- **duration_days**: Time in days (derived from hours)

**Note**: For draft PRs, workflow start is when the PR was marked as "ready_for_review", not when it was created.

## Feature Categories

### 1. PR Identifiers & Status
- `pr_number`: PR number
- `pr_id`: GitHub PR ID
- `state`: PR state (open/closed)
- `merged`: Boolean - whether PR was merged
- `draft`: Boolean - whether PR was a draft

### 2. Temporal Features
- `created_at`: When PR was created (ISO 8601)
- `closed_at`: When PR was closed (ISO 8601)
- `merged_at`: When PR was merged (ISO 8601)
- `updated_at`: Last update time (ISO 8601)
- `ready_for_review_time`: When draft PR was marked ready (ISO 8601)
- `workflow_start_time`: Actual workflow start time (ISO 8601)

### 3. Author Features
- `author_login`: PR author username
- `author_id`: PR author GitHub ID
- `merged_by_login`: Username who merged the PR

### 4. Reviewer Features
- `reviewer_count`: Number of unique reviewers
- `reviewers`: Pipe-separated list of reviewer usernames (e.g., "user1|user2|user3")

### 5. Code Metrics
- `additions`: Lines added
- `deletions`: Lines deleted
- `changed_files`: Number of files changed
- `commits`: Number of commits
- `total_lines_changed`: additions + deletions

### 6. Activity Metrics
- `comments`: Number of issue comments
- `review_comments`: Number of review comments (inline code comments)

### 7. Review Statistics
- `review_count`: Total number of reviews
- `review_approved_count`: Number of approved reviews
- `review_commented_count`: Number of commented reviews
- `review_changes_requested_count`: Number of change requests
- `review_dismissed_count`: Number of dismissed reviews

### 8. Review Timing
- `first_review_time`: Timestamp of first review (ISO 8601)
- `first_approval_time`: Timestamp of first approval (ISO 8601)
- `time_to_first_review_hours`: Hours from workflow start to first review
- `time_to_first_approval_hours`: Hours from workflow start to first approval

### 9. Timeline Statistics
- `timeline_event_count`: Total number of timeline events
- `commit_count_timeline`: Number of commit events in timeline
- `comment_count_timeline`: Number of comment events in timeline
- `review_requested_count`: Number of review request events

### 10. Text Features (for NLP)
- `title`: PR title (full text)
- `body`: PR description/body (full text)
- `title_length`: Character count of title
- `body_length`: Character count of body

### 11. Repository Features
- `repo_language`: Primary programming language of the repository

## Usage Notes

1. **Missing Values**: Some PRs may have `None` for duration if they're still open or missing timestamps
2. **Reviewers Format**: Reviewers are pipe-separated (`|`) for easy parsing
3. **Text Fields**: Title and body contain full text - may need preprocessing for ML (tokenization, embeddings, etc.)
4. **Categorical Features**: Consider encoding categorical features like `author_login`, `reviewers`, `state`, `repo_language` for ML models

## Statistics

- Total PRs: 233
- Valid durations: 225 (some PRs may be open or missing data)
- Average duration: ~2.36 days (56.73 hours)
- Duration range: 0 to ~39.5 days

## Suggested Feature Engineering

For ML models, consider:

1. **Categorical Encoding**: One-hot or label encoding for authors, reviewers, languages
2. **Text Processing**: 
   - TF-IDF or word embeddings for title/body
   - Extract keywords, checklists, task references
3. **Derived Features**:
   - Reviewer response rate
   - Code review intensity (comments per line)
   - Time of day/week patterns
   - Author experience (historical PR count)
4. **Interaction Features**:
   - Author-reviewer combinations
   - File type changes (if available)
   - PR size categories (small/medium/large)

