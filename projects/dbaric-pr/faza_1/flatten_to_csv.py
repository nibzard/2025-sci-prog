#!/usr/bin/env python3
"""
Script to flatten GitHub PR JSON data into CSV format for ML analysis.
Extracts features for predicting PR duration (time from open to close/merge).
"""

import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional


def parse_iso_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 date string to datetime object."""
    if not date_str:
        return None
    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))


def calculate_duration_hours(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    """Calculate duration in hours between two datetimes."""
    if not start or not end:
        return None
    delta = end - start
    return delta.total_seconds() / 3600.0


def extract_reviewers(pr_data: Dict[str, Any]) -> tuple:
    """Extract unique reviewer logins and count from requested_reviewers and reviews."""
    reviewers = set()
    
    # From requested_reviewers
    if 'requested_reviewers' in pr_data.get('pr', {}):
        requested = pr_data['pr']['requested_reviewers']
        if isinstance(requested, list):
            for reviewer in requested:
                if isinstance(reviewer, dict) and 'login' in reviewer:
                    reviewers.add(reviewer['login'])
        elif isinstance(requested, dict) and 'users' in requested:
            for reviewer in requested['users']:
                if isinstance(reviewer, dict) and 'login' in reviewer:
                    reviewers.add(reviewer['login'])
    
    # From reviews
    if 'reviews' in pr_data:
        for review in pr_data['reviews']:
            if isinstance(review, dict) and 'user' in review:
                user = review['user']
                if isinstance(user, dict) and 'login' in user:
                    reviewers.add(user['login'])
    
    return sorted(list(reviewers)), len(reviewers)


def extract_review_stats(reviews: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract review statistics."""
    stats = {
        'review_count': len(reviews),
        'review_approved_count': 0,
        'review_commented_count': 0,
        'review_changes_requested_count': 0,
        'review_dismissed_count': 0,
    }
    
    for review in reviews:
        if isinstance(review, dict):
            state = review.get('state', '').upper()
            if state == 'APPROVED':
                stats['review_approved_count'] += 1
            elif state == 'COMMENTED':
                stats['review_commented_count'] += 1
            elif state == 'CHANGES_REQUESTED':
                stats['review_changes_requested_count'] += 1
            elif state == 'DISMISSED':
                stats['review_dismissed_count'] += 1
    
    return stats


def find_ready_for_review_time(timeline_events: List[Dict[str, Any]]) -> Optional[str]:
    """Find when PR was marked as ready_for_review (for draft PRs)."""
    for event in timeline_events:
        if isinstance(event, dict) and event.get('event') == 'ready_for_review':
            return event.get('created_at')
    return None


def extract_timeline_stats(timeline_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract statistics from timeline events."""
    stats = {
        'timeline_event_count': len(timeline_events),
        'commit_count_timeline': 0,
        'comment_count_timeline': 0,
        'review_requested_count': 0,
        'ready_for_review_time': None,
    }
    
    for event in timeline_events:
        if isinstance(event, dict):
            event_type = event.get('event', '')
            if event_type == 'committed':
                stats['commit_count_timeline'] += 1
            elif event_type == 'commented':
                stats['comment_count_timeline'] += 1
            elif event_type == 'review_requested':
                stats['review_requested_count'] += 1
            elif event_type == 'ready_for_review':
                stats['ready_for_review_time'] = event.get('created_at')
    
    return stats


def extract_pr_features(pr_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all features from a PR data structure."""
    pr = pr_data.get('pr', {})
    reviews = pr_data.get('reviews', [])
    timeline_events = pr_data.get('timeline_events', [])
    
    # Basic PR info
    pr_number = pr.get('number')
    pr_id = pr.get('id')
    state = pr.get('state', '')
    draft = pr.get('draft', False)
    title = pr.get('title', '')
    body = pr.get('body', '')
    
    # Author
    author_login = ''
    author_id = None
    if 'user' in pr and isinstance(pr['user'], dict):
        author_login = pr['user'].get('login', '')
        author_id = pr['user'].get('id')
    
    # Reviewers
    reviewer_logins, reviewer_count = extract_reviewers(pr_data)
    reviewers_str = '|'.join(reviewer_logins) if reviewer_logins else ''
    
    # Dates
    created_at_str = pr.get('created_at')
    closed_at_str = pr.get('closed_at')
    merged_at_str = pr.get('merged_at')
    updated_at_str = pr.get('updated_at')
    
    created_at = parse_iso_date(created_at_str)
    closed_at = parse_iso_date(closed_at_str)
    merged_at = parse_iso_date(merged_at_str)
    
    # Determine workflow start (for draft PRs, use ready_for_review time)
    workflow_start = created_at
    ready_for_review_time = find_ready_for_review_time(timeline_events)
    if draft and ready_for_review_time:
        workflow_start = parse_iso_date(ready_for_review_time)
    
    # Determine end time (merged_at or closed_at)
    end_time = merged_at if merged_at else closed_at
    
    # Calculate duration
    duration_hours = calculate_duration_hours(workflow_start, end_time)
    duration_days = duration_hours / 24.0 if duration_hours else None
    
    # Code metrics
    additions = pr.get('additions', 0)
    deletions = pr.get('deletions', 0)
    changed_files = pr.get('changed_files', 0)
    commits = pr.get('commits', 0)
    total_lines_changed = additions + deletions
    
    # Activity metrics
    comments = pr.get('comments', 0)
    review_comments = pr.get('review_comments', 0)
    
    # Review statistics
    review_stats = extract_review_stats(reviews)
    
    # Timeline statistics
    timeline_stats = extract_timeline_stats(timeline_events)
    
    # Review timing
    first_review_time = None
    first_approval_time = None
    if reviews:
        review_times = []
        approval_times = []
        for review in reviews:
            if isinstance(review, dict):
                submitted_at = review.get('submitted_at')
                if submitted_at:
                    review_times.append(parse_iso_date(submitted_at))
                    if review.get('state', '').upper() == 'APPROVED':
                        approval_times.append(parse_iso_date(submitted_at))
        
        if review_times:
            first_review_time = min(review_times)
        if approval_times:
            first_approval_time = min(approval_times)
    
    # Calculate time to first review
    time_to_first_review_hours = None
    if workflow_start and first_review_time:
        time_to_first_review_hours = calculate_duration_hours(workflow_start, first_review_time)
    
    # Calculate time to first approval
    time_to_first_approval_hours = None
    if workflow_start and first_approval_time:
        time_to_first_approval_hours = calculate_duration_hours(workflow_start, first_approval_time)
    
    # Repo info
    repo_language = ''
    if 'head' in pr and isinstance(pr['head'], dict):
        repo = pr['head'].get('repo', {})
        if isinstance(repo, dict):
            repo_language = repo.get('language', '')
    
    # Merged by
    merged_by_login = ''
    if 'merged_by' in pr and isinstance(pr['merged_by'], dict):
        merged_by_login = pr['merged_by'].get('login', '')
    
    # Build feature dictionary
    features = {
        # Target variable
        'duration_hours': duration_hours,
        'duration_days': duration_days,
        
        # PR identifiers
        'pr_number': pr_number,
        'pr_id': pr_id,
        'state': state,
        'merged': pr.get('merged', False),
        'draft': draft,
        
        # Dates
        'created_at': created_at_str,
        'closed_at': closed_at_str,
        'merged_at': merged_at_str,
        'updated_at': updated_at_str,
        'ready_for_review_time': ready_for_review_time,
        'workflow_start_time': workflow_start.isoformat() if workflow_start else None,
        
        # Author
        'author_login': author_login,
        'author_id': author_id,
        'merged_by_login': merged_by_login,
        
        # Reviewers
        'reviewer_count': reviewer_count,
        'reviewers': reviewers_str,
        
        # Code metrics
        'additions': additions,
        'deletions': deletions,
        'changed_files': changed_files,
        'commits': commits,
        'total_lines_changed': total_lines_changed,
        
        # Activity metrics
        'comments': comments,
        'review_comments': review_comments,
        
        # Review statistics
        'review_count': review_stats['review_count'],
        'review_approved_count': review_stats['review_approved_count'],
        'review_commented_count': review_stats['review_commented_count'],
        'review_changes_requested_count': review_stats['review_changes_requested_count'],
        'review_dismissed_count': review_stats['review_dismissed_count'],
        
        # Review timing
        'first_review_time': first_review_time.isoformat() if first_review_time else None,
        'first_approval_time': first_approval_time.isoformat() if first_approval_time else None,
        'time_to_first_review_hours': time_to_first_review_hours,
        'time_to_first_approval_hours': time_to_first_approval_hours,
        
        # Timeline statistics
        'timeline_event_count': timeline_stats['timeline_event_count'],
        'commit_count_timeline': timeline_stats['commit_count_timeline'],
        'comment_count_timeline': timeline_stats['comment_count_timeline'],
        'review_requested_count': timeline_stats['review_requested_count'],
        
        # Text features (for NLP analysis)
        'title': title,
        'body': body,
        'title_length': len(title),
        'body_length': len(body) if body else 0,
        
        # Repo info
        'repo_language': repo_language,
    }
    
    return features


def main():
    input_file = 'prs2.json'
    output_file = 'prs_features.csv'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        prs_data = json.load(f)
    
    print(f"Processing {len(prs_data)} PRs...")
    
    # Extract features for all PRs
    all_features = []
    for pr_data in prs_data:
        features = extract_pr_features(pr_data)
        all_features.append(features)
    
    # Get all unique keys (columns)
    if not all_features:
        print("No PRs found!")
        return
    
    columns = list(all_features[0].keys())
    
    # Write to CSV
    print(f"Writing features to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_features)
    
    print(f"Done! Extracted {len(columns)} features for {len(all_features)} PRs")
    print(f"Output saved to {output_file}")
    
    # Print summary statistics
    valid_durations = [f['duration_hours'] for f in all_features if f['duration_hours'] is not None]
    if valid_durations:
        print(f"\nDuration statistics:")
        print(f"  Valid PRs: {len(valid_durations)}")
        print(f"  Min duration: {min(valid_durations):.2f} hours ({min(valid_durations)/24:.2f} days)")
        print(f"  Max duration: {max(valid_durations):.2f} hours ({max(valid_durations)/24:.2f} days)")
        print(f"  Avg duration: {sum(valid_durations)/len(valid_durations):.2f} hours ({sum(valid_durations)/len(valid_durations)/24:.2f} days)")


if __name__ == '__main__':
    main()

