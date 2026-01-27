#!/usr/bin/env python3
"""
Fetch PR data from GitHub API given a PR URL and GitHub token.
"""

import requests
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import time


def parse_pr_url(pr_url: str) -> tuple:
    """Parse GitHub PR URL to extract owner, repo, and PR number."""
    # Pattern: https://github.com/{owner}/{repo}/pull/{number}
    pattern = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
    match = re.match(pattern, pr_url)
    if not match:
        raise ValueError(f"Invalid PR URL format: {pr_url}")
    return match.groups()


def parse_iso_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 date string to datetime object."""
    if not date_str:
        return None
    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))


def fetch_paginated(url: str, headers: Dict[str, str]) -> List[Dict]:
    """Fetch paginated GitHub API response."""
    all_data = []
    current_url = url
    while current_url:
        response = requests.get(current_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            all_data.extend(data)
        else:
            all_data.append(data)
        
        # Check for pagination
        link_header = response.headers.get('Link', '')
        next_match = re.search(r'<([^>]+)>; rel="next"', link_header)
        if next_match:
            current_url = next_match.group(1)
            time.sleep(0.1)  # Rate limiting
        else:
            current_url = None
    
    return all_data


def fetch_pr_data(github_token: str, pr_url: str) -> Dict[str, Any]:
    """
    Fetch all PR data from GitHub API.
    
    Args:
        github_token: GitHub personal access token
        pr_url: Full GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)
    
    Returns:
        Dictionary containing PR data, reviews, requested_reviewers, and timeline_events
    """
    owner, repo, pr_number = parse_pr_url(pr_url)
    
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    
    print(f"Fetching PR #{pr_number} from {owner}/{repo}...")
    
    # Fetch PR data
    pr_url_api = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
    response = requests.get(pr_url_api, headers=headers)
    response.raise_for_status()
    pr = response.json()
    print("✅ Fetched PR data")
    
    time.sleep(0.1)
    
    # Fetch reviews
    reviews_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews?per_page=100'
    reviews = fetch_paginated(reviews_url, headers)
    print(f"✅ Fetched {len(reviews)} reviews")
    
    time.sleep(0.1)
    
    # Fetch requested reviewers
    requested_reviewers_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers'
    response = requests.get(requested_reviewers_url, headers=headers)
    response.raise_for_status()
    requested_reviewers = response.json()
    print("✅ Fetched requested reviewers")
    
    time.sleep(0.1)
    
    # Fetch timeline events
    timeline_url = f'https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/timeline?per_page=100'
    timeline_events = fetch_paginated(timeline_url, headers)
    print(f"✅ Fetched {len(timeline_events)} timeline events")
    
    return {
        'pr': pr,
        'reviews': reviews,
        'requested_reviewers': requested_reviewers,
        'timeline_events': timeline_events
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python fetch_pr_data.py <github_token> <pr_url>")
        print("Example: python fetch_pr_data.py ghp_xxx https://github.com/owner/repo/pull/123")
        sys.exit(1)
    
    token = sys.argv[1]
    url = sys.argv[2]
    
    try:
        pr_data = fetch_pr_data(token, url)
        print("\n✅ Successfully fetched PR data!")
        print(f"PR Title: {pr_data['pr'].get('title', 'N/A')}")
        print(f"State: {pr_data['pr'].get('state', 'N/A')}")
        print(f"Reviews: {len(pr_data['reviews'])}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
