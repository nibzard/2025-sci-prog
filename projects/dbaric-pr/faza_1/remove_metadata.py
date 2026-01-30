#!/usr/bin/env python3
"""
Script to remove metadata fields from GitHub PR JSON data.
Removes fields like issue_url, gravatar_id, node_id, API URLs, etc.
"""

import json
import sys


# Fields to remove - metadata that's not essential PR data
METADATA_FIELDS = {
    # API URLs
    'url',
    'issue_url',
    'diff_url',
    'patch_url',
    'commits_url',
    'review_comments_url',
    'review_comment_url',
    'comments_url',
    'statuses_url',
    'html_url',
    
    # GitHub internal IDs
    'node_id',
    'gravatar_id',
    
    # Link objects
    '_links',
    
    # User metadata URLs
    'followers_url',
    'following_url',
    'gists_url',
    'starred_url',
    'subscriptions_url',
    'organizations_url',
    'repos_url',
    'events_url',
    'received_events_url',
    
    # User metadata flags
    'site_admin',
    'user_view_type',
    'type',
    
    # Other metadata
    'performed_via_github_app',
    'author_association',
    
    # Repo metadata URLs
    'forks_url',
    'keys_url',
    'collaborators_url',
    'teams_url',
    'hooks_url',
    'issue_events_url',
    'assignees_url',
    'branches_url',
    'tags_url',
    'blobs_url',
    'git_tags_url',
    'git_refs_url',
    'trees_url',
    'languages_url',
    'stargazers_url',
    'contributors_url',
    'subscribers_url',
    'subscription_url',
    'git_commits_url',
    'comments_url',
    'issue_comment_url',
    'contents_url',
    'compare_url',
    'merges_url',
    'archive_url',
    'downloads_url',
    'issues_url',
    'pulls_url',
    'milestones_url',
    'notifications_url',
    'labels_url',
    'releases_url',
    'deployments_url',
    'git_url',
    'ssh_url',
    'clone_url',
    'svn_url',
    
    # Commit metadata URLs
    'pull_request_url',
}


def remove_metadata(obj):
    """
    Recursively remove metadata fields from a JSON object.
    """
    if isinstance(obj, dict):
        # Create a new dict without metadata fields
        cleaned = {}
        for key, value in obj.items():
            if key not in METADATA_FIELDS:
                cleaned[key] = remove_metadata(value)
        return cleaned
    elif isinstance(obj, list):
        # Recursively process each item in the list
        return [remove_metadata(item) for item in obj]
    else:
        # Return primitive values as-is
        return obj


def main():
    input_file = 'prs.json'
    output_file = 'prs2.json'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} PRs...")
    cleaned_data = remove_metadata(data)
    
    print(f"Writing cleaned data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Cleaned data saved to {output_file}")


if __name__ == '__main__':
    main()

