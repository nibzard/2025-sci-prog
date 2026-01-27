#!/usr/bin/env python3
"""
Script to clean up PR features CSV data (v2).
Removes template sections, counts media, and extracts structured data.
"""

import csv
import re
from typing import Dict, Any, Tuple


def count_images_and_videos(body: str) -> Tuple[int, int]:
    """Count images and videos in body before removing them."""
    if not body:
        return 0, 0
    
    # Count images: <img> tags and markdown images ![alt](url)
    img_pattern = r'<img[^>]*>|!\[.*?\]\([^\)]+\.(jpg|jpeg|png|gif|svg|webp|bmp)\)'
    images = len(re.findall(img_pattern, body, re.IGNORECASE))
    
    # Count videos: <video> tags and markdown videos ![alt](url) with video extensions
    video_pattern = r'<video[^>]*>|!\[.*?\]\([^\)]+\.(mp4|webm|ogg|mov|avi|mkv)\)'
    videos = len(re.findall(video_pattern, body, re.IGNORECASE))
    
    return images, videos


def remove_images_and_videos(body: str) -> str:
    """Remove all images and videos from body."""
    if not body:
        return ''
    
    # Remove <img> tags
    body = re.sub(r'<img[^>]*>', '', body, flags=re.IGNORECASE)
    
    # Remove <video> tags
    body = re.sub(r'<video[^>]*>.*?</video>', '', body, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove markdown images ![alt](url)
    body = re.sub(r'!\[.*?\]\([^\)]+\)', '', body)
    
    return body


def extract_type_of_change(body: str) -> Dict[str, bool]:
    """Extract type of change from body checkboxes."""
    result = {
        'is_bug_fix': False,
        'is_new_feature': False,
        'is_update': False,
        'is_refactor': False,
    }
    
    if not body:
        return result
    
    # Look for Type of change section
    type_section = re.search(
        r'##\s*Type of change.*?(?=##|\Z)',
        body,
        re.IGNORECASE | re.DOTALL
    )
    
    if type_section:
        section_text = type_section.group(0)
        result['is_bug_fix'] = bool(re.search(r'- \[X\]\s*:bug:', section_text, re.IGNORECASE))
        result['is_new_feature'] = bool(re.search(r'- \[X\]\s*:building_construction:', section_text, re.IGNORECASE))
        result['is_update'] = bool(re.search(r'- \[X\]\s*:wrench:', section_text, re.IGNORECASE))
        result['is_refactor'] = bool(re.search(r'- \[X\]\s*:axe:', section_text, re.IGNORECASE))
    
    return result


def remove_type_of_change_section(body: str) -> str:
    """Remove the entire '## Type of change' section from body."""
    if not body:
        return body
    
    # Remove the entire section including header and all checkboxes
    body = re.sub(
        r'##\s*Type of change.*?(?=##|\Z)',
        '',
        body,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    return body


def extract_apps(body: str) -> Dict[str, bool]:
    """Extract apps affected from body checkboxes."""
    result = {
        'is_backend': False,
        'is_frontend': False,
    }
    
    if not body:
        return result
    
    # Look for Apps section
    apps_section = re.search(
        r'##\s*Apps.*?(?=##|\Z)',
        body,
        re.IGNORECASE | re.DOTALL
    )
    
    if apps_section:
        section_text = apps_section.group(0)
        result['is_backend'] = bool(re.search(r'- \[X\]\s*:back:', section_text, re.IGNORECASE))
        result['is_frontend'] = bool(re.search(r'- \[X\]\s*:front:', section_text, re.IGNORECASE))
    
    return result


def remove_apps_section(body: str) -> str:
    """Remove the entire '## Apps' section from body."""
    if not body:
        return body
    
    # Remove the entire section including header and all checkboxes
    body = re.sub(
        r'##\s*Apps.*?(?=##|\Z)',
        '',
        body,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    return body


def remove_merge_commit_message_section(body: str) -> str:
    """Remove the '## Merge commit message' section."""
    if not body:
        return body
    
    # Remove the entire section - match header and everything until next ## header or end of string
    # Handle both "## Merge commit message" and "##Merge commit message" (no space)
    body = re.sub(
        r'##\s*Merge\s+commit\s+message.*?(?=\n##|\Z)',
        '',
        body,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    
    # Also try a more aggressive pattern that matches the header and following lines
    # until we hit another header or end
    lines = body.split('\n')
    cleaned_lines = []
    skip_until_next_header = False
    
    for line in lines:
        # Check if this is the merge commit message header
        if re.match(r'^\s*##\s*Merge\s+commit\s+message\s*$', line, re.IGNORECASE):
            skip_until_next_header = True
            continue
        
        # If we're skipping, check if we hit another header
        if skip_until_next_header:
            if line.strip().startswith('##'):
                skip_until_next_header = False
                cleaned_lines.append(line)
            # Otherwise skip this line
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def remove_task_lines(body: str) -> str:
    """Remove all 'Task: [' lines."""
    if not body:
        return body
    
    # Remove lines that start with "Task: [" (with optional whitespace)
    lines = body.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that match "Task: [..." pattern
        if re.match(r'^\s*Task:\s*\[', line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def extract_description(body: str) -> str:
    """Extract just the description section (first meaningful content)."""
    if not body:
        return ''
    
    # Try to find Description section
    desc_match = re.search(
        r'#\s*Description\s*\n\s*(.*?)(?=\n##|\Z)',
        body,
        re.IGNORECASE | re.DOTALL
    )
    
    if desc_match:
        desc = desc_match.group(1).strip()
        # Remove empty lines and clean up
        lines = [line.strip() for line in desc.split('\n') if line.strip()]
        desc = ' '.join(lines)
        # Remove template text if still present
        desc = re.sub(r'Summarize what changed.*?', '', desc, flags=re.IGNORECASE | re.DOTALL)
        desc = re.sub(r'Help readers understand.*?', '', desc, flags=re.IGNORECASE | re.DOTALL)
        return desc.strip()
    
    # If no Description section, return first meaningful paragraph
    lines = body.split('\n')
    first_para = []
    for line in lines:
        line = line.strip()
        # Skip empty lines, headers, and checkbox lines
        if not line or line.startswith('#') or line.startswith('- ['):
            if first_para:
                break
            continue
        # Skip if it looks like template text
        if any(template in line.lower() for template in ['summarize', 'help readers', 'always attach']):
            continue
        first_para.append(line)
        # Stop at end of sentence or if we have substantial content
        if (line.endswith('.') or line.endswith('!') or line.endswith('?')) and len(' '.join(first_para)) > 50:
            break
    
    result = ' '.join(first_para).strip()
    # Clean up any remaining template text
    result = re.sub(r'Summarize what changed.*?', '', result, flags=re.IGNORECASE | re.DOTALL)
    return result


def clean_body_text_v2(body: str) -> str:
    """Clean body text by removing template sections and media."""
    if not body:
        return ''
    
    # First, remove template text patterns
    template_patterns = [
        r'Summarize what changed and why\.?\s*Help readers understand the impact of your changes\.?',
        r'Always attach a video or screenshot showing the update\.?\s*If possible, include before and after comparisons\.?',
        r'Briefly summarize what changed in this update using non-technical language\.?\s*Keep it short \(about 1-2 sentences\); this will be shown to customers\.?',
    ]
    
    for pattern in template_patterns:
        body = re.sub(pattern, '', body, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Remove template-like text in description section
    body = re.sub(
        r'#\s*Description\s*\n\s*Summarize what changed and why[^\n]*\n[^\n]*Help readers understand[^\n]*',
        '# Description\n',
        body,
        flags=re.IGNORECASE | re.MULTILINE
    )
    body = re.sub(
        r'Always attach a video or screenshot[^\n]*\n[^\n]*If possible[^\n]*',
        '',
        body,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    # Remove sections (order matters - remove after extraction)
    body = remove_merge_commit_message_section(body)
    body = remove_type_of_change_section(body)
    body = remove_apps_section(body)
    body = remove_task_lines(body)
    
    # Remove images and videos
    body = remove_images_and_videos(body)
    
    # Remove empty sections (headers with no content)
    lines = body.split('\n')
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # If it's a header (starts with #)
        if line.strip().startswith('#'):
            # Check if next non-empty line is also a header or if section is empty
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            # If next non-empty line is a header, this section is empty - skip it
            if j < len(lines) and lines[j].strip().startswith('#'):
                i = j
                continue
        cleaned_lines.append(line)
        i += 1
    
    body = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive empty lines
    body = re.sub(r'\n{3,}', '\n\n', body)
    
    # Remove leading/trailing whitespace
    body = body.strip()
    
    return body


def clean_title(title: str) -> str:
    """Clean up title text."""
    if not title:
        return ''
    
    # Remove extra whitespace
    cleaned = ' '.join(title.split())
    
    return cleaned


def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single row and add cleaned/extracted fields."""
    new_row = row.copy()
    
    # Get original body - prefer body_cleaned, but fall back to body
    # Note: body_cleaned may still have template sections, so we'll clean it again
    original_body = row.get('body_cleaned') or row.get('body', '')
    
    # Count images and videos BEFORE removing them
    images_count, videos_count = count_images_and_videos(original_body)
    new_row['images_count'] = images_count
    new_row['videos_count'] = videos_count
    
    # Extract structured data BEFORE removing sections
    type_of_change = extract_type_of_change(original_body)
    new_row.update(type_of_change)
    
    apps = extract_apps(original_body)
    new_row.update(apps)
    
    # Clean body (removes sections, images, videos, etc.)
    cleaned_body = clean_body_text_v2(original_body)
    new_row['body_cleaned_v2'] = cleaned_body
    new_row['body_length_cleaned_v2'] = len(cleaned_body)
    
    # Extract description from cleaned body
    description = extract_description(cleaned_body)
    new_row['description_v2'] = description
    new_row['description_length_v2'] = len(description)
    
    # Clean title
    new_row['title_cleaned_v2'] = clean_title(row.get('title', ''))
    
    return new_row


def main():
    input_file = 'prs_features_cleaned.csv'
    output_file = 'prs_features_cleaned_v2.csv'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"Processing {len(rows)} rows...")
    
    # Process all rows
    processed_rows = []
    for i, row in enumerate(rows):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(rows)} rows...")
        processed_row = process_row(row)
        # Remove original 'body' and 'body_cleaned' fields (we have body_cleaned_v2)
        if 'body' in processed_row:
            del processed_row['body']
        if 'body_cleaned' in processed_row:
            del processed_row['body_cleaned']
        processed_rows.append(processed_row)
    
    # Determine new fieldnames (add new fields at the end)
    new_fields = [
        'images_count',
        'videos_count',
        'is_bug_fix',
        'is_new_feature',
        'is_update',
        'is_refactor',
        'is_backend',
        'is_frontend',
        'body_cleaned_v2',
        'body_length_cleaned_v2',
        'description_v2',
        'description_length_v2',
        'title_cleaned_v2',
    ]
    
    # Keep original fieldnames, but exclude 'body' and 'body_cleaned' fields (we have body_cleaned_v2)
    fields_to_exclude = {'body', 'body_cleaned'}
    filtered_fieldnames = [f for f in fieldnames if f not in fields_to_exclude]
    output_fieldnames = filtered_fieldnames + new_fields
    
    print(f"Writing cleaned data to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)
    
    print(f"Done! Cleaned data saved to {output_file}")
    print(f"\nNew fields added:")
    for field in new_fields:
        print(f"  - {field}")
    
    # Print statistics
    bodies_cleaned = sum(1 for row in processed_rows if row.get('body_cleaned_v2') != row.get('body_cleaned', row.get('body', '')))
    descriptions_extracted = sum(1 for row in processed_rows if row.get('description_v2'))
    total_images = sum(int(row.get('images_count', 0) or 0) for row in processed_rows)
    total_videos = sum(int(row.get('videos_count', 0) or 0) for row in processed_rows)
    frontend_count = sum(1 for row in processed_rows if str(row.get('is_frontend', '')).lower() == 'true' or row.get('is_frontend') is True)
    backend_count = sum(1 for row in processed_rows if str(row.get('is_backend', '')).lower() == 'true' or row.get('is_backend') is True)
    
    print(f"\nStatistics:")
    print(f"  Bodies cleaned: {bodies_cleaned}/{len(rows)}")
    print(f"  Descriptions extracted: {descriptions_extracted}/{len(rows)}")
    print(f"  Total images found: {total_images}")
    print(f"  Total videos found: {total_videos}")
    print(f"  Frontend PRs: {frontend_count}")
    print(f"  Backend PRs: {backend_count}")


if __name__ == '__main__':
    main()

