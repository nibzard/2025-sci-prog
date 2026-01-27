#!/usr/bin/env python3
"""
Script to clean up PR features CSV data.
Removes template text, extracts unique content, and performs general data cleanup.
"""

import csv
import re
from typing import Dict, Any


# Template patterns to remove from body
TEMPLATE_PATTERNS = [
    r'Summarize what changed and why\.?\s*Help readers understand the impact of your changes\.?',
    r'Always attach a video or screenshot showing the update\.?\s*If possible, include before and after comparisons\.?',
    r'Briefly summarize what changed in this update using non-technical language\.?\s*Keep it short \(about 1-2 sentences\); this will be shown to customers\.?',
    r'Task: \[TASK_ID\]\(https://trello\.com/c/TASK_ID\)',
    r'Task: \[.*?\]\(https://trello\.com/c/\)',  # Empty task ID
    r'## Type of change\s*\n\s*- \[ \] :bug: Bug fix\s*\n\s*- \[ \] :building_construction: New feature\s*\n\s*- \[ \] :wrench: Update to an existing feature\s*\n\s*- \[ \] :axe: Refactor / Cleanup',
    r'## Apps\s*\n\s*- \[ \] :back: Backend\s*\n\s*- \[ \] :front: Frontend',
    r'# Checklist:\s*\n\s*- \[ \] Self code review\s*\n\s*- \[ \] Self Q&A test\s*\n\s*- \[ \] Tests for new or changed hooks, components, forms, and other objects\s*\n\s*- \[ \] Mobile layout',
]


def clean_body_text(body: str) -> str:
    """Remove template text and clean up body content."""
    if not body:
        return ''
    
    # Remove template patterns
    cleaned = body
    for pattern in TEMPLATE_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Remove template-like text in description section
    cleaned = re.sub(
        r'#\s*Description\s*\n\s*Summarize what changed and why[^\n]*\n[^\n]*Help readers understand[^\n]*',
        '# Description\n',
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE
    )
    cleaned = re.sub(
        r'Always attach a video or screenshot[^\n]*\n[^\n]*If possible[^\n]*',
        '',
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    # Remove empty sections (headers with no content)
    lines = cleaned.split('\n')
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
    
    cleaned = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive empty lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


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
        r'## Type of change.*?(?=##|\Z)',
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
        r'## Apps.*?(?=##|\Z)',
        body,
        re.IGNORECASE | re.DOTALL
    )
    
    if apps_section:
        section_text = apps_section.group(0)
        result['is_backend'] = bool(re.search(r'- \[X\]\s*:back:', section_text, re.IGNORECASE))
        result['is_frontend'] = bool(re.search(r'- \[X\]\s*:front:', section_text, re.IGNORECASE))
    
    return result


def extract_task_id(body: str) -> str:
    """Extract Trello task ID from body."""
    if not body:
        return ''
    
    # Look for Task: [ID](url) pattern
    match = re.search(r'Task:\s*\[([^\]]+)\]\(https://trello\.com/c/[^\)]+\)', body, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return ''


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
        # Remove images and links that are just placeholders
        desc = re.sub(r'<img[^>]*>', '', desc)
        desc = re.sub(r'!\[.*?\]\(.*?\)', '', desc)
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
    
    # Clean body
    original_body = row.get('body', '')
    cleaned_body = clean_body_text(original_body)
    new_row['body_cleaned'] = cleaned_body
    new_row['body_length_cleaned'] = len(cleaned_body)
    
    # Extract description only
    description = extract_description(cleaned_body)
    new_row['description'] = description
    new_row['description_length'] = len(description)
    
    # Extract structured data
    type_of_change = extract_type_of_change(original_body)
    new_row.update(type_of_change)
    
    apps = extract_apps(original_body)
    new_row.update(apps)
    
    task_id = extract_task_id(original_body)
    new_row['task_id'] = task_id
    
    # Clean title
    new_row['title_cleaned'] = clean_title(row.get('title', ''))
    
    return new_row


def main():
    input_file = 'prs_features.csv'
    output_file = 'prs_features_cleaned.csv'
    
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
        processed_rows.append(processed_row)
    
    # Determine new fieldnames (add new fields at the end)
    new_fields = [
        'body_cleaned',
        'body_length_cleaned',
        'description',
        'description_length',
        'is_bug_fix',
        'is_new_feature',
        'is_update',
        'is_refactor',
        'is_backend',
        'is_frontend',
        'task_id',
        'title_cleaned',
    ]
    
    # Keep original fieldnames, add new ones
    output_fieldnames = list(fieldnames) + new_fields
    
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
    bodies_cleaned = sum(1 for row in processed_rows if row.get('body_cleaned') != row.get('body', ''))
    descriptions_extracted = sum(1 for row in processed_rows if row.get('description'))
    task_ids_found = sum(1 for row in processed_rows if row.get('task_id'))
    
    print(f"\nStatistics:")
    print(f"  Bodies cleaned: {bodies_cleaned}/{len(rows)}")
    print(f"  Descriptions extracted: {descriptions_extracted}/{len(rows)}")
    print(f"  Task IDs found: {task_ids_found}/{len(rows)}")


if __name__ == '__main__':
    main()

