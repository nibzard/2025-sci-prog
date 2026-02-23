# utils.py
"""
Utility functions for the laptop analysis pipeline.
Provides consistent slug generation and file path handling.
"""

import re
import os


def laptop_name_to_slug(laptop_name: str) -> str:
    """
    Convert laptop name to a filesystem-safe slug.
    
    Rules:
    - lowercase
    - spaces -> underscores
    - remove special chars (keep alphanumeric and underscores)
    
    Example:
        "Lenovo Legion Y540" -> "lenovo_legion_y540"
        "MacBook M4 Pro" -> "macbook_m4_pro"
    """
    text = laptop_name.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove special chars
    text = re.sub(r"\s+", "_", text)          # spaces to underscores
    text = re.sub(r"_+", "_", text)           # collapse multiple underscores
    return text.strip("_")


def title_to_slug(title: str) -> str:
    """
    Convert post title to a filesystem-safe slug for markdown files.
    
    Uses hyphens (more readable for markdown filenames).
    """
    text = title.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")[:100]  # limit length


def get_search_results_path(laptop_name: str, data_dir: str = "data") -> str:
    """
    Get the path for search results JSON file.
    
    Example:
        "Lenovo Legion Y540" -> "data/lenovo_legion_y540_search_results.json"
    """
    slug = laptop_name_to_slug(laptop_name)
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"{slug}_search_results.json")


def get_laptop_markdown_dir(laptop_name: str, base_dir: str = "knowledge/reddit") -> str:
    """
    Get the directory path for a laptop's markdown files.
    
    Example:
        "Lenovo Legion Y540" -> "knowledge/reddit/lenovo_legion_y540/"
    """
    slug = laptop_name_to_slug(laptop_name)
    path = os.path.join(base_dir, slug)
    os.makedirs(path, exist_ok=True)
    return path


def get_scraped_urls_path(laptop_name: str, data_dir: str = "data") -> str:
    """
    Get the path for tracking scraped URLs.
    
    Example:
        "Lenovo Legion Y540" -> "data/lenovo_legion_y540_scraped_urls.json"
    """
    slug = laptop_name_to_slug(laptop_name)
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"{slug}_scraped_urls.json")


def get_analysis_output_path(laptop_name: str, output_dir: str = "analysis") -> str:
    """
    Get the path for analysis output JSON file.
    
    Example:
        "Lenovo Legion Y540" -> "analysis/lenovo_legion_y540_analysis.json"
    """
    slug = laptop_name_to_slug(laptop_name)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{slug}_analysis.json")
