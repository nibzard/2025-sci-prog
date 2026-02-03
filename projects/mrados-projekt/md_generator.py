# md_generator.py
"""
Markdown Generator - Converts scraped Reddit posts to markdown files.
Organizes files by laptop name in subdirectories.
"""

import os
from utils import title_to_slug, get_laptop_markdown_dir


def save_post_as_markdown(
    post: dict, 
    laptop_name: str | None = None, 
    base_dir: str = "knowledge/reddit"
) -> str:
    """
    Convert a scraped Reddit post to a markdown file.
    
    Args:
        post: Dictionary with url, title, body, comments, scraped_at
        laptop_name: Name of the laptop (for organizing in subdirectory)
        base_dir: Base directory for markdown files
        
    Returns:
        Path to the created markdown file
    """
    # Determine output directory
    if laptop_name:
        output_dir = get_laptop_markdown_dir(laptop_name, base_dir)
    else:
        output_dir = base_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename from title
    slug = title_to_slug(post["title"])
    filename = f"{slug}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Build markdown content
    comments_section = ""
    if post.get("comments"):
        comments_list = "\n".join(f"- {comment}" for comment in post["comments"])
        comments_section = f"""

## Comments

{comments_list}
"""
    
    markdown = f"""---
source: reddit
url: {post['url']}
scraped_at: {post['scraped_at']}
laptop: {laptop_name or 'unknown'}
---

# {post['title']}

{post.get('body', '')}
{comments_section}"""

    # Write file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Saved Markdown: {filepath}")
    return filepath


def get_existing_markdown_files(laptop_name: str, base_dir: str = "knowledge/reddit") -> list[str]:
    """
    Get list of existing markdown files for a laptop.
    
    Args:
        laptop_name: Name of the laptop
        base_dir: Base directory for markdown files
        
    Returns:
        List of existing markdown file paths
    """
    output_dir = get_laptop_markdown_dir(laptop_name, base_dir)
    
    if not os.path.exists(output_dir):
        return []
    
    return [
        os.path.join(output_dir, f) 
        for f in os.listdir(output_dir) 
        if f.endswith('.md')
    ]


def main() -> None:
    """Test the markdown generator."""
    test_post = {
        "url": "https://www.reddit.com/r/LenovoLegion/comments/1o7tk8q/my_legion_y540_still_serving_me_after_5_years/",
        "title": "My legion Y540 still serving me after 5 years",
        "body": "Bought this in 2020 during lockdown and have been using it since then, learnt music production on it, blender, unity, programming, video editing. Now I am in my first year college and it's still working smooth as butter.",
        "comments": [
            "Great to hear! I have the same laptop.",
            "The Y540 is a solid machine."
        ],
        "scraped_at": "2026-01-26 18:48:19"
    }
    
    filepath = save_post_as_markdown(test_post, laptop_name="Lenovo Legion Y540")
    print(f"Created: {filepath}")


if __name__ == "__main__":
    main()
