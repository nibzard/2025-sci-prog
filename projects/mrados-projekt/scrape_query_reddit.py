# scrape_query_reddit.py
"""
Reddit Search Scraper - Searches Reddit for posts about a specific laptop.
Saves results with dynamic filenames based on laptop name.
Skips scraping if data already exists.
"""

from urllib.parse import quote_plus
import requests
import json
import csv
import time
import os
from bs4 import BeautifulSoup
from utils import laptop_name_to_slug, get_search_results_path


def generate_search_terms(laptop_name: str) -> list[str]:
    """
    Generate search terms for a laptop.
    
    Example:
        "Lenovo Legion Y540" -> [
            "lenovo legion y540",
            "lenovo legion y540 review",
            "lenovo legion y540 reddit"
        ]
    """
    base = laptop_name.lower()
    return [
        base,
        f"{base} review",
        f"{base} reddit"
    ]


def scrape_reddit_search(laptop_name: str) -> list[dict]:
    """
    Scrape Reddit search results for a specific laptop.
    
    Args:
        laptop_name: Name of the laptop (e.g., "Lenovo Legion Y540")
        
    Returns:
        List of search result blocks with query, results, and metadata
    """
    search_terms = generate_search_terms(laptop_name)
    BASE_SEARCH_URL = "https://www.reddit.com/search/?q="

    headers = {
        'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    all_data = []
    for query in search_terms:
        print(f'Scraping reddit_query: {query}')
        encoded_query = quote_plus(query)
        search_url = BASE_SEARCH_URL + encoded_query
        print(f'Search URL: {search_url}')

        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            posts = []
            seen_urls = set()

            for link in soup.find_all('a', href=True):
                href = link['href']
                title = link.get_text(strip=True)
                if ("/comments/" in href
                    and title
                    and href not in seen_urls):
                    full_url = href if href.startswith("http") else "https://www.reddit.com" + href
                    posts.append({
                        "title": title,
                        "url": full_url,
                        "source": "reddit_search"
                    })
                    seen_urls.add(href)
                    
            all_data.append({
                "query": query,
                "search_url": search_url,
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "results": posts
            })

            time.sleep(2)  # Rate limiting

        except Exception as e:
            print(f"Error fetching {query}: {e}")

    return all_data


def save_search_results(data: list[dict], laptop_name: str, data_dir: str = "data") -> str:
    """
    Save search results to JSON and CSV files with dynamic filenames.
    
    Args:
        data: Search results data
        laptop_name: Name of the laptop (for filename generation)
        data_dir: Directory to save files in
        
    Returns:
        Path to the saved JSON file
    """
    if not data:
        print("No data to save.")
        return ""
    
    slug = laptop_name_to_slug(laptop_name)
    os.makedirs(data_dir, exist_ok=True)
    
    filename_json = os.path.join(data_dir, f"{slug}_search_results.json")
    filename_csv = os.path.join(data_dir, f"{slug}_search_results.csv")
    
    # Save JSON
    try:
        with open(filename_json, "w", encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"Reddit search results saved to: {filename_json}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

    # Save CSV
    try:
        with open(filename_csv, "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Query", "Title", "URL", "Scraped_at"])

            for block in data:
                for post in block["results"]:
                    writer.writerow([
                        block["query"],
                        post["title"],
                        post["url"],
                        block["scraped_at"]
                    ])
        print(f"Reddit search results saved to: {filename_csv}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
    
    return filename_json


def load_existing_search_results(laptop_name: str, data_dir: str = "data") -> list[dict] | None:
    """
    Load existing search results if they exist.
    
    Args:
        laptop_name: Name of the laptop
        data_dir: Directory to look for files
        
    Returns:
        Existing data if found, None otherwise
    """
    filepath = get_search_results_path(laptop_name, data_dir)
    
    if os.path.exists(filepath):
        print(f"Found existing search results: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing data: {e}")
            return None
    return None


def scrape_reddit_queries(laptop_name: str, force_refresh: bool = False, data_dir: str = "data") -> list[dict]:
    """
    Main entry point: Scrape or load Reddit search results for a laptop.
    
    Args:
        laptop_name: Name of the laptop (e.g., "Lenovo Legion Y540")
        force_refresh: If True, scrape even if data exists
        data_dir: Directory for data files
        
    Returns:
        Search results data
    """
    # Check for existing data
    if not force_refresh:
        existing_data = load_existing_search_results(laptop_name, data_dir)
        if existing_data:
            print(f"Using cached search results for '{laptop_name}'")
            return existing_data
    
    # Scrape new data
    print(f"Scraping Reddit search results for '{laptop_name}'...")
    data = scrape_reddit_search(laptop_name)
    
    # Save the data
    save_search_results(data, laptop_name, data_dir)
    
    return data


def extract_unique_urls(search_data: list[dict]) -> list[str]:
    """
    Extract all unique Reddit post URLs from search results.
    
    Args:
        search_data: Search results data from scrape_reddit_queries
        
    Returns:
        List of unique URLs
    """
    seen = set()
    urls = []
    
    for block in search_data:
        for post in block.get("results", []):
            url = post.get("url", "")
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    
    print(f"Extracted {len(urls)} unique URLs from search results")
    return urls


def main() -> None:
    """Test the refactored scraper."""
    # Example: scrape for Lenovo Legion Y540
    laptop_name = "Lenovo Legion Y540"
    
    data = scrape_reddit_queries(laptop_name)
    
    # Show unique URLs
    urls = extract_unique_urls(data)
    print(f"\nUnique URLs found: {len(urls)}")
    for url in urls[:5]:  # Show first 5
        print(f"  - {url}")


if __name__ == "__main__":
    main()
