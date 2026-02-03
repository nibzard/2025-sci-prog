#reddit_post_scrapper.py

import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup


def scrape_reddit_post(post_url: str) -> dict | None:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }

    print(f"Scraping post: {post_url}")

    try:
        response = requests.get(post_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "No title"

        # Post body
        body_parts = []
        body_container = soup.find(
            "div",
            {"property": "schema:articleBody"} 
        )
        if body_container:
            for paragraph in body_container.find_all("p"):
                text = paragraph.get_text(separator="", strip=True)
                if text:
                    body_parts.append(text)
        
        body_text = " ".join(body_parts)


        # Top-level comments (visible only)
        comments = []
        comment_containers = soup.find_all(
            "div",
            {"slot": "comment"}
        )

        for container in comment_containers:
            for p in container.find_all("p"):
                text = p.get_text(strip=True)
                if text:
                    comments.append(text)
        

        print(f"Scraped {len(body_parts)} body paragraphs and {len(comments)} comments.")
        print(f"url: {post_url}")
        print(f"title: {title}")
        print(f"body: {body_text}")
        print(f"comments: {comments[:10]}...")  # print first 10
        print(f"scraped_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return {
            "url": post_url,
            "title": title,
            "body": body_text,
            "comments": comments[:10],  # limit for sanity
            "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"Error scraping post {post_url}: {e}")
        return None
    

def main() -> None:
    scrape_reddit_post("https://www.reddit.com/r/LenovoLegion/comments/1o7tk8q/my_legion_y540_still_serving_me_after_5_years/")
    

if __name__ == "__main__":
    main()