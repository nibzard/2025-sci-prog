import requests
import json
import csv
import time
from bs4 import BeautifulSoup

#fuction - > main scraper to get content from reddit
def scrape_reddit() -> list[dict]:
    subreddit = {
        "https://www.reddit.com/r/LenovoLegion",
        "https://www.reddit.com/r/GamingLaptops",
        "https://www.reddit.com/r/Lenovo"
    }

    all_data = []
    for url in subreddit:
        headers = {
            'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        subreddit_name = url.split('/')[-1]
        print(f'Scraping subreddit: {subreddit_name}')

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            subreddit_data = {
                'subreddit': subreddit_name,
                "url": url,
                "title": soup.title.string if soup.title else 'No title',
                "scraped_at" : time.strftime("%Y-%m-%d %H:%M:%S")
            }
            topics = []
            for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
                text = heading.get_text(strip=True)

                if text and len(text) > 3:
                    if any(keyword in text.lower() for keyword in ["ideapad", "setup", "laptop", "thinkpad", "legion", "yoga"]):
                        topics.append({
                            "title": text,
                            "type": "lenovo_topic"
                        })
            discussions = []
            seen_urls = set()

            for link in soup.find_all("a", href=True):
                text = link.get_text(strip=True)
                href = link["href"]

                if (text and len(text) > 1 and "/comments/" in href and href not in seen_urls):
                    seen_urls.add(href)
                    discussions.append({
                        "title": text[:100] + "..." if len(text) > 100 else text,
                        "url": href,
                        "type": "discussion"
                    })
            subreddit_data["lenovo_topics"] = topics
            subreddit_data["discussions"] = discussions

            all_data.append(subreddit_data)
            time.sleep(2)

        except Exception as e:
            print(f"Error fetching {url}: {e}")

    return all_data


def save_scraped_data(data, filename_json="lenovo_topics.json", filename_csv="lenovo_topics.csv"):
    if not data:
        print("No data to save.")
        return
    #json
    try:
        with open(filename_json, "w", encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=True, indent=2)
        print(f"Lenovo data saved to: {filename_json}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

    #csv
    try:
        with open(filename_csv, "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Subreddit", "Type", "Title", "URL", "Scraped_at"])

            for subreddit_data in data:
                subreddit = subreddit_data["subreddit"]
                scraped_at = subreddit_data["scraped_at"]
                for topic in subreddit_data.get("lenovo_topics", []):
                    writer.writerow([subreddit, topic['type'], "", scraped_at])
                
                for discussion in subreddit_data.get("discussions", []):
                    writer.writerow([subreddit, discussion['type'], discussion['title'], discussion['url'], scraped_at])
                
            print("All topics saved to files!")
    except Exception as e:
        print(f"Error saving CSV file: {e}")



def main() -> None:
    data = scrape_reddit()

    if data:
        print(f'Processing data...')
        total_topics = 0
        total_discussions = 0

        for subreddit_data in data:
            topics_count = len(subreddit_data.get("lenovo_topics", []))
            discussions_count = len(subreddit_data.get("discussions", []))
            
            total_topics += topics_count
            total_discussions += discussions_count

        print(f"Total topics: {total_topics}")
        print(f"Total discussions: {total_discussions}")

        save_scraped_data(data)
    else:
        print("No data scraped.")

if __name__ == "__main__":
    main()