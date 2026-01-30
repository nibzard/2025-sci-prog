import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from abc import ABC, abstractmethod

class SearchProvider(ABC):
    @abstractmethod
    def search(self, query, target_count):
        pass

class GoogleSearchProvider(SearchProvider):
    def __init__(self, api_key, cse_id):
        self.api_key = api_key
        self.cse_id = cse_id
        self.url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query, target_count):
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(target_count, 10)
        }
        try:
            response = requests.get(self.url, params=params)
            if response.status_code == 200:
                data = response.json()
                return [{"url": item["link"], "rank": i+1} for i, item in enumerate(data.get("items", []))]
            else:
                print(f"    Google API Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    Search error: {e}")
        return []

class ManualSearchProvider(SearchProvider):
    """Fallback provider that reads from a local JSON file of pre-discovered URLs."""
    def __init__(self, discovery_file="data/persona_raw/manual_discovery.json"):
        self.discovery_file = discovery_file
        self.data = {}
        if os.path.exists(discovery_file):
            with open(discovery_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

    def search(self, query, target_count):
        # Match the job from the query (e.g., job in query)
        for job, urls in self.data.items():
            if f'"{job}"' in query or job in query:
                return [{"url": url, "rank": i+1} for i, url in enumerate(urls[:target_count])]
        return []

class SerpAPISearchProvider(SearchProvider):
    """Uses SerpAPI for Google searches (free tier: 100 searches/month)"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://serpapi.com/search"
    
    def search(self, query, target_count):
        params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query,
            "num": min(target_count, 10)
        }
        try:
            response = requests.get(self.url, params=params)
            if response.status_code == 200:
                data = response.json()
                results = []
                for i, item in enumerate(data.get("organic_results", [])):
                    results.append({
                        "url": item.get("link"),
                        "rank": i + 1
                    })
                return results[:target_count]
            else:
                print(f"    SerpAPI Error: {response.status_code}")
                return []
        except Exception as e:
            print(f"    SerpAPI search error: {e}")
            return []

class PersonaHarvester:
    def __init__(self, search_provider, run_id="upgrade_08"):
        self.run_id = run_id
        self.search_provider = search_provider
        self.blacklist = ["about", "privacy", "terms", "help", "support", "blog", "press", "contact", "login", "signup", "pricing", "search"]
        
    def canonicalize_url(self, url):
        url = url.split('?')[0].split('#')[0]
        url = url.rstrip('/')
        if url.startswith('http://'):
            url = 'https://' + url[7:]
        return url.lower()

    def is_valid_profile_url(self, url):
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if 'about.me' not in parsed.netloc.lower():
                return False
            segments = parsed.path.strip("/").split("/")
            if len(segments) != 1 or segments[0] in self.blacklist:
                return False
            return True
        except:
            return False

    def scrape_profile(self, url):
        """Scrapes bio from about.me profile using Playwright for JS rendering."""
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=15000)
                
                # Wait for content to load
                try:
                    page.wait_for_selector('body', timeout=5000)
                except:
                    pass
                
                # Extract bio from common about.me selectors
                bio = None
                
                # Try multiple selectors
                selectors = [
                    '.bio-content',
                    '.description',
                    '[data-testid="bio"]',
                    '.about-text',
                    'p.bio',
                    'div.bio'
                ]
                
                for selector in selectors:
                    try:
                        element = page.locator(selector).first
                        if element.count() > 0:
                            bio = element.inner_text()
                            if bio and len(bio.strip()) > 20:
                                break
                    except:
                        continue
                
                # Fallback: get all paragraph text
                if not bio or len(bio.strip()) < 20:
                    try:
                        paragraphs = page.locator('p').all()
                        texts = [p.inner_text() for p in paragraphs if len(p.inner_text().strip()) > 20]
                        bio = ' '.join(texts[:5])  # First 5 substantial paragraphs
                    except:
                        pass
                
                # Last resort: visible body text
                if not bio or len(bio.strip()) < 20:
                    try:
                        bio = page.locator('body').inner_text()[:2000]
                    except:
                        bio = None
                
                browser.close()
                
                if not bio or len(bio.strip()) < 10:
                    print(f"    No substantial bio found for {url}")
                    return None
                
                return self.sanitize_bio(bio)
                
        except Exception as e:
            print(f"    Error scraping {url} with Playwright: {e}")
            # Fallback to requests-based scraping
            return self._scrape_with_requests(url)
    
    def _scrape_with_requests(self, url):
        """Fallback scraping method using requests."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: 
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            og_desc = soup.find('meta', attrs={"property": "og:description"})
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            
            if og_desc and og_desc.get('content'):
                return self.sanitize_bio(og_desc['content'])
            elif meta_desc and meta_desc.get('content'):
                return self.sanitize_bio(meta_desc['content'])
            
            return None
        except:
            return None

    def sanitize_bio(self, bio):
        bio = re.sub(r'\S+@\S+', '[EMAIL]', bio)
        bio = re.sub(r'\+?\d[\d -]{7,}\d', '[PHONE]', bio)
        bio = re.sub(r'@\w+', '[HANDLE]', bio)
        return ' '.join(bio.split())

    def calculate_score(self, bio):
        """Simplified scoring: prioritize length, minimal penalties"""
        if not bio: return -10, 0
        words = bio.split()
        word_count = len(words)
        score = word_count  # Base score is just word count
        
        # Spam penalties only
        if len(re.findall(r'http[s]?://\S+', bio)) > 5: score -= 50
        if bio.count('#') > 15: score -= 30
        
        return score, word_count

    def harvest_for_job(self, job):
        print(f"Harvesting candidates for: {job}")
        
        # Simplified query - just find the job on about.me
        queries = [
            f'site:about.me "{job}"'
        ]
        
        candidates = []
        seen_urls = set()
        
        for query in queries:
            if len(candidates) >= 5: break
            results = self.search_provider.search(query, 10)
            for res in results:
                url = self.canonicalize_url(res['url'])
                if self.is_valid_profile_url(url) and url not in seen_urls:
                    candidates.append({
                        "run_id": self.run_id,
                        "job_primary": job,
                        "url": url,
                        "rank": res.get("rank", 0),
                        "retrieved_at": datetime.now().isoformat()
                    })
                    seen_urls.add(url)
        
        best_candidate = None
        max_score = -100
        
        for cand in candidates:
            print(f"  Evaluating: {cand['url']}")
            bio = self.scrape_profile(cand['url'])
            if bio:
                score, wc = self.calculate_score(bio)
                cand.update({"selected_bio_text": bio, "word_count": wc, "quality_score": score})
                print(f"    Words: {wc}, Score: {score}")
                if score > max_score:
                    max_score = score
                    best_candidate = cand
            time.sleep(1)
            
        if best_candidate and best_candidate['word_count'] >= 20:  # Minimum 20 words
            best_candidate['low_quality'] = False
            return best_candidate
        
        print(f"  [FAILURE] No profile with >20 words found for {job}")
        return None

    def cleanup_agents(self, failed_professions, csv_path="data/users_features.csv"):
        """Removes agents with failed professions from the CSV."""
        if not failed_professions:
            return
        
        print(f"\n--- Cleaning up agents for {len(failed_professions)} failed professions ---")
        import pandas as pd
        
        try:
            df = pd.read_csv(csv_path)
            initial_count = len(df)
            
            # Profession is stored as string representation of list: ['writer']
            def should_keep(prof_str):
                try:
                    profs = eval(prof_str)
                    return not any(p in failed_professions for p in profs)
                except:
                    return prof_str not in failed_professions
            
            df = df[df['profession'].apply(should_keep)]
            
            removed = initial_count - len(df)
            df.to_csv(csv_path, index=False)
            print(f"  Removed {removed} agents. New total: {len(df)}")
            
        except Exception as e:
            print(f"  Error cleaning up CSV: {e}")

def main():
    # Setup
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if api_key and cse_id:
        provider = GoogleSearchProvider(api_key, cse_id)
        print("Using Google Custom Search API")
    else:
        provider = ManualSearchProvider()
        print("Using Manual Search Provider (data/persona_raw/manual_discovery.json)")

    harvester = PersonaHarvester(provider)
    os.makedirs("data/persona_raw", exist_ok=True)
    
    # Identify unique professions from current users
    import pandas as pd
    try:
        df = pd.read_csv("data/users_features.csv")
        professions = set()
        for _, row in df.iterrows():
            try:
                profs = eval(row['profession'])
                professions.update(profs)
            except:
                professions.add(row['profession'])
        professions = sorted(list(professions))
    except:
        professions = []
    
    output_file = "data/persona_raw/job_best_profile.jsonl"
    successful_profs = set()
    failed_profs = []
    
    for job in professions:
        best = harvester.harvest_for_job(job)
        if best:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(best) + "\n")
            print(f"  Success: {job} -> {best['url']} (Score: {best['quality_score']})")
            successful_profs.add(job)
        else:
            failed_profs.append(job)
            
    # Remove agents without personas
    # harvester.cleanup_agents(failed_profs)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
