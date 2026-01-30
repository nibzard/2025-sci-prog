from world.persona_harvester import PersonaHarvester, GoogleSearchProvider
import os
from dotenv import load_dotenv

load_dotenv()

provider = GoogleSearchProvider(os.getenv('GOOGLE_API_KEY'), os.getenv('GOOGLE_CSE_ID'))
harvester = PersonaHarvester(provider)

print("Testing search for 'writer'...")
results = provider.search('site:about.me "writer"', 5)
print(f"Found {len(results)} search results")
for r in results[:3]:
    print(f"  - {r}")

if results:
    print("\nTesting scraping first URL...")
    bio = harvester.scrape_profile(results[0]['url'])
    if bio:
        print(f"SUCCESS! Scraped {len(bio.split())} words")
        print(f"First 100 chars: {bio[:100]}")
    else:
        print("FAILED to scrape")
