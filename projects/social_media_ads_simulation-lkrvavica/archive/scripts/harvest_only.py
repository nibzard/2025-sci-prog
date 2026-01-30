"""
Quick Persona Harvesting (No Prompts)
Just harvests all bios and shows stats
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world.persona_harvester import PersonaHarvester, ManualSearchProvider, GoogleSearchProvider

def get_all_professions():
    """Extract unique professions from users_features.csv"""
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
        return sorted(list(professions))
    except Exception as e:
        print(f"Error reading professions: {e}")
        return []

def main():
    print("=" * 60)
    print("HARVESTING ALL PERSONAS")
    print("=" * 60)
    
    # Setup provider
    load_dotenv()
    
    # Try SerpAPI first (easiest), then Google CSE, then manual
    serpapi_key = os.getenv("SERPAPI_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if serpapi_key:
        from world.persona_harvester import SerpAPISearchProvider
        provider = SerpAPISearchProvider(serpapi_key)
        print("Using SerpAPI\n")
    elif google_key and cse_id:
        provider = GoogleSearchProvider(google_key, cse_id)
        print("Using Google Custom Search\n")
    else:
        provider = ManualSearchProvider()
        print("Using Manual Search (limited)\n")
    
    harvester = PersonaHarvester(provider)
    os.makedirs("data/persona_raw", exist_ok=True)
    
    professions = get_all_professions()
    print(f"Found {len(professions)} professions\n")
    
    output_file = "data/persona_raw/job_best_profile.jsonl"
    
    # Backup existing file
    if os.path.exists(output_file):
        backup = output_file + ".backup"
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(output_file, backup)
        print(f"Backed up to {backup}\n")
    
    successful = []
    failed = []
    
    for i, job in enumerate(professions, 1):
        print(f"[{i}/{len(professions)}] {job}")
        try:
            best = harvester.harvest_for_job(job)
            if best:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(best) + "\n")
                print(f"  ✓ {best['word_count']} words from {best['url']}")
                successful.append(job)
            else:
                print(f"  ✗ Failed")
                failed.append(job)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append(job)
        
        if i < len(professions):
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {len(successful)}/{len(professions)} successful")
    print("=" * 60)
    
    if failed:
        print(f"\nFailed ({len(failed)}): {', '.join(failed)}")

if __name__ == "__main__":
    main()
