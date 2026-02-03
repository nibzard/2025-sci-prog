"""
Batch Persona Generation Script

This script harvests and synthesizes personas for all professions in the simulation.
Run this after fixing API keys to generate all 44 personas automatically.

Usage:
    python generate_all_personas.py

Requirements:
    - Valid GOOGLE_API_KEY or OPENAI_API_KEY in .env
    - Playwright installed (already done)
    - data/users_features.csv with all professions
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world.persona_harvester import PersonaHarvester, ManualSearchProvider, GoogleSearchProvider
from world.persona_generator import main as generate_personas

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

def harvest_all_professions(professions):
    """Harvest bios for all professions"""
    print("\n=== PHASE 1: HARVESTING ===\n")
    
    # Setup provider
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if api_key and cse_id:
        provider = GoogleSearchProvider(api_key, cse_id)
        print("Using Google Custom Search API")
    else:
        provider = ManualSearchProvider()
        print("Using Manual Search Provider")
        print("NOTE: You'll need to populate data/persona_raw/manual_discovery.json")
        print("      with URLs for each profession.\n")
    
    harvester = PersonaHarvester(provider)
    os.makedirs("data/persona_raw", exist_ok=True)
    
    output_file = "data/persona_raw/job_best_profile.jsonl"
    
    # Clear existing file
    if os.path.exists(output_file):
        backup = output_file + ".backup"
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(output_file, backup)
        print(f"Backed up existing file to {backup}\n")
    
    successful = []
    failed = []
    
    for i, job in enumerate(professions, 1):
        print(f"[{i}/{len(professions)}] Harvesting: {job}")
        try:
            best = harvester.harvest_for_job(job)
            if best:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(best) + "\n")
                print(f"  ✓ Success: {best['url']} (Score: {best['quality_score']}, Words: {best['word_count']})")
                successful.append(job)
            else:
                print(f"  ✗ Failed: No high-quality profile found")
                failed.append(job)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append(job)
        
        # Polite delay between requests
        if i < len(professions):
            time.sleep(2)
    
    print(f"\n=== HARVESTING COMPLETE ===")
    print(f"Successful: {len(successful)}/{len(professions)}")
    print(f"Failed: {len(failed)}/{len(professions)}")
    
    if failed:
        print(f"\nFailed professions: {', '.join(failed)}")
        print("You may need to manually curate these or adjust search queries.")
    
    return successful, failed

def synthesize_all_personas():
    """Generate 300-500 word narratives for all harvested bios"""
    print("\n=== PHASE 2: SYNTHESIS ===\n")
    
    try:
        generate_personas()
        print("\n=== SYNTHESIS COMPLETE ===")
        return True
    except Exception as e:
        print(f"\n✗ Synthesis failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that GOOGLE_API_KEY or OPENAI_API_KEY is set in .env")
        print("2. Verify the API key is valid and has quota")
        print("3. Check that data/persona_raw/job_best_profile.jsonl exists")
        return False

def verify_personas():
    """Verify all personas were generated correctly"""
    print("\n=== VERIFICATION ===\n")
    
    if not os.path.exists("data/persona_mapping.json"):
        print("✗ persona_mapping.json not found")
        return False
    
    with open("data/persona_mapping.json", 'r') as f:
        mapping = json.load(f)
    
    print(f"Total personas: {len(mapping)}")
    
    issues = []
    for job, filepath in mapping.items():
        if not os.path.exists(filepath):
            issues.append(f"  ✗ {job}: File not found ({filepath})")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            word_count = len(content.split())
            
            if word_count < 300:
                issues.append(f"  ⚠ {job}: Only {word_count} words (expected 300-500)")
            elif word_count > 500:
                issues.append(f"  ⚠ {job}: {word_count} words (expected 300-500)")
            else:
                print(f"  ✓ {job}: {word_count} words")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(issue)
        return False
    
    print("\n✓ All personas verified!")
    return True

def main():
    print("=" * 60)
    print("PERSONA BATCH GENERATION")
    print("=" * 60)
    
    # Get all professions
    professions = get_all_professions()
    if not professions:
        print("✗ No professions found in data/users_features.csv")
        return
    
    print(f"\nFound {len(professions)} unique professions:")
    for prof in professions:
        print(f"  - {prof}")
    
    input("\nPress Enter to start harvesting...")
    
    # Phase 1: Harvest
    successful, failed = harvest_all_professions(professions)
    
    if not successful:
        print("\n✗ No profiles harvested. Cannot proceed to synthesis.")
        return
    
    input("\nPress Enter to start synthesis...")
    
    # Phase 2: Synthesize
    if not synthesize_all_personas():
        print("\n✗ Synthesis failed. Please fix API keys and try again.")
        return
    
    # Phase 3: Verify
    verify_personas()
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review generated personas in data/persona_synthesized/")
    print("2. Run the simulation to test persona integration")
    print("3. Compare agent responses with and without personas")

if __name__ == "__main__":
    main()
