
import sys
import os
import json
import time

# Ensure src in path
sys.path.append(os.getcwd())

from src.simulation.persona import refine_persona_text, clean_persona_text_heuristic

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single(job, path):
    try:
        if not os.path.exists(path):
            return f"Skipping {job} (file not found)"
            
        with open(path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        print(f"Refining {job}...")
        refined = refine_persona_text(raw_text, job)
        
        if refined and len(refined) > 100:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(refined)
            return f"✅ Done {job}"
        else:
            return f"❌ Failed {job} (empty output)"
    except Exception as e:
        return f"❌ Error {job}: {e}"

def main():
    map_file = "data/persona_mapping.json"
    if not os.path.exists(map_file):
        print("Mapping file not found.")
        return

    with open(map_file, 'r') as f:
        mapping = json.load(f)

    print(f"Found {len(mapping)} personas to refine. Starting parallel processing...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single, job, path): job for job, path in mapping.items()}
        
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()
