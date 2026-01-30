"""
Convert harvested bios to persona format
Creates individual .txt files and mapping.json
"""

import json
import os

# Read harvested data
with open("data/persona_raw/job_best_profile.jsonl", 'r', encoding='utf-8') as f:
    profiles = [json.loads(line) for line in f]

print(f"Converting {len(profiles)} personas...")

# Create output directory
os.makedirs("data/persona_synthesized", exist_ok=True)

# Create mapping
mapping = {}

for profile in profiles:
    job = profile['job_primary']
    bio = profile['selected_bio_text']
    
    # Save to individual file
    filename = f"data/persona_synthesized/{job}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(bio)
    
    # Add to mapping
    mapping[job] = filename
    
    print(f"  ✓ {job}: {len(bio.split())} words")

# Save mapping
with open("data/persona_mapping.json", 'w', encoding='utf-8') as f:
    json.dump(mapping, f, indent=2)

print(f"\n✅ Created {len(mapping)} persona files")
print(f"✅ Created persona_mapping.json")
print("\nPersonas are ready to use!")
