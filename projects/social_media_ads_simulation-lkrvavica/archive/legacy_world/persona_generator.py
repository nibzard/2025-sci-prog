import os
import json
import time
from world import engine

def load_raw_profiles(file_path="data/persona_raw/job_best_profile.jsonl"):
    profiles = []
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            profiles.append(json.loads(line))
    return profiles

def synthesize_persona(profile, config):
    job = profile.get("job_primary", "Unknown")
    bio = profile.get("selected_bio_text", "")
    
    prompt = f"""
    You are a professional persona designer for a high-fidelity social simulation.
    I will provide you with a raw draft/bio of a real person who works as a {job}.
    
    RAW BIO:
    {bio}
    
    YOUR TASK:
    Rewrite this into a rich, evocative, and deeply human first-person narrative persona. 
    The final description must be between 300 and 500 words.
    
    GUIDELINES:
    1. VOICE: Capture a distinct "voice". If they are a {job}, how does that specific career affect their language, their level of stress, their attention to detail, and their visual priorities?
    2. DEPTH: Go beyond the resume. Describe their daily routine, their small motivations, their secret frustrations, and what they care about when they are "off the clock".
    3. ATMOSPHERE: Use sensory details. Instead of saying "I like art," describe the specific smell of oil paints or the way morning light hits a canvas.
    4. NO BULLETS: Write this as a continuous, flowing internal monologue or self-reflection.
    5. NON-TECHNICAL: Avoid corporate jargon. Make them feel like a real person someone would meet in a coffee shop.
    6. CONTEXT: Briefly mention how their background might make them perceive advertisements—do they look at the design, the message, the price, or do they mostly try to ignore the noise?
    
    FORMAT:
    Return only the narrative text. Do not include any headers or meta-commentary.
    """
    
    # Use Google API since OpenAI key appears expired
    response = engine.query_llm(prompt, config, backend="google", json_mode=False)
    
    return response

def main():
    config = engine.load_config()
    raw_profiles = load_raw_profiles()
    
    if not raw_profiles:
        print("No raw profiles found to synthesize.")
        return
        
    output_dir = "data/persona_synthesized"
    os.makedirs(output_dir, exist_ok=True)
    
    mapping_file = "data/persona_mapping.json"
    persona_mapping = {}
    if os.path.exists(mapping_file):
        with open(mapping_file, "r", encoding="utf-8") as f:
            persona_mapping = json.load(f)

    for profile in raw_profiles:
        job = profile["job_primary"]
        if job in persona_mapping:
            print(f"Skipping {job}, already synthesized.")
            continue
            
        print(f"Synthesizing persona for: {job}...")
        try:
            narrative = synthesize_persona(profile, config)
            
            # Save the individual narrative to a file
            filename = f"{job.replace(' ', '_')}.txt"
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                f.write(narrative)
            
            persona_mapping[job] = os.path.join(output_dir, filename)
            
            # Save mapping incrementally
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(persona_mapping, f, indent=4)
                
            print(f"  Done. Word count: {len(narrative.split())}")
            time.sleep(1) # Avoid rate limits
        except Exception as e:
            print(f"  Error synthesizing {job}: {e}")

if __name__ == "__main__":
    main()
