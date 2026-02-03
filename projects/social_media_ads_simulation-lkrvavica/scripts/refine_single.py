
import sys
import os
sys.path.append(os.getcwd())

from src.simulation.persona import refine_persona_text

def main():
    path = "data/persona_synthesized/writer.txt"
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    
    print("Refining writer...")
    try:
        refined = refine_persona_text(raw, "writer")
        print(f"Result (len={len(refined)}):")
        print(refined[:100] + "...")
        
        if len(refined) > 100:
             with open(path, 'w', encoding='utf-8') as f:
                    f.write(refined)
             print("Saved.")
        else:
            print("Output too short, not saving.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Ensure env is loaded
    main()
