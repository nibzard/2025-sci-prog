
import pickle
import os
import sys

# Ensure src in path
sys.path.append(os.getcwd())

def inspect():
    print("--- Inspecting Data ---")
    
    # 1. Check Persona Mapping
    map_file = "data/persona_mapping.json"
    if os.path.exists(map_file):
        print(f"✅ {map_file} exists.")
        # Read content snippet
        with open(map_file, 'r') as f:
            print(f"Content preview: {f.read()[:100]}...")
    else:
        print(f"❌ {map_file} MISSING!")

    # 2. Check User Pickle
    user_file = "data/all_users.pkl"
    if not os.path.exists(user_file):
        print(f"❌ {user_file} MISSING!")
        return

    try:
        with open(user_file, "rb") as f:
            users = pickle.load(f)
        
        print(f"Loaded {len(users)} users.")
        
        # Check first user details
        u = users[0]
        print(f"User 0 ID: {u.user_id}")
        print(f"Profession: {u.profession}")
        print(f"Persona Narrative: {'✅ Present' if u.persona_narrative else '❌ None'} (Len: {len(u.persona_narrative) if u.persona_narrative else 0})")
        print(f"Daily Events: {u.daily_event_summaries}")
        
    except Exception as e:
        print(f"Error reading pickle: {e}")

if __name__ == "__main__":
    inspect()
