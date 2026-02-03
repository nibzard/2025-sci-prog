
import sys
import os
import pickle
import json
import random
import duckdb
import pandas as pd
from sklearn.cluster import KMeans

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# HACK: Legacy pickles refer to 'world.definitions'. 
# We moved 'world' to 'archive/legacy_world'.
# We map 'world' to the archived location so pickle can find the classes.
legacy_world_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "archive", "legacy_world"))
if os.path.exists(legacy_world_path):
    # We need to make 'archive.legacy_world' importable as 'world'
    # Simplest way: add archive directory to sys.path and rename legacy_world to world logicially?
    # Or just mock the module if we only needed class structure, but pickle needs exact class.
    
    # Let's add archive folder to path, so 'legacy_world' is importable.
    # But pickle looks for 'world'.
    # So we can create a dummy world module in sys.modules that points to legacy_world.
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("world", os.path.join(legacy_world_path, "__init__.py"))
    if spec:
        module = importlib.util.module_from_spec(spec)
        sys.modules["world"] = module
        spec.loader.exec_module(module)
        
        # Now load definitions
        spec_def = importlib.util.spec_from_file_location("world.definitions", os.path.join(legacy_world_path, "definitions.py"))
        module_def = importlib.util.module_from_spec(spec_def)
        sys.modules["world.definitions"] = module_def
        spec_def.loader.exec_module(module_def)

from src.simulation.models import User as NewUser, Ad as NewAd
from src.data.db import db_client

def load_refined_personas(users):
    """Load refined persona narratives from disk if available."""
    print("🎭 Checking for refined persona narratives...")
    mapping_path = "data/persona_mapping.json"
    
    if not os.path.exists(mapping_path):
        print("⚠️ No persona mapping found. Skipping refinement load.")
        return

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    updated_count = 0
    for user in users:
        # Extract job
        try:
            prof = eval(user.profession) if isinstance(user.profession, str) else user.profession
            job = prof[0] if prof else None
        except: 
            job = None
            
        if job and job in mapping:
            file_path = mapping[job]
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as pf:
                        refined_text = pf.read().strip()
                        if refined_text:
                            user.persona_narrative = refined_text
                            updated_count += 1
                except Exception as e:
                    print(f"⚠️ Error reading persona for {job}: {e}")

    print(f"✅ Loaded refined narratives for {updated_count} users.")

def init_db_groupings(users):
    """Populate users_grouping table in DuckDB using K-Means Clustering."""
    print("🔄 Initializing 'users_grouping' table in database...")
    if not users:
        print("❌ No users to populate in DB.")
        return

    print(f"🧠 Training K-Means model for {len(users)} users...")
    
    # 1. Prepare Data for Clustering
    users_data = []
    for user in users:
        # Safely handle list-as-string fields
        try:
            prof = tuple(sorted(eval(user.profession))) if isinstance(user.profession, str) else tuple(sorted(user.profession))
        except: prof = ("unknown",)
            
        try:
            hoby = tuple(sorted(eval(user.hobby))) if isinstance(user.hobby, str) else tuple(sorted(user.hobby))
        except: hoby = ("unknown",)
            
        users_data.append({
            "user_id": user.user_id,
            "gender": user.gender,
            "age": user.age,
            "profession": prof,
            "hobby": hoby,
            "family": user.family,
            "activity_level": getattr(user, "activity_level", 50),
            "risk_tolerance": getattr(user, "risk_tolerance", 50),
            "social_engagement": getattr(user, "social_engagement", 50)
        })
        
    df = pd.DataFrame(users_data)
    
    # exclude ID from training data
    training_data = df.drop(columns=["user_id"])
    
    # One-Hot Encode categorical features
    df_encoded = pd.get_dummies(training_data, columns=["gender", "profession", "hobby", "family"])
    
    # 2. Train K-Means
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["group"] = kmeans.fit_predict(df_encoded)
    
    # 3. Prepare Insert Data
    # Map group ID to string label (e.g., 'Group 0', 'Group 1') or keep int? 
    # Original random logic used 'A'/'B'. 
    # Targeting logic handles strings. Let's use string 'Group X'.
    
    data = []
    for _, row in df.iterrows():
        group_label = f"Group {row['group']}"
        data.append((str(row['user_id']), group_label, 0)) # user_id, group, day=0

    # 4. Insert into DB
    con = db_client.get_connection()
    try:
        con.execute("DROP TABLE IF EXISTS users_grouping")
        con.execute("""
            CREATE TABLE users_grouping (
                user_id VARCHAR,
                "group" VARCHAR,
                day INTEGER
            )
        """)
        
        con.executemany("INSERT INTO users_grouping VALUES (?, ?, ?)", data)
        count = con.execute("SELECT COUNT(*) FROM users_grouping").fetchone()[0]
        print(f"✅ DB Populated: {count} users assigned to {5} clusters.")
    except Exception as e:
        print(f"❌ DB Error: {e}")
    finally:
        con.close()


def migrate():
    print("Loading legacy data...")
    try:
        with open("data/all_users.pkl", "rb") as f:
            old_users = pickle.load(f)
        with open("data/all_ads.pkl", "rb") as f:
            old_ads = pickle.load(f)
    except FileNotFoundError:
        print("⚠️  No legacy data found in 'data/all_users.pkl'.")
        print("   If this is a new installation, please ensure you have placed your initial data files.")
        print("   Or run the simulation generation scripts first.")
        return

    print(f"Migrating {len(old_users)} users and {len(old_ads)} ads...")

    new_users = []
    for u in old_users:
        # Create NewUser. Copy internal dict.
        # Check if NewUser dataclass has same fields.
        # User is NOT a dataclass in models.py (I defined it as class similar to old one, but cleaner).
        # Actually I defined User as class in models.py, Ad as dataclass.
        
        nu = NewUser(
            user_id=u.user_id,
            gender=u.gender,
            age=u.age,
            profession=u.profession,
            hobby=u.hobby,
            family=u.family
        )
        # Copy other props
        nu.persona_narrative = getattr(u, "persona_narrative", None)
        nu.emotional_state = u.emotional_state
        nu.friend_list = u.friend_list
        nu.activity_level = getattr(u, "activity_level", 50)
        nu.risk_tolerance = getattr(u, "risk_tolerance", 50)
        nu.social_engagement = getattr(u, "social_engagement", 50)
        
        # New model has specific lists for events?
        nu.active_events = [] 
        nu.last_event_days = getattr(u, "last_event_days", {})
        nu.daily_event_summaries = getattr(u, "daily_event_summaries", [])
        
        nu.active_events = [] 
        nu.last_event_days = getattr(u, "last_event_days", {})
        nu.daily_event_summaries = getattr(u, "daily_event_summaries", [])
        
        # Always reload persona to get the refined version
        nu.load_persona()
        
        new_users.append(nu)

    new_ads = []
    for a in old_ads:
        # Ad is a dataclass in models.py
        # Need to match constructor args or fields
        
        # Dataclass fields: 
        # ad_id, group, emotion_label, message_type, visual_style, 
        # num_people, people_present, people_area_ratio, product_present, 
        # product_area_ratio, object_count, object_list, dominant_element, 
        # text_present, text_area_ratio, avg_font_size_proxy, dominant_colors, 
        # brightness_category, saturation_category, hue_category, visual_impact, description
        
        # Old Ad __init__ has same fields + description=None
        
        na = NewAd(
            ad_id=a.ad_id,
            group=a.group,
            emotion_label=a.emotion_label,
            message_type=a.message_type,
            visual_style=a.visual_style,
            num_people=a.num_people,
            people_present=a.people_present,
            people_area_ratio=a.people_area_ratio,
            product_present=a.product_present,
            product_area_ratio=a.product_area_ratio,
            object_count=a.object_count,
            object_list=a.object_list,
            dominant_element=a.dominant_element,
            text_present=a.text_present,
            text_area_ratio=a.text_area_ratio,
            avg_font_size_proxy=a.avg_font_size_proxy,
            dominant_colors=a.dominant_colors,
            brightness_category=a.brightness_category,
            saturation_category=a.saturation_category,
            hue_category=a.hue_category,
            visual_impact=a.visual_impact,
            description=getattr(a, "description", None)
        )
        # Runtime fields
        na.day_of_entry = a.day_of_entry
        na.interaction_rate = a.interaction_rate
        na.is_active = a.is_active
        
        new_ads.append(na)

    # Load refined personas from disk (if available)
    load_refined_personas(new_users)

    print("Saving migrated data...")
    with open("data/all_users.pkl", "wb") as f:
        pickle.dump(new_users, f)
    with open("data/all_ads.pkl", "wb") as f:
        pickle.dump(new_ads, f)
        
    print("Migration complete. Pickles now use src.simulation.models.")
    
    # Initialize DB groupings
    init_db_groupings(new_users)

if __name__ == "__main__":
    migrate()
