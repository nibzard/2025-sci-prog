
import sys
import os
import duckdb

# Add src to path
sys.path.append(os.getcwd())

# We might need to reload if running in same session, but here we run script fresh.
from src.data.db import db_client

def test_insert():
    print("Testing DB Insert...")
    
    # Check Schema first
    try:
        con = duckdb.connect("data/databases/simulation_db.duckdb")
        print("Schema Info:")
        df = con.execute("PRAGMA table_info('interactions')").df()
        print(df[['cid', 'name', 'type']])
        con.close()
    except Exception as e:
        print(f"Schema Check Failed: {e}")
    
    interaction = {
        "ad_id": "test_ad_01",
        "interaction_rate": 0.5,
        "user_id": "test_user_417",
        "acute_irritation": 10.0,
        "acute_interest": 20.0,
        "acute_arousal": 30.0,
        "bias_irritation": 5.0,
        "bias_trust": 6.0,
        "bias_fatigue": 7.0,
        "acute_irritation_change": 1.0,
        "acute_interest_change": 2.0,
        "acute_arousal_change": 3.0,
        "bias_irritation_change": 0.1,
        "bias_trust_change": 0.2,
        "bias_fatigue_change": 0.3,
        "ignore": False,
        "click": True,
        "like": False,
        "dislike": False,
        "share": 0,
        "reaction_description": "Test reaction",
        "prompt": "Test prompt"
    }
    
    try:
        db_client.save_interaction(interaction)
        print("Insert Successful!")
    except Exception as e:
        print(f"Insert Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_insert()
