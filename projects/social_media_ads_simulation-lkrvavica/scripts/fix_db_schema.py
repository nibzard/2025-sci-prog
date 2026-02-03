
import sys
import os
import duckdb

# Add src to path
sys.path.append(os.getcwd())

from src.data.db import db_client

def fix_schema():
    print("Fixing DB Schema...")
    db_path = "data/databases/simulation_db.duckdb"
    con = duckdb.connect(db_path)
    
    try:
        # 1. Rename old table
        print("Renaming 'interactions' to 'interactions_backup'...")
        con.execute("ALTER TABLE interactions RENAME TO interactions_backup")
        
        # 2. Re-create new table using DBService (which ensures correct schema)
        print("Creating new 'interactions' table...")
        db_client._ensure_schema()
        
        # 3. Migrate data
        print("Migrating data from backup...")
        # We assume columns mostly match, but ad_id needs cast.
        # We need to map columns explicitly to be safe.
        
        # Get columns of backup
        cols_df = con.execute("PRAGMA table_info('interactions_backup')").df()
        bk_cols = set(cols_df['name'].values)
        
        # Build select list
        # Map old ad_id (int) to new ad_id (varchar) -> CAST(ad_id AS VARCHAR)
        select_parts = []
        target_cols = []
        
        # Common columns
        common = [
            "interaction_id", "ad_id", "interaction_rate", "user_id",
            "acute_irritation", "acute_interest", "acute_arousal",
            "bias_irritation", "bias_trust", "bias_fatigue",
            "ignore", "click", "like", "dislike", "share", "reaction_description"
        ]
        
        for c in common:
            if c in bk_cols:
                target_cols.append(c)
                if c == "ad_id":
                    select_parts.append("CAST(ad_id AS VARCHAR)")
                else:
                    select_parts.append(c)
        
        # Construct query
        q = f"""
            INSERT INTO interactions ({", ".join(target_cols)})
            SELECT {", ".join(select_parts)}
            FROM interactions_backup
        """
        print(f"Running migration query...")
        con.execute(q)
        print("Migration successful.")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        # If rename failed (maybe interactions doesn't exist?), try just ensure schema
        if "Table with name interactions does not exist" in str(e):
             print("Table didn't exist, creating new one...")
             db_client._ensure_schema()
    finally:
        con.close()

if __name__ == "__main__":
    fix_schema()
