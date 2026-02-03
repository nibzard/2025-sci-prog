
import duckdb
import pandas as pd
import os
from typing import Dict, Any, Optional

class DBService:
    def __init__(self, db_path: str = "data/databases/simulation_db.duckdb"):
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_schema()

    def get_connection(self):
        return duckdb.connect(self.db_path)

    def _ensure_schema(self):
        """Idempotent schema migration."""
        con = self.get_connection()
        try:
            con.execute("CREATE SEQUENCE IF NOT EXISTS interaction_id_seq")
            con.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id INTEGER DEFAULT nextval('interaction_id_seq'),
                    ad_id VARCHAR,
                    interaction_rate DOUBLE,
                    user_id VARCHAR,
                    acute_irritation DOUBLE,
                    acute_interest DOUBLE,
                    acute_arousal DOUBLE,
                    bias_irritation DOUBLE,
                    bias_trust DOUBLE,
                    bias_fatigue DOUBLE,
                    acute_irritation_change DOUBLE,
                    acute_interest_change DOUBLE,
                    acute_arousal_change DOUBLE,
                    bias_irritation_change DOUBLE,
                    bias_trust_change DOUBLE,
                    bias_fatigue_change DOUBLE,
                    ignore BOOLEAN,
                    click BOOLEAN,
                    "like" BOOLEAN,
                    dislike BOOLEAN,
                    share INTEGER,
                    reaction_description TEXT,
                    prompt TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Migration check for existing columns if needed (legacy support)
            cols = con.execute("PRAGMA table_info('interactions')").df()
            existing_cols = set(cols['name'].values)
            
            if 'prompt' not in existing_cols:
                con.execute("ALTER TABLE interactions ADD COLUMN prompt TEXT")
            if 'timestamp' not in existing_cols:
                con.execute("ALTER TABLE interactions ADD COLUMN timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                
        finally:
            con.close()

    def fetch_groupings(self) -> pd.DataFrame:
        con = self.get_connection()
        try:
            # Check if table exists first
            tables = con.execute("SHOW TABLES").df()
            if 'users_grouping' not in tables['name'].values:
                return pd.DataFrame(columns=["user_id", "group"])
                
            df = con.execute("SELECT user_id, \"group\" FROM users_grouping WHERE day = 0").df()
            return df
        except Exception as e:
            print(f"DB Error fetching groupings: {e}")
            return pd.DataFrame(columns=["user_id", "group"])
        finally:
            con.close()

    def save_interaction(self, interaction: Dict[str, Any]):
        con = self.get_connection()
        try:
            # 1. Generate ID explicitly in Python to ensure correct parameter binding
            # This avoids issues where DuckDB might misinterpret mixed literals/placeholders
            seq_id = con.execute("SELECT nextval('interaction_id_seq')").fetchone()[0]
            
            # 2. Insert with pure parameters
            con.execute("""
                INSERT INTO interactions (
                    interaction_id, ad_id, interaction_rate, user_id, 
                    acute_irritation, acute_interest, acute_arousal, 
                    bias_irritation, bias_trust, bias_fatigue,
                    acute_irritation_change, acute_interest_change, acute_arousal_change, 
                    bias_irritation_change, bias_trust_change, bias_fatigue_change,
                    "ignore", click, "like", dislike, share, reaction_description, prompt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                seq_id,
                interaction['ad_id'], interaction.get('interaction_rate', 0.0), interaction['user_id'],
                interaction['acute_irritation'], interaction['acute_interest'], interaction['acute_arousal'],
                interaction['bias_irritation'], interaction['bias_trust'], interaction['bias_fatigue'],
                interaction['acute_irritation_change'], interaction['acute_interest_change'], interaction['acute_arousal_change'],
                interaction['bias_irritation_change'], interaction['bias_trust_change'], interaction['bias_fatigue_change'],
                interaction['ignore'], interaction['click'], interaction['like'], interaction['dislike'], interaction['share'],
                interaction['reaction_description'], interaction.get('prompt', '')
            ])
        finally:
            con.close()

# Singleton
db_client = DBService()
