
import os
import yaml
import pickle
from typing import Dict, Any, Tuple, List
from src.core.config import config

STATE_FILE = "context/simulation_state.yaml"

class Repository:
    def __init__(self):
        self.state_file = STATE_FILE
        # Data paths from config
        self.users_pkl = "data/all_users.pkl"
        self.ads_pkl = "data/all_ads.pkl"

    def load_simulation_state(self) -> Dict[str, Any]:
        defaults = {
            "ads_scheduled_for_day": [],
            "current_simulation_day": 0,
            "active_ads": []  # Ensure this key exists
        }
        
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    loaded = yaml.safe_load(f) or {}
                    # Update defaults with loaded values (preserves defaults for missing keys)
                    defaults.update(loaded)
                    return defaults
            except Exception as e:
                print(f"Error loading state: {e}. Using defaults.")
                return defaults
                
        return defaults

    def save_simulation_state(self, state: Dict[str, Any]):
        with open(self.state_file, "w") as f:
            yaml.safe_dump(state, f)

    def load_data(self) -> Tuple[List, List]:
        """Load legacy pickle data (Users and Ads)."""
        # Note: This returns raw objects. 
        # In a full refactor, we should ensure these objects match src.simulation.models
        try:
            with open(self.users_pkl, "rb") as f:
                users = pickle.load(f)
            with open(self.ads_pkl, "rb") as f:
                ads = pickle.load(f)
        except FileNotFoundError:
            print("Warning: Data pickles not found.")
            users, ads = [], []
        return ads, users

repository = Repository()
