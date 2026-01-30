import os
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import yaml

class SimulationConfig(BaseSettings):
    # Simulation Parameters
    simulation_days: int = 100
    agents_count: int = 50
    max_ads_shown_per_day: int = 5
    seed: int = 42
    experiment_id: str = "default_exp"
    
    # Paths from original config (can be overriden)
    ad_data_path: str = "data/ads_features.csv"
    user_data_path: str = "data/users_features.csv"
    
    # LLM Settings
    llm_backend: str = "openai"
    openai_model: str = "gpt-4o"
    google_model: str = "gemini-1.5-flash"
    local_model: str = "llama3"
    local_llm_url: str = "http://localhost:11434"
    
    # API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None
    
    model_config = SettingsConfigDict(
        # Look for .env in project root relative to this file
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @classmethod
    def load(cls, yaml_path: str = "context/simulation_config.yaml") -> "SimulationConfig":
        """Load config from YAML first, then override with Env vars."""
        # 1. Load YAML defaults if available
        yaml_config = {}
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}

        # 2. Instantiate (Pydantic will merge Env vars over these values)
        return cls(**yaml_config)

# Singleton global instance
config = SimulationConfig.load()
