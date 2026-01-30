# Ad Simulation Engine

## 🚀 Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Setup**:
    - Copy `.env.example` to `.env` (if provided) or create one with your API keys:
      ```
      OPENAI_API_KEY=sk-...
      # or
      GOOGLE_API_KEY=...
      ```

3.  **Initialize Database**:
    - The database is excluded from git to prevent conflicts. 
    - Initialize it and load sample data by running:
      ```bash
      python scripts/migrate_pickles.py
      ```
      *(This script migrates/seeds the `data/databases/simulation_db.duckdb` file from raw data sources).*

4.  **Run the Visualizer**:
    ```bash
    streamlit run src/ui/visualizer.py
    ```

## 📂 Project Structure
- `src/`: Core application logic (Modularized).
- `data/`: Data storage (JSON configuration, Pickles).
- `scripts/`: Utilities for scraping and data maintenance.
- `docs/`: Detailed documentation.

## Legacy Code
Old implementation files are archived in `archive/legacy_world/` for reference.
ulation
