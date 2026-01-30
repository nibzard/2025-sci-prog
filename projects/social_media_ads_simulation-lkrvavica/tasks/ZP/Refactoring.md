# Refactoring Analysis and Plan

## 1. Current State Assessment

The current codebase has grown organically and contains several architectural anti-patterns that hinder scalability and maintainability.

### Bad Practices Identified
-   **God Objects:** `world/engine.py` is overloaded. It handles configuration, database connections, LLM API calls, prompt engineering, and simulation logic.
-   **Mixed Responsibilities:** `world/simulator.py` mixes UI logic (Marimo), event loop management, and business logic. `world/visualizer.py` duplicates initialization logic found in `engine.py`.
-   **Hardcoded Paths:** Numerous references to `data/...` and `context/...` are scattered throughout the code, making it brittle to directory changes.
-   **Global State/Side Effects:** `load_env()` and `load_config()` are invoked in global scopes or repeatedly in functions, leading to unpredictable state initialization.
-   **Incomplete Dependencies:** `requirements.txt` is missing key libraries used in recent features: `playwright`, `pytest` (implied for testing), and potentially `serpapi` dependencies if they aren't raw requests.
-   **Scattered Tests:** Tests are in `tests/` but cover disjointed features (`events`, `response_validation`) without a cohesive suite or runner configuration.
-   **Naming Inconsistencies:** `initialization.py` vs `initialisation.py` (both exist in `world/`).
-   **Shadowing:** Variables like `config` are often shadowed in function arguments, leading to potential confusion.

### Good Practices to Preserve
-   **Data Classes:** The usage of `Ad` and `User` classes in `definitions.py` provides some structure.
-   **Deterministic Seeding:** The Event system uses SHA256 seeding for reproducibility.
-   **DuckDB Integration:** Efficient local OLAP storage for interactions.
-   **Data-Driven Design:** Loading personas and events from external JSON/JSONL files allows for content updates without code changes.

## 2. Refactoring Plan

### Phase 1: Project Restructuring & Dependencies
**Goal:** Establish a standard Python project layout and ensure reproducible environments.

1.  **Directory Layout:**
    ```
    ads_sim/
    ├── src/
    │   ├── core/           # Config, logging, env, paths
    │   ├── data/           # Data access objects (DAO), DB connections
    │   ├── services/       # External services (LLM, SerpAPI)
    │   ├── simulation/     # Core entities (User, Ad, Event) and Logic
    │   └── ui/             # Visualization and Notebooks
    ├── tests/              # Structured test suite
    ├── scripts/            # Setup and utility scripts
    ├── data/               # (Symlinked or configured path for runtime data)
    └── config/             # (Symlinked or configured path for runtime config)
    ```
2.  **Update `requirements.txt`:** Add `playwright`, `pytest`, `pytest-cov`, and freeze versions.
3.  **Config Module:** Create `src/core/config.py` using `pydantic-settings` or a robust singleton for typesafe configuration access.

### Phase 2: Core Module Extraction
**Goal:** Break `engine.py` into focused modules.

1.  **`src/simulation/models.py`:** Move `User`, `Ad`, `Event` definitions here. Use `pydantic` for better validation if possible.
2.  **`src/services/llm.py`:** Extract `query_llm` and backend management into a dedicated `LLMService` class.
3.  **`src/data/db.py`:** dedicated class for DuckDB connection management and queries (DAO pattern). Remove direct SQL strings from business logic.
4.  **`src/simulation/prompting.py`:** Extract `DEFAULT_PROMPT_TEMPLATE` and `format_prompt` logic.

### Phase 3: Logic Decoupling
**Goal:** Separate Simulation Loop from UI.

1.  **`src/simulation/engine.py`:** Create a `SimulationEngine` class that manages the day loop, event triggering, and ad selection *purely*. It should return state objects, not Marimo UI elements.
2.  **`src/ui/marimo_app.py`:**  Refactor `world/simulator.py` to be a thin View layer that consumes `SimulationEngine`.
3.  **`src/ui/streamlit_app.py`:** Refactor `world/visualizer.py` to use `SimulationEngine` instead of duplicating logic.

### Phase 4: Testing & Quality
**Goal:** 100% Core Coverage.

1.  **Test Suite:** detailed `tests/unit` and `tests/integration` folders.
2.  **Fixtures:** centralized `conftest.py` for mocking LLMs and DBs.
3.  **Coverage:** Ensure all logic branches (actions, events, validations) are covered.

## 3. Immediate Next Steps (Task List)

- [ ] **Infrastructure Setup**: Create folders, move files, update `requirements.txt`.
- [ ] **Module Extraction**: Split `engine.py` into `src/core`, `src/services`, `src/simulation`.
- [ ] **Dependency Injection**: Update consumers to import from new locations.
- [ ] **Test Migration**: Move and update existing tests.
- [ ] **Verification**: Run full simulation to ensure no regression.

## Implementation Flow

### Step 1: Project Structuring (Completed)
- Created directory tree: `src/{core,data,services,simulation,ui}`, `tests/{unit,integration}`, `config/`, `scripts/`.
- Updated `requirements.txt` with `pydantic`, `pytest`, `playwright`.
- Implemented `src/core/config.py` using Pydantic `BaseSettings` for type-safe configuration.

### Step 2: Core Module Extraction (Completed)
- **Models**: Extracted `User`, `Ad`, `Event` to `src/simulation/models.py`. Converted `Ad` to `@dataclass`.
- **Services**:
  - `src/services/llm.py`: Unified `LLMService` handling OpenAI/Google/Local.
  - `src/data/db.py`: Encapsulated DuckDB logic in `DBService`.
  - `src/data/repository.py`: Created `Repository` for file-based persistence (pickles/yaml).
- **Simulation Logic**:
  - `src/simulation/events.py`: `EventManager` for efficient event pooling.
  - `src/simulation/targeting.py`: `select_agents` logic.
  - `src/simulation/prompting.py`: `format_prompt` and templates.
  - `src/simulation/engine.py`: Orchestrator functions (`schedule_for_day`, `process_ad`).

### Step 3: Data Migration (Completed)
- **Problem**: Existing pickle files contained `world.definitions.User` instances, incompatible with `src.simulation.models.User`.
- **Solution**: Created `scripts/migrate_pickles.py` to map legacy objects to new classes and save them back to `data/`.

### Step 4: UI Refactoring (Completed)
- **Streamlit**: Rebuilt as `src/ui/visualizer.py`, importing from `src` modules.
- **Marimo**: Rebuilt as `src/ui/app.py`, removing inline class definitions.

### Step 5: Testing (Completed)
- Created `tests/conftest.py` for path setup.
- Migrated tests to `tests/unit/`.
- verified logic with `run_tests.py`.

### Phase 6: Cleanup & Reorganization (Completed)
- **Objective**: Remove legacy code and clean up the root directory.
- **Actions**:
  - **Archived**: Moved `world/` (legacy implementation) to `archive/legacy_world/`.
  - **Scripts**: Moved root-level scripts (`harvest_only.py`, `generate_all.py` etc.) to `archive/scripts/`.
  - **Docs**: Moved documentation files (`.md`) to `docs/`.
  - **Harvester**: Migrated `persona_harvester.py` from `world/` to `scripts/harvest_personas.py` (standalone utility).
  - **Git**: Created `.gitignore` to exclude secrets (`.env`), database files, and cache directories.

## Structure Critique: Old vs New

### 🔴 Old Structure (`world/` + flat root)
- **Monolithic**: `world/` contained everything: engine, events, viz, scraping. Hard to navigate.
- **Mixed Concerns**: `engine.py` handled logic, state, AND some database-like operations.
- **Root Clutter**: 15+ python scripts in the root directory made it unclear what the entry point was.
- **Hardcoded Paths**: Logic often relied on relative paths that broke if valid from different locations.
- **State Coupling**: State was passed around in ad-hoc dictionaries or global variables.

### 🟢 New Structure (`src/`)
- **Modular**: `src/` is split into logical domains:
  - `core/`: Config & type definitions.
  - `data/`: Database & persistence (Repositories).
  - `simulation/`: Pure business logic (Engine, Models, Events).
  - `services/`: External integrations (LLM).
  - `ui/`: Presentation layer (Streamlit).
- **Separation of Concerns**: UI logic (`ui/`) is completely decoupled from Simulation logic (`simulation/`).
- **Type Safety**: Heavy use of `Pydantic` and `dataclasses` ensures data integrity between modules.
- **Testability**: Logic is now importable and testable without side-effects (e.g., `tests/unit/`).
- **Clean Root**: Root only contains configuration (`.env`, `requirements.txt`) and high-level documentation. Entries are clear (`streamlit run src/ui/visualizer.py`).
