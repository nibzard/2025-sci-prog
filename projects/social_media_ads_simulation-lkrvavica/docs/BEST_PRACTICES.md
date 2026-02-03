# Mixed-Intelligence Engineering: Best Practices & Guidelines

## Overview
This document serves as the "Constitution" for any codebase co-developed by **Humans and AI Agents**. It outlines universal architectural standards, coding principles, and collaboration protocols to prevent the common pitfalls of hybrid development (e.g., "God Objects", hardcoding, and architectural drift).

---

## 1. Architectural Integrity

### 🛑 Anti-Pattern: The Root Dump
*   **Do not** dump source files into the root directory.
*   **Do not** mix scripts, application logic, and configuration in one bucket.
*   **Do not** create "God Objects" (single files/classes that handle DB, Logic, UI, and API).

### ✅ Best Practice: Standard Layout
Adopt a modular structure immediately, even for prototypes.

```
project_root/
├── src/                # Application Source Code (The "Product")
│   ├── core/           # Config, Logging, Shared Utilities
│   ├── domain/         # Business Logic & Models (Pure Python/Code)
│   ├── infrastructure/ # Database, External APIs, File I/O
│   └── interface/      # UI, Web API routes, CLI handlers
├── tests/              # Structured Test Suite (Mirroring src/)
├── scripts/            # Ops, Migration, & Maintenance Scripts (Not app logic)
├── docs/               # Architecture, Plans, Context
└── config/             # Configuration files (if not using pure env)
```

**Rule:** `src/` is for code that *is* the application. `scripts/` is for code that *acts on* the application.

---

## 2. Configuration & State Management

### 🛑 Anti-Pattern: Hardcoded & Hidden
*   **Do not** hardcode file paths (e.g., `open("C:/Users/...")`).
*   **Do not** hardcode magic numbers or API keys.
*   **Do not** scatter `os.environ` or `.env` loading calls throughout the codebase.

### ✅ Best Practice: Centralized Typesafe Config
*   **Single Source of Truth:** Create a centralized `config.py` (using `pydantic-settings` or similar) that loads all environment variables once.
*   **Relative Paths:** Always derive paths relative to the **Project Root**, unrelated to where the script is run from.
*   **Secrets:** Never commit secrets. Use `.env.example` templates.

---

## 3. Data Integrity & Type Safety

### 🛑 Anti-Pattern: Primitive Obsession & Dictionaries
*   **Do not** pass untyped dictionaries/maps as core data structures.
*   **Do not** rely on stringly-typed data (e.g., passing `"true"` instead of `True`).

### ✅ Best Practice: Models & Contracts
*   **Formal Schemas:** Use **Dataclasses**, **Pydantic Models**, or **Structs** for all domain entities.
*   **Explicit Contracts:** Function signatures must declare types. Agents rely on these signatures to understand how to use your code without hallucinating.
*   **Validation:** Validate inputs at the system boundary (API/CLI), then trust the data internally.

---

## 4. Human-Agent Protocols 🤝

### For Agents 🤖
1.  **Read Before Write:** Before creating a new file, **search** the codebase. Does a utility already exist? Does a folder already exist for this domain?
2.  **Respect Scope:** If assigned a task in the UI layer, do not refactor the Database layer unless explicitly authorized.
3.  **Update "Memory":** When finding or fixing a bug, update the documentation/tasks so future agents (and humans) know the current state.
4.  **No Silent Failures:** Do not catch generic `Exception` without logging or re-raising. Silence causes debugging nightmares.

### For Humans 🧑‍💻
1.  **Architecture as Context:** When prompting an agent, briefly point them to the relevant modules (e.g., *"Add this feature to `src/domain`, utilizing the config in `src/core`"*).
2.  **Code Review:** Be the gatekeeper. Reject "lazy" Agent code that adds files to the root or duplicates logic. Force them to refactor into the established structure.
3.  **Living Docs:** Maintain a `tasks.md` or `plan.md`. Agents work best when they can "see" the implementation plan.

---

## 5. Testing & Reproducibility

*   **Tests are Mandatory:** Agents are prone to regression (breaking old things while fixing new ones). High test coverage is the only safety net.
*   **Mock Externalities:** Unit tests must not hit live APIs or production databases.
*   **Deterministic Logic:** Seed random number generators in reproducible workflows (simulations, training).

---

## Summary
**Consistency is Speed.**
In a Mixed-Intelligence project, clear boundaries and strict typing allow Humans and Agents to trust each other's code. If the architecture is ambiguous, the Agent will guess (and often guess wrong).
