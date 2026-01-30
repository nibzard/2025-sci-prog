# Interaction History Management System

- Creating a system to store interaction history and determine optimal agent data compression.

## Objectives and research question

Design and implement a robust system for storing all agent-ad interactions in DuckDB, and develop multiple schemes for compressing agent history data to optimize LLM context usage. Research question: Which compression method best balances information retention with context efficiency?

## Detailed specs

### Functional requirements
- Store all interactions (user_id, ad_id, timestamp, actions, metadata)
- Explore and evalueate possibilities for compression scheme
- Create the compression scheme

### Technical requirements
- DuckDB database (`data/databases/simulation_db.duckdb`)
- Python database interface

### Subtasks
1. **Create an interaction history management system that stores each interaction** - Design database table and storage functions
- Create a database table that will store all interactions
- features that will be stored:
- tbd: user feature types are not defined yet
2. **Select the best compression system** - Evaluate possibilities
- vector emotional model with acute and bias emotional effects is estimated to be the best fit, with open posibility for changing the model if major flaws in this model appears during implementation or simulation
- the system will describe agent's shift in emotion, and it will describe acute emotional effects which will be short therm reaction to ad and will decay fast and completely reset at the end of the day and bias which will show longer-therm consenquences of the interaction and will decay slower over time
3. **Create the compression system** - Define where and how will this system be implemented (not to be implemented yet)
- the system will be implemented in User class
- each user will be given the emotional state features: acute_irritation, acute_interest, acute_arousal, bias_irritation, bias_trust, bias_fatigue
- agent will respond with emotionional state shifts for each feature
5. **Persist Agent and Ad Objects** - Implement pickle storage
- Save `all_ads` and `all_users` lists as pickle files in `initialization.py`
- Load these lists in `simulator.py` to replace CSV/DataFrame usage
4. **Create an user grups storage system** - Crate a table in database that will store the info which user is in which 
- Create a database table that will store all user grouping iteration
- the table will store:
    - day (intager, non-nullable)
    - user_id (intager, non-nullable)
    - group (intager, non-nullable)
    - model (text, nullable)
    - confidence (float, nullable)

### Dependencies
- None (foundational infrastructure)

### Input data

## Output
- `data/databases/simulation_db.duckdb` - DuckDB database

## Workflow, algorithms and procedures

## Issues and challenges

## Results and conclusions

## Notes
