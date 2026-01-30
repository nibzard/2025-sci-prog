import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import json
    import random
    import pandas as pd
    import yaml
    import copy
    import numpy as np
    import duckdb
    import pickle
    import requests
    import os
    import sys

    # Ensure local modules can be imported
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    from world.definitions import Ad, User
    from world import engine
    from world.events import EventManager # Added import

    engine.load_env()
    return Ad, User, copy, duckdb, engine, json, mo, np, os, pd, pickle, random, requests, sys, yaml, EventManager


@app.cell
def _(copy, yaml):
    def _read_config():
        with open("context/simulation_config.yaml", "r") as config_file:
            config_str = config_file.read()
        _loaded_config = yaml.safe_load(config_str)
        return copy.deepcopy(_loaded_config)
    config = _read_config()
    return (config,)


@app.cell
def _(mo):
    backend_dropdown = mo.ui.dropdown(
        options=["local", "openai", "google"],
        value="openai",
        label="LLM Backend"
    )
    backend_dropdown
    return (backend_dropdown,)


@app.cell
def _(backend_dropdown, engine):
    fetch_available_models = engine.fetch_available_models
    return (fetch_available_models,)


@app.cell
def _(backend_dropdown, fetch_available_models, mo):
    _options = fetch_available_models(backend_dropdown.value)
    # intelligent default
    _default = _options[0] if _options else None

    # Try to keep previous selection if valid, or pick a smart default based on backend
    if backend_dropdown.value == "openai" and "gpt-4o" in _options:
        _default = "gpt-4o"
    elif backend_dropdown.value == "google" and "gemini-1.5-flash" in _options:
         _default = "gemini-1.5-flash"
    elif backend_dropdown.value == "local" and "llama3" in _options:
        _default = "llama3"

    model_dropdown = mo.ui.dropdown(
        options=_options,
        value=_default,
        label="Model"
    )
    model_dropdown
    return (model_dropdown,)


@app.cell
def _(engine):
    all_ads, all_users = engine.load_data()
    return all_ads, all_users


@app.cell
def _(engine):
    user_groups_df = engine.fetch_groupings()
    return (user_groups_df,)


@app.cell
def _(engine):
    select_agents = engine.select_agents
    return (select_agents,)


@app.cell
def _(engine):
    format_prompt = engine.format_prompt
    return (format_prompt,)


@app.cell
def _(backend_dropdown, engine, model_dropdown):
    def query_llm(prompt, config):
        backend = backend_dropdown.value
        model_name = model_dropdown.value
        return engine.query_llm(prompt, config, backend, model_name)
    return (query_llm,)


@app.cell
def _(engine):
    update_user_emotional_state = engine.update_user_emotional_state
    return (update_user_emotional_state,)


@app.cell
def _(engine):
    save_interaction_to_db = engine.save_interaction_to_db
    return (save_interaction_to_db,)


@app.cell
def _(engine):
    handle_share_action = engine.handle_share_action
    return (handle_share_action,)


@app.cell
def _(engine):
    process_interactions = engine.process_ad
    return (process_interactions,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    run_step = mo.ui.button(label="Run Simulation Step",value=0,  on_click=lambda x: x + 1)
    run_step
    return (run_step,)


@app.cell
def _(
    all_ads,
    all_users,
    backend_dropdown,
    current_day,
    duckdb,
    mo,
    model_dropdown,
    run_step,
    scheduled_ads,
):
    def _create_dashboard():
        # --- Frame 1: Configuration & Control ---
        controls = mo.vstack([
            mo.md("### ⚙️ Simulation Controls"),
            mo.hstack([backend_dropdown, model_dropdown], justify="start"),
            mo.hstack([
                mo.md(f"**Day:** {current_day}"),
                mo.md(f"**Next Ad:** {scheduled_ads[0] if scheduled_ads else 'None'}"),
                run_step
            ], justify="start", align="center")
        ], gap=1)

        # --- Frame 2: Population & Ad States ---
        if all_users:
            avg_interest = sum(u.emotional_state['acute_interest'] for u in all_users) / len(all_users)
            avg_irritation = sum(u.emotional_state['acute_irritation'] for u in all_users) / len(all_users)
            agent_stats = mo.stat(label="Avg Population Interest", value=f"{avg_interest:.1f}")
            irritation_stats = mo.stat(label="Avg Population Irritation", value=f"{avg_irritation:.1f}")
        else:
            agent_stats = mo.md("No agents")
            irritation_stats = mo.md("")

        active_ads_count = len([ad for ad in all_ads if ad.is_active])
        ad_stats = mo.stat(label="Active Ads", value=f"{active_ads_count} / {len(all_ads)}")

        # --- Frame 3: Decision Results ---
        con = None
        try:
            con = duckdb.connect("data/databases/simulation_db.duckdb")
            recent_df = con.execute("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 5").df()
            interactions_view = mo.ui.table(recent_df, label="Latest Decisions") if not recent_df.empty else mo.md("_Waiting for first interaction..._")
        except Exception as e:
            print(f"DEBUG: interactions_view failed: {e}")
            interactions_view = mo.md("_Initializing database..._")
        finally:
            if con:
                con.close()

        # Final Layout Assembly
        layout = mo.vstack([
            mo.md("# � Simulation Management Console"),
            mo.hstack([controls, mo.vstack([agent_stats, irritation_stats, ad_stats])], justify="space-between", align="start"),
            mo.md("---"),
            mo.md("### 🧠 Live Interaction Stream"),
            interactions_view
        ], gap=2)
        return layout

    dashboard = _create_dashboard()
    dashboard
    return


@app.cell
def _(
    all_ads,
    all_users,
    config,
    engine,
    mo,
    process_interactions,
    config,
    engine,
    mo,
    process_interactions,
    run_step,
    EventManager, # Added EventManager
):
    def _perform_step():
        run_step # Trigger on button click
        state = engine.load_simulation_state()
        curr_day = state.get("current_simulation_day", 0)
        sched_ads = state.get("ads_scheduled_for_day", [])
        output = []

        if run_step.value > 0:
            if not sched_ads:
                # Increment day before scheduling
                state["current_simulation_day"] += 1
                curr_day = state["current_simulation_day"]
                
                sched_ads = engine.schedule_for_day(curr_day, all_ads, config)
                state["ads_scheduled_for_day"] = sched_ads
                engine.save_simulation_state(state)
                
                # --- NEW: Daily Event Generation ---
                event_manager = EventManager(config)
                for user in all_users:
                    # Get events for this day
                    # Note: We retrieve them but where do we store them?
                    # The spec says "Inject into LLM prompt". 
                    # The User object now has `active_events` state, but `get_daily_events` returns strings.
                    # We should store the daily text strings on the user temporarily for the day's prompt
                    # OR regenerate them inside process_interactions? 
                    # Better to generate once per day and store on user.
                    
                    user.daily_event_summaries = event_manager.get_daily_events(curr_day, user)
                    
                output.append(mo.md(f"✅ **Schedule Refilled for Day {curr_day}:** {len(sched_ads)} ads added to queue. Daily events updated."))
            else:
                # Process next ad
                ad_id = sched_ads.pop(0)
                target_ad = next((ad for ad in all_ads if ad.ad_id == ad_id), None)

                if target_ad:
                    output.append(mo.md(f"🚀 **Processing Ad {ad_id} (Day {curr_day})**"))
                    results = process_interactions(target_ad, all_users, config, current_day=curr_day)
                    output.append(mo.md(f"Done! {len(results)} agents interacted."))

                    _threshold = config.get("ad_deactivation_threshold", -5)
                    if target_ad.interaction_rate < _threshold:
                        target_ad.is_active = False
                        output.append(mo.md(f"🚫 **Ad {ad_id} deactivated** (Rate: {target_ad.interaction_rate:.2f} < {_threshold})"))
                else:
                    output.append(mo.md(f"❌ Error: Ad {ad_id} not found."))

                # Update state
                state["ads_scheduled_for_day"] = sched_ads
                engine.save_simulation_state(state)

        return curr_day, sched_ads, mo.vstack(output) if output else mo.md("_Click the button to process the next step._")

    current_day, scheduled_ads, step_output = _perform_step()
    step_output
    return all_ads, current_day, scheduled_ads, step_output


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
