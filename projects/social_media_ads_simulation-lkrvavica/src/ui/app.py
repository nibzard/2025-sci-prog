
import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import json
    import os
    import sys
    import pandas as pd
    
    # Ensure root path
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
        
    from src.core.config import config
    from src.data.repository import repository
    from src.services.llm import llm_client
    from src.data.db import db_client
    from src.simulation.events import EventManager
    from src.simulation.engine import schedule_for_day, process_ad
    
    event_manager = EventManager()
    
    return (
        EventManager,
        config,
        db_client,
        event_manager,
        json,
        llm_client,
        mo,
        os,
        pd,
        process_ad,
        repository,
        schedule_for_day,
        sys,
    )


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
def _(backend_dropdown, llm_client, mo):
    _options = llm_client.fetch_available_models(backend_dropdown.value)
    _default = _options[0] if _options else "default"
    
    # Smart default
    if backend_dropdown.value == "openai" and "gpt-4o" in _options:
        _default = "gpt-4o"
    elif backend_dropdown.value == "google" and "gemini-1.5-flash" in _options:
         _default = "gemini-1.5-flash"

    model_dropdown = mo.ui.dropdown(
        options=_options,
        value=_default,
        label="Model"
    )
    model_dropdown
    return (model_dropdown,)


@app.cell
def _(repository):
    all_ads, all_users = repository.load_data()
    return all_ads, all_users


@app.cell
def _(mo):
    run_step = mo.ui.button(label="Run Simulation Step", value=0, on_click=lambda x: x + 1)
    run_step
    return (run_step,)


@app.cell
def _(
    all_ads,
    all_users,
    backend_dropdown,
    db_client,
    mo,
    model_dropdown,
    repository,
    run_step,
):
    # State Loading (Reactive to run_step)
    # We load state every time this cell runs (which happens when run_step changes)
    # However, to display current state BEFORE run, we need to load it.
    # marimo cells are reactive.
    
    _state = repository.load_simulation_state()
    _curr_day = _state.get("current_simulation_day", 0)
    _sched = _state.get("ads_scheduled_for_day", [])
    
    def _create_dashboard():
        # Frame 1
        controls = mo.vstack([
            mo.md("### ⚙️ Simulation Controls (Refactored)"),
            mo.hstack([backend_dropdown, model_dropdown], justify="start"),
            mo.hstack([
                mo.md(f"**Day:** {_curr_day}"),
                mo.md(f"**Next Ad:** {_sched[0] if _sched else 'None'}"),
                run_step
            ], justify="start", align="center")
        ], gap=1)

        # Frame 2
        active_ads_count = len([ad for ad in all_ads if ad.is_active])
        ad_stats = mo.stat(label="Active Ads", value=f"{active_ads_count} / {len(all_ads)}")
        
        # Frame 3 (DB)
        try:
            con = db_client.get_connection()
            recent_df = con.execute("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 5").df()
            con.close()
            interactions_view = mo.ui.table(recent_df, label="Latest Decisions") if not recent_df.empty else mo.md("_Waiting for intx..._")
        except Exception as e:
            interactions_view = mo.md(f"_DB Error: {e}_")

        layout = mo.vstack([
            mo.md("#  Simulation Management Console"),
            mo.hstack([controls, ad_stats], justify="space-between", align="start"),
            mo.md("---"),
            interactions_view
        ], gap=2)
        return layout

    dashboard = _create_dashboard()
    dashboard
    return dashboard,


@app.cell
def _(
    all_ads,
    all_users,
    backend_dropdown,
    event_manager,
    mo,
    model_dropdown,
    process_ad,
    repository,
    run_step,
    schedule_for_day,
):
    def _perform_step():
        # Dependency on run_step to trigger
        run_step
        
        # Load state freshly
        state = repository.load_simulation_state()
        curr_day = state.get("current_simulation_day", 0)
        sched_ads = state.get("ads_scheduled_for_day", [])
        output = []

        if run_step.value > 0:
            if not sched_ads:
                # NEW DAY
                state["current_simulation_day"] += 1
                curr_day = state["current_simulation_day"]
                
                sched_ads = schedule_for_day(curr_day, all_ads)
                state["ads_scheduled_for_day"] = sched_ads
                repository.save_simulation_state(state)
                
                # Events
                for user in all_users:
                    user.daily_event_summaries = event_manager.get_daily_events(curr_day, user)
                    
                output.append(mo.md(f"✅ **Day {curr_day} Started:** {len(sched_ads)} ads scheduled."))
            else:
                # PROCESS AD
                ad_id = sched_ads.pop(0)
                target_ad = next((ad for ad in all_ads if ad.ad_id == ad_id), None)

                if target_ad:
                    output.append(mo.md(f"🚀 **Processing Ad {ad_id}**"))
                    results = process_ad(
                        target_ad, 
                        all_users, 
                        backend=backend_dropdown.value, 
                        model_name=model_dropdown.value,
                        current_day=curr_day
                    )
                    output.append(mo.md(f"Done! {len(results)} interactions."))
                else:
                    output.append(mo.md(f"❌ Error: Ad {ad_id} not found."))

                state["ads_scheduled_for_day"] = sched_ads
                repository.save_simulation_state(state)
                
        return mo.vstack(output) if output else mo.md("")

    step_output = _perform_step()
    step_output
    return (step_output,)


@app.cell
def _():
    return
