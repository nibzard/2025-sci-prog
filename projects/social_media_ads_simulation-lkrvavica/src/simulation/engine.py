
import json
import random
from typing import List, Optional, Callable, Dict, Any

from src.core.config import config
from src.simulation.models import Ad, User
from src.services.llm import llm_client
from src.data.db import db_client
from src.simulation.targeting import select_agents
from src.simulation.prompting import format_prompt
from src.simulation.core import validate_and_sanitize_interaction, update_user_emotional_state

def schedule_for_day(current_day: int, all_ads: List[Ad], active_ads_list: List[str] = None) -> tuple[List[str], List[str]]:
    """
    Select ads for the day's schedule.
    - Activates ads entering on current day (adds to active_ads_list)
    - Schedules all currently active ads
    - Randomly samples if there are too many active ads
    
    Returns:
        (scheduled_ad_ids, updated_active_ads_list)
    """
    if active_ads_list is None:
        active_ads_list = []
    
    # Activate ads entering today
    # Activate ads entering today
    for ad in all_ads:
        # Robust comparison: handle potential None or string/int mismatches
        try:
            entry_day = int(ad.day_of_entry) if ad.day_of_entry is not None else -1
            curr_day_int = int(current_day)
            
            # Catch-up logic: Activate if intended entry day is passed or today
            if entry_day <= curr_day_int and ad.ad_id not in active_ads_list:
                active_ads_list.append(ad.ad_id)
                ad.is_active = True
        except (ValueError, TypeError):
            continue
    
    # Get all currently active ads
    active_ads = [ad for ad in all_ads if ad.ad_id in active_ads_list]
    
    # Sample if needed
    max_ads = config.max_ads_shown_per_day
    if len(active_ads) > max_ads:
        selected = random.sample(active_ads, max_ads)
        return [ad.ad_id for ad in selected], active_ads_list
    
    return [ad.ad_id for ad in active_ads], active_ads_list

def process_ad(
    target_ad: Ad, 
    all_users: List[User], 
    backend: str = None, 
    model_name: str = None, 
    custom_template: str = None, 
    on_agent_start: Callable = None, 
    on_agent_finish: Callable = None, 
    current_day: int = 0
) -> List[Dict[str, Any]]:
    """
    Processes an ad across multiple agents.
    """
    user_groups_df = db_client.fetch_groupings()
    
    # We pass user_groups_df to targeting
    selected_agents = select_agents(target_ad, all_users, user_groups_df)
    
    results = []
    for agent in selected_agents:
        if on_agent_start: 
            on_agent_start(agent)
        
        prompt = format_prompt(agent, target_ad, custom_template, current_day=current_day)
        
        # Query LLM
        response_str = llm_client.query(prompt, backend=backend, model_name=model_name)
        
        try:
            if not response_str or response_str == "{}":
                raise ValueError("Empty or fallback response received")
                
            response_json = json.loads(response_str)
            
            # Validation & Sanitization
            response_json = validate_and_sanitize_interaction(response_json)
            
            # State Update
            update_user_emotional_state(agent, response_json)
            
            interaction = {
                "user_id": agent.user_id,
                "ad_id": target_ad.ad_id,
                "interaction_rate": target_ad.interaction_rate,
                "acute_irritation": agent.emotional_state["acute_irritation"],
                "acute_interest": agent.emotional_state["acute_interest"],
                "acute_arousal": agent.emotional_state["acute_arousal"],
                "bias_irritation": agent.emotional_state["bias_irritation"],
                "bias_trust": agent.emotional_state["bias_trust"],
                "bias_fatigue": agent.emotional_state["bias_fatigue"],
                "acute_irritation_change": response_json.get("acute_irritation_change", 0),
                "acute_interest_change": response_json.get("acute_interest_change", 0),
                "acute_arousal_change": response_json.get("acute_arousal_change", 0),
                "bias_irritation_change": response_json.get("bias_irritation_change", 0),
                "bias_trust_change": response_json.get("bias_trust_change", 0),
                "bias_fatigue_change": response_json.get("bias_fatigue_change", 0),
                "ignore": response_json.get("ignore", False),
                "click": response_json.get("click", False),
                "like": response_json.get("like", False),
                "dislike": response_json.get("dislike", False),
                "share": response_json.get("share", 0),
                "reaction_description": response_json.get("reaction_description", ""),
                "prompt": prompt
            }
            
            db_client.save_interaction(interaction)
            results.append(interaction)
            
            # Update Ad Rate
            target_ad.update_interaction_rate(
                click=1 if interaction['click'] else 0,
                share=1 if interaction['share'] > 0 else 0,
                like=1 if interaction['like'] else 0,
                dislike=1 if interaction['dislike'] else 0,
                ignore=1 if interaction['ignore'] else 0
            )
            
            if on_agent_finish: 
                on_agent_finish(agent, interaction)
                
        except Exception as e:
            print(f"Error processing agent {agent.user_id}: {e}")
            # Ensure on_agent_finish is called even on error if needed?
            # Original code didn't.
            
    return results

def handle_share_action(agent: User, ad: Ad, share_val: int) -> Optional[str]:
    """
    Handle the side effects of a share action.
    This logic touches simulation state (schedule queue), requiring a store access.
    
    Refactoring Note: This function accesses 'context/simulation_state.yaml'.
    Logic should ideally be injected or handled by a State Manager.
    For now, we will duplicate the YAML loading logic or assume a StateService exists?
    
    We haven't created StateService yet. Let's create `src/data/repository.py` for this.
    For now, I'll inline the YAML load using `src.data.repository` IF it existed.
    I will handle this in `src/simulation/engine.py` using helper for now, 
    but marked for future refactor.
    """
    if not share_val:
        return None
    
    # We need to import load/save state. Ideally from a repository.
    # I will add a local helper or import from a new module `src.data.repository`.
    from src.data.repository import repository
    
    state = repository.load_simulation_state()
    state["ads_scheduled_for_day"].append(ad.ad_id)
    repository.save_simulation_state(state)
    
    friend_id = random.choice(agent.friend_list) if agent.friend_list else "a friend"
    return friend_id
