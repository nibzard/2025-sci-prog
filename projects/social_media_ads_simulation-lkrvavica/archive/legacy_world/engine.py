import json
import random
import os
import requests
import yaml
import duckdb
import pandas as pd
import numpy as np
import pickle
from world.definitions import Ad, User
from dotenv import load_dotenv, find_dotenv

# Shared State and Config
STATE_FILE = "context/simulation_state.yaml"
CONFIG_FILE = "context/simulation_config.yaml"

DEFAULT_PROMPT_TEMPLATE = """
    You are a {age} year old {profession}.
    {persona_narrative}

    It is Day {current_day} of the simulation.
    Here is what is happening in your life today:
    {daily_events}

    You are currently on your daily phone break, casually scrolling through your feed. You encounter an ad.
    
    [Ad Context]
    {ad_description}

    [Instructions]
    Your task is to react to this ad based on your specific persona, your current mood, and the events happening in your life today.
    
    1. **Internal Monologue:** First, provide a rich, first-person thought process (40-80 words). How does this ad land with you *right now*? Does it clash with your bad mood? Is it a welcome distraction? Connect the ad's visuals/message to your specific background and today's events.
    2. **Action:** Decide how you interact with it (ignore, click, like, dislike, share).
    3. **Emotional Shift:** Quantify how this momentary experience shifts your internal state.

    Return your response in this JSON format:
    {
        "reaction_description": "string",  # Your internal monologue
        "ignore": boolean,
        "click": boolean,
        "like": boolean,
        "dislike": boolean,
        "share": 0 or 1,
        "acute_irritation_change": number, # -50 to 50
        "acute_interest_change": number,   # -50 to 50
        "acute_arousal_change": number,    # -50 to 50
        "bias_irritation_change": number,  # -50 to 50
        "bias_trust_change": number,       # -50 to 50
        "bias_fatigue_change": number      # -50 to 50
    }
    
    Note on Actions:
    - You MUST choose at least one action (including 'ignore').
    - 'ignore' is incompatible with other actions.
    - 'click', 'like', 'dislike', 'share' can be combined logicially.
    """

def clean_json_response(response_str):
    """Clean up markdown blocks and whitespace from LLM response."""
    if not response_str:
        return "{}"
    
    cleaned = response_str.strip()
    
    # Remove markdown code blocks
    if cleaned.startswith("```"):
        # Find the first newline to skip '```json' or just '```'
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline:].strip()
        
        # Remove trailing '```'
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
            
    # If it's still empty or doesn't look like JSON, return empty object
    if not cleaned or not (cleaned.startswith("{") and cleaned.endswith("}")):
        return "{}"
        
    return cleaned

def load_env():
    possible_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.getcwd()), ".env"),
        "../.env",
        ".env"
    ]
    loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            load_dotenv(path)
            loaded = True
            break
    if not loaded:
        fd_path = find_dotenv()
        if fd_path:
            load_dotenv(fd_path)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return {}

def load_simulation_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return yaml.safe_load(f) or {"ads_scheduled_for_day": [], "current_simulation_day": 0}
    return {"ads_scheduled_for_day": [], "current_simulation_day": 0}

def save_simulation_state(state):
    with open(STATE_FILE, "w") as f:
        yaml.safe_dump(state, f)

def handle_share_action(agent, ad, share_val):
    """
    Handle the side effects of a share action.
    When a user shares an ad, it gets added to the global schedule queue
    so other users (simulating friends/network) see it.
    """
    if not share_val:
        return None
        
    state = load_simulation_state()
    # Add ad back to queue if it's not already there too many times
    # In a real system, we'd target the user's friends specifically, 
    # but for this simulation we'll just reinject it into the global day queue.
    state["ads_scheduled_for_day"].append(ad.ad_id)
    save_simulation_state(state)
    
    # Pick a random friend name for the UI update if available
    friend_id = random.choice(agent.friend_list) if agent.friend_list else "a friend"
    return friend_id

def load_data():
    try:
        with open("data/all_users.pkl", "rb") as f:
            users = pickle.load(f)
        with open("data/all_ads.pkl", "rb") as f:
            ads = pickle.load(f)
    except FileNotFoundError:
        users, ads = [], []
    return ads, users

def fetch_groupings():
    con = None
    try:
        con = duckdb.connect("data/databases/simulation_db.duckdb")
        df = con.execute("SELECT user_id, \"group\" FROM users_grouping WHERE day = 0").df()
    except Exception:
        df = pd.DataFrame(columns=["user_id", "group"])
    finally:
        if con: con.close()
    return df

# Simulation Logic

def select_agents(ad, users, config, user_groups_df, return_stats=False):
    def tppm_mocker(ad, users):
        return {user.user_id: random.random() for user in users}
    
    predictions = tppm_mocker(ad, users)
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['user_id', 'prediction'])
    merged_df = pd.merge(predictions_df, user_groups_df, on='user_id')
    group_avg = merged_df.groupby('group')['prediction'].mean().reset_index()
    total_avg = group_avg['prediction'].sum()
    agents_exposed_to_ad = config.get('agents_exposed_to_ad', 10)

    if total_avg > 0:
        scaling_factor = agents_exposed_to_ad / total_avg
        group_avg['scaled_prediction'] = group_avg['prediction'] * scaling_factor
    else:
        num_groups = len(group_avg)
        group_avg['scaled_prediction'] = agents_exposed_to_ad / num_groups if num_groups > 0 else 0

    selected_agents = []
    selection_details = []
    
    for _, row in group_avg.iterrows():
        group_id = row['group']
        target_count = int(round(row['scaled_prediction']))
        group_user_ids = user_groups_df[user_groups_df['group'] == group_id]['user_id'].tolist()
        group_users = [u for u in users if u.user_id in group_user_ids]
        
        count = min(len(group_users), target_count)
        sampled = random.sample(group_users, count) if group_users else []
        selected_agents.extend(sampled)
        
        selection_details.append({
            "group": group_id,
            "available": len(group_users),
            "target": target_count,
            "selected_count": count,
            "selected_ids": [u.user_id for u in sampled]
        })
    
    if return_stats:
        return selected_agents, selection_details
    return selected_agents

def format_prompt(agent, ad, custom_template=None, current_day=0):
    template = custom_template if custom_template else DEFAULT_PROMPT_TEMPLATE
    
    # 1. Persona Narrative
    if getattr(agent, "persona_narrative", None):
        persona_text = agent.persona_narrative
    else:
        # Fallback to feature list if no narrative exists
        persona_text = f"Features: {agent.to_message_format()}"

    # 2. Daily Events
    daily_events = getattr(agent, "daily_event_summaries", [])
    if daily_events:
        events_str = "\n".join([f"- {ev}" for ev in daily_events])
    else:
        events_str = "Nothing significant is happening today."

    # 3. Ad Description
    if getattr(ad, "description", None):
        ad_text = ad.description
    else:
        ad_text = (
            f"Visual Style: {ad.visual_style}\n"
            f"Dominant Element: {ad.dominant_element}\n"
            f"Emotion: {ad.emotion_label}\n"
            f"Message: {ad.message_type}"
        )
    # Append the raw JSON for completeness if needed, or keep it clean. 
    # The new prompt design focuses on narrative, but having raw data is useful for the LLM.
    # We will append it to the ad_text to ensure the LLM definitely knows what it is reacting to.
    ad_text += f"\n\n Technical Details: {ad.to_message_format()}"

    # 4. Replacement
    prompt = template.replace("{age}", str(agent.age))
    prompt = prompt.replace("{profession}", str(agent.profession))
    prompt = prompt.replace("{persona_narrative}", persona_text)
    prompt = prompt.replace("{current_day}", str(current_day))
    prompt = prompt.replace("{daily_events}", events_str)
    prompt = prompt.replace("{ad_description}", ad_text)
    
    return prompt
    
    prompt = prompt.replace("{ad_json}", ad_json)
    prompt = prompt.replace("{ad_description}", ad.description or "No description available.")
    return prompt

def query_llm(prompt, config, backend=None, model_name=None, json_mode=True):
    backend = backend or config.get("llm_backend", "openai")
    
    # Use backend-specific defaults from config if model_name is not provided
    if not model_name:
        if backend == "openai":
            model_name = config.get("openai_model", "gpt-4o")
        elif backend == "google":
            model_name = config.get("google_model", "gemini-1.5-flash")
        elif backend == "local":
            model_name = config.get("local_model", "llama3")
        else:
            model_name = "gpt-4o"

    load_env()

    if backend == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key: return "{}" if json_mode else ""
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return clean_json_response(content) if json_mode else content.strip()
        except Exception as e:
            print(f"DEBUG: OpenAI Query Failed: {e}")
            return "{}" if json_mode else ""
        
    elif backend == "google":
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY")
        if not key: return "{}" if json_mode else ""
        genai.configure(api_key=key)
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return clean_json_response(response.text) if json_mode else response.text.strip()
        except Exception as e:
            print(f"DEBUG: Google Query Failed: {e}")
            return "{}" if json_mode else ""

    elif backend == "local":
        url = config.get("local_llm_url", "http://localhost:11434/api/generate")
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
            
        try:
            r = requests.post(url, json=payload)
            r.raise_for_status()
            content = r.json().get("response", "{}")
            return clean_json_response(content) if json_mode else content.strip()
        except Exception as e:
            print(f"DEBUG: Local Query Failed: {e}")
            return "{}" if json_mode else ""
    return "{}" if json_mode else ""

def update_user_emotional_state(user, response_json):
    for key in ["acute_irritation", "acute_interest", "acute_arousal", "bias_irritation", "bias_trust", "bias_fatigue"]:
        change = response_json.get(f"{key}_change", 0)
        user.emotional_state[key] = max(0, min(100, user.emotional_state[key] + change))

def save_interaction_to_db(interaction):
    con = None
    try:
        con = duckdb.connect("data/databases/simulation_db.duckdb")
        con.execute("CREATE SEQUENCE IF NOT EXISTS interaction_id_seq")
        # Migration logic (same as in simulator.py)
        cols = con.execute("PRAGMA table_info('interactions')").df()
        if 'prompt' not in cols['name'].values:
            con.execute("ALTER TABLE interactions ADD COLUMN prompt TEXT")
        if 'timestamp' not in cols['name'].values:
            con.execute("ALTER TABLE interactions ADD COLUMN timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        
        con.execute("""
            INSERT INTO interactions (
                interaction_id, ad_id, interaction_rate, user_id, 
                acute_irritation, acute_interest, acute_arousal, 
                bias_irritation, bias_trust, bias_fatigue,
                acute_irritation_change, acute_interest_change, acute_arousal_change, 
                bias_irritation_change, bias_trust_change, bias_fatigue_change,
                "ignore", click, "like", dislike, share, reaction_description, prompt
            ) VALUES (nextval('interaction_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            interaction['ad_id'], interaction.get('interaction_rate', 0.0), interaction['user_id'],
            interaction['acute_irritation'], interaction['acute_interest'], interaction['acute_arousal'],
            interaction['bias_irritation'], interaction['bias_trust'], interaction['bias_fatigue'],
            interaction['acute_irritation_change'], interaction['acute_interest_change'], interaction['acute_arousal_change'],
            interaction['bias_irritation_change'], interaction['bias_trust_change'], interaction['bias_fatigue_change'],
            interaction['ignore'], interaction['click'], interaction['like'], interaction['dislike'], interaction['share'],
            interaction['reaction_description'], interaction.get('prompt', '')
        ])
    finally:
        if con: con.close()

def schedule_for_day(current_day, all_ads, config):
    active_ads = [ad for ad in all_ads if ad.is_active or ad.day_of_entry == current_day]
    for ad in active_ads: ad.is_active = True
    max_ads = config.get("max_ads_shown_per_day", 5)
    if len(active_ads) > max_ads:
        selected = random.sample(active_ads, max_ads)
        return [ad.ad_id for ad in selected]
    return [ad.ad_id for ad in active_ads]

def fetch_available_models(backend):
    models = []
    load_env()
    if backend == "google":
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY")
        if key:
            try:
                genai.configure(api_key=key)
                all_models = genai.list_models()
                models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
                models = [m.replace("models/", "") for m in models]
            except Exception: models = ["gemini-1.5-flash", "gemini-pro"]
    elif backend == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if key:
            try:
                headers = {"Authorization": f"Bearer {key}"}
                r = requests.get("https://api.openai.com/v1/models", headers=headers)
                if r.status_code == 200:
                    models = [m["id"] for m in r.json().get("data", []) if "gpt" in m["id"]]
                    models.sort()
            except Exception: models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    elif backend == "local":
        config = load_config()
        url = config.get("local_llm_url", "http://localhost:11434").replace("/api/generate", "") + "/api/tags"
        try:
            r = requests.get(url)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
        except Exception: models = ["llama3", "mistral"]
    return models if models else ["default"]

def validate_and_sanitize_interaction(data):
    """
    Validates and sanitizes the LLM response data.
    Enforces mutual exclusivity of 'ignore' and other actions.
    Ensures at least one action is taken.
    """
    # 1. Enforce specific types & Defaults
    defaults = {
        'ignore': False, 'click': False, 'like': False, 'dislike': False, 'share': 0,
        'reaction_description': "",
        'acute_irritation_change': 0, 'acute_interest_change': 0, 'acute_arousal_change': 0,
        'bias_irritation_change': 0, 'bias_trust_change': 0, 'bias_fatigue_change': 0
    }
    for k, v in defaults.items():
        if k not in data:
            data[k] = v

    # Ensure bools
    for key in ['ignore', 'click', 'like', 'dislike']:
        data[key] = bool(data.get(key, False))
    
    # Ensure share is int 0 or 1
    data['share'] = 1 if data.get('share') else 0

    # 2. Logic Check: Ignore vs Others
    # If ignore is True, force others to False/0
    if data['ignore']:
        if data['click'] or data['like'] or data['dislike'] or data['share']:
            # Conflict detected: strict enforcement -> Ignore wins if explicitly set
            data['click'] = False
            data['like'] = False
            data['dislike'] = False
            data['share'] = 0
    
    # 3. Logic Check: At least one action
    has_action = data['ignore'] or data['click'] or data['like'] or data['dislike'] or data['share']
    if not has_action:
        # Fallback: If nothing selected, default to ignore
        data['ignore'] = True

    return data

def process_ad(target_ad, all_users, config, backend=None, model_name=None, custom_template=None, on_agent_start=None, on_agent_finish=None, current_day=0):
    """Processes an ad across multiple agents."""
    user_groups_df = fetch_groupings()
    selected_agents = select_agents(target_ad, all_users, config, user_groups_df)
    
    results = []
    for agent in selected_agents:
        if on_agent_start: on_agent_start(agent)
        
        prompt = format_prompt(agent, target_ad, custom_template, current_day=current_day)
        response_str = query_llm(prompt, config, backend, model_name)
        
        try:
            if not response_str or response_str == "{}":
                raise ValueError("Empty or fallback response received")
                
            response_json = json.loads(response_str)
            
            # --- VALIDATION & SANITIZATION ---
            response_json = validate_and_sanitize_interaction(response_json)
            
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
            
            save_interaction_to_db(interaction)
            results.append(interaction)
            
            # Update Ad Rate
            target_ad.update_interaction_rate(
                click=1 if interaction['click'] else 0,
                share=1 if interaction['share'] > 0 else 0,
                like=1 if interaction['like'] else 0,
                dislike=1 if interaction['dislike'] else 0,
                ignore=1 if interaction['ignore'] else 0
            )
            
            if on_agent_finish: on_agent_finish(agent, interaction)
        except Exception as e:
            print(f"Error processing agent {agent.user_id}: {e}")
            
    return results
