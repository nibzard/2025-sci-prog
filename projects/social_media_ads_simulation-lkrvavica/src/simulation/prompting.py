
from src.simulation.models import User, Ad

DEFAULT_PROMPT_TEMPLATE = """
IMPORTANT: Your main goal is reacting to ad, you do so by adopting the described personality and visualizing ad's description, and react to given ad accordingly.
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
IMPORTANT: Your main goal is reacting to ad
"""

def format_prompt(agent: User, ad: Ad, custom_template: str = None, current_day: int = 0) -> str:
    template = custom_template if custom_template else DEFAULT_PROMPT_TEMPLATE
    
    # 1. Persona Narrative
    persona_text = agent.persona_narrative if agent.persona_narrative else f"Features: {agent.to_message_format()}"

    # 2. Daily Events
    daily_events = getattr(agent, "daily_event_summaries", [])
    if daily_events:
        events_str = "\n".join([f"- {ev}" for ev in daily_events])
    else:
        events_str = "Nothing significant is happening today."

    # 3. Ad Description
    if ad.description:
        ad_text = ad.description
    else:
        ad_text = (
            f"Visual Style: {ad.visual_style}\n"
            f"Dominant Element: {ad.dominant_element}\n"
            f"Emotion: {ad.emotion_label}\n"
            f"Message: {ad.message_type}"
        )
    # Append raw JSON
    ad_text += f"\n\n Technical Details: {ad.to_message_format()}"

    # 4. Replacement
    prompt = template.replace("{age}", str(agent.age))
    prompt = prompt.replace("{profession}", str(agent.profession))
    prompt = prompt.replace("{persona_narrative}", persona_text)
    prompt = prompt.replace("{current_day}", str(current_day))
    prompt = prompt.replace("{daily_events}", events_str)
    prompt = prompt.replace("{ad_description}", ad_text)
    
    return prompt
