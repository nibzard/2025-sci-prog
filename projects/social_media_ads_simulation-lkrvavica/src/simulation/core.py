
from typing import Dict, Any
from src.simulation.models import User

def validate_and_sanitize_interaction(data: Dict[str, Any]) -> Dict[str, Any]:
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
    if data['ignore']:
        if data['click'] or data['like'] or data['dislike'] or data['share']:
            # Conflict: Ignore wins
            data['click'] = False
            data['like'] = False
            data['dislike'] = False
            data['share'] = 0
    
    # 3. Logic Check: At least one action
    has_action = data['ignore'] or data['click'] or data['like'] or data['dislike'] or data['share']
    if not has_action:
        data['ignore'] = True

    return data

def update_user_emotional_state(user: User, response_json: Dict[str, Any]):
    """Update user emotional state in-place based on interaction response."""
    keys = ["acute_irritation", "acute_interest", "acute_arousal", "bias_irritation", "bias_trust", "bias_fatigue"]
    for key in keys:
        change = response_json.get(f"{key}_change", 0)
        # Assuming emotional state is 0-100
        current = user.emotional_state.get(key, 0)
        user.emotional_state[key] = max(0, min(100, current + change))
