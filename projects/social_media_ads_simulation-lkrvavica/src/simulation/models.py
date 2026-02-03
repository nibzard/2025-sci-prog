
import json
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class Ad:
    ad_id: str
    group: str
    emotion_label: str
    message_type: str
    visual_style: str
    num_people: int
    people_present: bool
    people_area_ratio: float
    product_present: bool
    product_area_ratio: float
    object_count: int
    object_list: List[str]
    dominant_element: str
    text_present: bool
    text_area_ratio: float
    avg_font_size_proxy: float
    dominant_colors: List[str]
    brightness_category: str
    saturation_category: str
    hue_category: str
    visual_impact: float
    description: Optional[str] = None
    
    # Runtime state
    day_of_entry: Optional[int] = None
    interaction_rate: float = 0.0
    is_active: bool = False

    def update_interaction_rate(self, click, share, like, dislike, ignore):
        self.interaction_rate += (click + share + 2 * like) - (
            2 * dislike + ignore
        )

    def to_message_format(self):
        return json.dumps(self.__dict__, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

@dataclass
class Event:
    event_id: str
    text: str = ""
    scope: str = "GLOBAL"
    tags: List[str] = field(default_factory=list)
    severity: float = 0.3
    polarity: str = "neutral"
    frequency_weight: float = 0.5
    cooldown_days: int = 0
    duration_days: int = 1
    expected_effects_hint: List[str] = field(default_factory=list)
    start_day: int = -1

    def __init__(self, data: Dict[str, Any]):
        self.event_id = data.get("event_id")
        self.text = data.get("text")
        self.scope = data.get("scope")
        self.tags = data.get("tags", [])
        self.severity = data.get("severity", 0.3)
        self.polarity = data.get("polarity", "neutral")
        self.frequency_weight = data.get("frequency_weight", 0.5)
        self.cooldown_days = data.get("cooldown_days", 0)
        self.duration_days = data.get("duration_days", 1)
        self.expected_effects_hint = data.get("expected_effects_hint", [])
        self.start_day = data.get("start_day", -1)

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "text": self.text,
            "scope": self.scope,
            "tags": self.tags,
            "severity": self.severity,
            "polarity": self.polarity,
            "frequency_weight": self.frequency_weight,
            "cooldown_days": self.cooldown_days,
            "duration_days": self.duration_days,
            "expected_effects_hint": self.expected_effects_hint,
            "start_day": self.start_day
        }

class User:
    def __init__(self, user_id, gender, age, profession, hobby, family):
        self.user_id = user_id
        self.gender = gender
        self.age = age
        self.profession = profession
        self.hobby = hobby
        self.family = family
        self.friend_list = []
        self.persona_narrative = None
        self.assign_propensity_features()
        self.emotional_state = {
            "acute_irritation": 0,
            "acute_interest": 0,
            "acute_arousal": 0,
            "bias_irritation": 0,
            "bias_trust": 0,
            "bias_fatigue": 0,
            "bias_trust": 0,
            "bias_fatigue": 0,
        }
        self.active_events: List[Event] = [] 
        self.last_event_days: Dict[str, int] = {} 
        self.daily_event_summaries: List[str] = []
        self.load_persona()

    def load_persona(self):
        """Load synthesized persona for this user's profession."""
        try:
            mapping_file = "data/persona_mapping.json"
            if not os.path.exists(mapping_file):
                return
            
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            # profession is stored as string representation of list in features CSV?
            # Or list? Logic copied from original.
            if isinstance(self.profession, str):
                try: 
                    job = eval(self.profession)[0] 
                except: 
                    job = self.profession
            else:
                job = self.profession[0]
            
            if job in mapping:
                persona_file = mapping[job]
                if os.path.exists(persona_file):
                    with open(persona_file, 'r', encoding='utf-8') as f:
                        self.persona_narrative = f.read()
        except Exception:
            pass

    def assign_propensity_features(self):
        self.activity_level = np.random.normal(50, 15)
        self.risk_tolerance = np.random.normal(50, 15)
        self.social_engagement = np.random.normal(50, 15)

    def add_friend(self, user_id):
        self.friend_list.append(user_id)

    def to_message_format(self):
        base_data = {
            "user_id": self.user_id,
            "gender": self.gender,
            "age": self.age,
            "profession": self.profession,
            "hobby": self.hobby,
            "family": self.family,
            "activity_level": self.activity_level,
            "risk_tolerance": self.risk_tolerance,
            "social_engagement": self.social_engagement,
            "emotional_state": self.emotional_state,
        }
        
        if self.persona_narrative:
            base_data["persona"] = self.persona_narrative
        
        return json.dumps(base_data, default=str)
