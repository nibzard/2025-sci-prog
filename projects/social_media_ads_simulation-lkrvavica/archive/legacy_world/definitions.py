import json
import numpy as np

class Ad:
    def __init__(
        self,
        ad_id,
        group,
        emotion_label,
        message_type,
        visual_style,
        num_people,
        people_present,
        people_area_ratio,
        product_present,
        product_area_ratio,
        object_count,
        object_list,
        dominant_element,
        text_present,
        text_area_ratio,
        avg_font_size_proxy,
        dominant_colors,
        brightness_category,
        saturation_category,
        hue_category,
        visual_impact,
        description=None,
    ):
        self.ad_id = ad_id
        self.group = group
        self.emotion_label = emotion_label
        self.message_type = message_type
        self.visual_style = visual_style
        self.num_people = num_people
        self.people_present = people_present
        self.people_area_ratio = people_area_ratio
        self.product_present = product_present
        self.product_area_ratio = product_area_ratio
        self.object_count = object_count
        self.object_list = object_list
        self.dominant_element = dominant_element
        self.text_present = text_present
        self.text_area_ratio = text_area_ratio
        self.avg_font_size_proxy = avg_font_size_proxy
        self.dominant_colors = dominant_colors
        self.brightness_category = brightness_category
        self.saturation_category = saturation_category
        self.hue_category = hue_category
        self.visual_impact = visual_impact
        self.description = description

        self.day_of_entry = None
        self.interaction_rate = 0
        self.is_active = False

    def update_interaction_rate(self, click, share, like, dislike, ignore):
        self.interaction_rate += (click + share + 2 * like) - (
            2 * dislike + ignore
        )

    def to_message_format(self):
        return json.dumps(self.__dict__)

class User:
    def __init__(self, user_id, gender, age, profession, hobby, family):
        self.user_id = user_id
        self.gender = gender
        self.age = age
        self.profession = profession
        self.hobby = hobby
        self.family = family
        self.friend_list = []
        self.persona_narrative = None  # NEW: Full 300-500 word bio
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
        self.active_events = [] # For multi-day events
        self.last_event_days = {} # For cooldown tracking
        self.load_persona()  # NEW: Load persona after initialization

    def load_persona(self):
        """Load synthesized persona for this user's profession."""
        import os
        try:
            mapping_file = "data/persona_mapping.json"
            if not os.path.exists(mapping_file):
                return
            
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            # profession is stored as ['writer'], extract first
            job = eval(self.profession)[0] if isinstance(self.profession, str) else self.profession[0]
            
            if job in mapping:
                persona_file = mapping[job]
                if os.path.exists(persona_file):
                    with open(persona_file, 'r', encoding='utf-8') as f:
                        self.persona_narrative = f.read()
        except Exception as e:
            # Silently fail if persona loading doesn't work
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
        
        return json.dumps(base_data)
