
import json
import hashlib
import random
import os
from typing import List, Dict, Optional, Any
from src.simulation.models import Event, User
from src.core.config import config

class EventManager:
    def __init__(self):
        self.experiment_id = config.experiment_id
        
        # Load pools
        self.global_pool = self._load_pool("data/events/global_event_pool.json")
        self.group_pool = self._load_pool("data/events/group_event_pool.json")
        self.personal_pool = self._load_pool("data/events/personal_event_pool.json")
        self.calendar = self._load_calendar("data/events/calendar.json")

    def _load_pool(self, filepath):
        if not os.path.exists(filepath):
            print(f"Warning: Event pool not found at {filepath}")
            return []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [Event(d) for d in data]
        except Exception as e:
            print(f"Error loading event pool {filepath}: {e}")
            return []

    def _load_calendar(self, filepath):
        if not os.path.exists(filepath):
            print(f"Warning: Calendar not found at {filepath}")
            return {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Map day -> list of Events
            cal = {}
            for item in data:
                day = item.get("day")
                if day not in cal:
                    cal[day] = []
                cal[day].append(Event(item))
            return cal
        except Exception as e:
            print(f"Error loading calendar {filepath}: {e}")
            return {}

    def get_daily_events(self, day: int, user: User) -> List[str]:
        """
        Main method to get events for a user on a specific day.
        Returns a string list of event descriptions.
        Updates user's active_events and last_event_days.
        """
        
        # 1. Manage Active Events (Duration > 1)
        still_active = []
        active_descriptions = []
        
        for ev in user.active_events:
            end_day = ev.start_day + ev.duration_days - 1
            if day <= end_day:
                still_active.append(ev)
                active_descriptions.append(ev.text)
        
        user.active_events = still_active
        
        new_events = []
        
        # 2. Fixed Events (Calendar) - Scope: GLOBAL (mostly)
        if day in self.calendar:
            for cal_ev in self.calendar[day]:
                cal_ev.start_day = day
                new_events.append(cal_ev)

        # 3. Global Random Events
        global_seed_base = f"{self.experiment_id}|{day}|GLOBAL"
        rng_global = self._get_rng(global_seed_base + "|RANDOM")
        
        if rng_global.random() < 0.5:
             ev = self._select_from_pool(self.global_pool, user, day, rng_global, "global")
             if ev:
                 ev.start_day = day
                 new_events.append(ev)

        # 4. Group Events
        user_group = getattr(user, "group", None)
        if user_group:
            group_seed_base = f"{self.experiment_id}|{day}|GROUP|{user_group}"
            rng_group = self._get_rng(group_seed_base)
            if rng_group.random() < 0.3:
                ev = self._select_from_pool(self.group_pool, user, day, rng_group, "group")
                if ev:
                    ev.start_day = day
                    new_events.append(ev)

        # 5. Personal Events
        personal_seed_base = f"{self.experiment_id}|{day}|PERSONAL|{user.user_id}"
        rng_personal = self._get_rng(personal_seed_base + "|1")
        
        # Mandatory 1
        ev1 = self._select_from_pool(self.personal_pool, user, day, rng_personal, "personal")
        if ev1:
            ev1.start_day = day
            new_events.append(ev1)
            
        # Optional 2nd
        rng_personal_2 = self._get_rng(personal_seed_base + "|2")
        if rng_personal_2.random() < 0.35:
            ev2 = self._select_from_pool(self.personal_pool, user, day, rng_personal_2, "personal")
            if ev2:
                ev2.start_day = day
                new_events.append(ev2)

        # Process New Events
        final_texts = active_descriptions[:]
        
        for ev in new_events:
            # Update User State
            user.last_event_days[ev.event_id] = day
            if ev.duration_days > 1:
                user.active_events.append(ev)
            
            final_texts.append(ev.text)
            
        return final_texts

    def _get_rng(self, seed_str):
        # Deterministic RNG based on sha256 of seed_str
        hash_hex = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
        seed_int = int(hash_hex[:16], 16)
        return random.Random(seed_int)

    def _select_from_pool(self, pool, user, day, rng, scope_debug):
        # Filter cooldowns
        candidates = []
        for ev in pool:
            last_day = user.last_event_days.get(ev.event_id, -999)
            if (day - last_day) >= ev.cooldown_days:
                candidates.append(ev)
        
        if not candidates:
            # Relax cooldowns if empty
            candidates = pool
        
        if not candidates:
            return None
            
        weights = [c.frequency_weight for c in candidates]
        try:
            selected = rng.choices(candidates, weights=weights, k=1)[0]
            # Return new instance
            return Event(selected.to_dict()) 
        except Exception as e:
            return None
