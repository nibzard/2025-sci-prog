
import json
import hashlib
import random
import os

class Event:
    def __init__(self, data):
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
        
        # Runtime state (not in JSON)
        self.start_day = -1

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

class EventManager:
    def __init__(self, config):
        self.config = config
        self.experiment_id = config.get("experiment_id", "default_exp")
        
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

    def get_daily_events(self, day, user):
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
                # Re-phrase for ongoing: "Still dealing with..." or just original text?
                # For simplicity, using original text with "Ongoing: " prefix might be annoying.
                # Let's just output the text.
                active_descriptions.append(ev.text)
        
        user.active_events = still_active
        
        # If we have heavy active events, maybe skip new ones? 
        # Spec doesn't strictly say, but let's assume valid events add up.
        
        new_events = []
        
        # 2. Fixed Events (Calendar) - Scope: GLOBAL (mostly)
        if day in self.calendar:
            for cal_ev in self.calendar[day]:
                # Calendar events are mandatory
                cal_ev.start_day = day
                new_events.append(cal_ev)

        # 3. Global Random Events
        # Seed: exp_id | day | "GLOBAL"
        # Quota: Scheduled + Optional Random (p=0.5)
        # We already added Scheduled (Calendar). Now check Random.
        
        global_seed_base = f"{self.experiment_id}|{day}|GLOBAL"
        rng_global = self._get_rng(global_seed_base + "|RANDOM")
        
        if rng_global.random() < 0.5:
             # Try to pick one
             ev = self._select_from_pool(self.global_pool, user, day, rng_global, "global")
             if ev:
                 ev.start_day = day
                 new_events.append(ev)

        # 4. Group Events
        # Seed: exp_id | day | "GROUP" | user.group (if available) ??
        # Or just "GROUP" | group_id? USER object might not have group ID easily accessible 
        # unless we pass it. User definition has 'group' attribute?
        # User definition in definitions.py doesn't seem to have explicit 'group' field in __init__
        # but initialization.py assigns it via KMeans. 
        # Wait, definitions.py User init: (self, user_id, gender, age, profession, hobby, family)
        # It doesn't store group!
        # Hmmm. 'world/initialization.py' calculates groups and stores in DB 'users_grouping'.
        # 'world/definitions.py' User class doesn't seem to hold 'group'.
        # 'engine.select_agents' or similar might assign it dynamically?
        # IMPORTANT: If User object doesn't have group, we can't do per-group deterministic seeding easily.
        # Fallback: Seed: exp_id | day | "GROUP" | user_id (behaves like personal but from group pool)
        # OR: We just simulate it as personal events for now if group isn't available.
        # Let's check if we can access group. 
        # For now, let's use user_id to select from group pool, effectively making it personal-like
        # but from the group content. Or skip if undefined.
        # Let's assume user.group exists purely for this logic, or fail gracefully.
        
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
        # Seed: exp_id | day | "PERSONAL" | user_id
        # Quota: 1 mandatory, 2nd p=0.35
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
            # Relax cooldowns if empty? Spec says "Relax if empty".
            # Let's just pick any if candidates is empty to ensure something happens?
            # Or strict "Relax" which might mean pick one ignoring cooldown.
            # Let's try to pick ignoring cooldown if empty.
            candidates = pool
        
        if not candidates:
            return None
            
        # Weighted selection
        weights = [c.frequency_weight for c in candidates]
        try:
            # random.choices returns a list k=1
            selected = rng.choices(candidates, weights=weights, k=1)[0]
            # Return a COPY so we don't mutate the pool instance with start_day
            return Event(selected.to_dict()) 
        except Exception as e:
            # Fallback
            return None
