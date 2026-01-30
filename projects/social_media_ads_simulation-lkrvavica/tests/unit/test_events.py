
import unittest
import os
from src.simulation.events import EventManager
from src.simulation.models import Event, User
from src.core.config import config

class TestEvents(unittest.TestCase):
    def setUp(self):
        # Override singleton config for testing
        config.experiment_id = "test_exp"
        config.seed = 42
        
        self.user = User(
            user_id="test_user_001",
            gender="f",
            age=30,
            profession="['writer']",
            hobby="['reading']",
            family="single"
        )
        
        # Initialize manager (will load real pools, but we overwrite them below)
        self.manager = EventManager()
        
        # Mock pools
        self.manager.global_pool = [
            Event({
                "event_id": "GLOB_TEST_01",
                "text": "Global Event 1",
                "scope": "global",
                "frequency_weight": 1.0,
                "cooldown_days": 1
            })
        ]
        
        self.manager.personal_pool = [
            Event({
                "event_id": "PERS_TEST_01",
                "text": "Personal Event 1",
                "scope": "personal",
                "frequency_weight": 1.0,
                "cooldown_days": 1
            })
        ]
        
        self.manager.calendar = {
            5: [Event({"event_id": "CAL_TEST_01", "text": "Payday", "scope": "global"})]
        }

    def test_deterministic_seeding(self):
        """Test that the same user/day produces the same events."""
        events_1 = self.manager.get_daily_events(day=10, user=self.user)
        
        # Reset state
        self.user.active_events = []
        self.user.last_event_days = {}
        
        events_2 = self.manager.get_daily_events(day=10, user=self.user)
        
        self.assertEqual(events_1, events_2)

    def test_calendar_event(self):
        """Test that fixed calendar events appear."""
        events = self.manager.get_daily_events(day=5, user=self.user)
        self.assertTrue(any("Payday" in e for e in events))
        
    def test_cooldowns(self):
        """Test that cooldowns prevent repetition."""
        self.manager.personal_pool[0].cooldown_days = 999
        
        # Day 1
        evs1 = self.manager.get_daily_events(day=1, user=self.user)
        
        # Check if we got the personal event
        got_it = any("Personal Event 1" in e for e in evs1)
        
        if got_it:
            # Day 2 should NOT have it
            evs2 = self.manager.get_daily_events(day=2, user=self.user)
            self.assertFalse(any("Personal Event 1" in e for e in evs2))

if __name__ == '__main__':
    unittest.main()
