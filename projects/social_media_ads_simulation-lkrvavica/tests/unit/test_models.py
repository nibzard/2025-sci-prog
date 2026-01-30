
import unittest
from src.simulation.models import Ad, User, Event

class TestModels(unittest.TestCase):
    def test_ad_dataclass(self):
        ad = Ad(
            ad_id="ad_test_1",
            group="group_A",
            emotion_label="happy",
            message_type="promo",
            visual_style="vibrant",
            num_people=1,
            people_present=True,
            people_area_ratio=0.5,
            product_present=True,
            product_area_ratio=0.5,
            object_count=1,
            object_list=["phone"],
            dominant_element="product",
            text_present=True,
            text_area_ratio=0.1,
            avg_font_size_proxy=12.0,
            dominant_colors=["red"],
            brightness_category="high",
            saturation_category="high",
            hue_category="warm",
            visual_impact=8.5,
            description="A red phone."
        )
        self.assertEqual(ad.ad_id, "ad_test_1")
        self.assertEqual(ad.description, "A red phone.")
        self.assertEqual(ad.interaction_rate, 0.0) # check default
        
        # Test method
        ad.update_interaction_rate(click=True, share=0, like=False, dislike=False, ignore=False)
        self.assertEqual(ad.interaction_rate, 1.0) # click=1
        
    def test_user_creation(self):
        user = User(
            user_id="u1",
            gender="m",
            age=25,
            profession=["dev"],
            hobby=["code"],
            family="single"
        )
        self.assertEqual(user.user_id, "u1")
        self.assertIn("acute_irritation", user.emotional_state)
        self.assertEqual(user.friend_list, [])
        
    def test_event_creation(self):
        ev = Event({
            "event_id": "e1",
            "text": "Rain",
            "scope": "global"
        })
        self.assertEqual(ev.event_id, "e1")
        self.assertEqual(ev.text, "Rain")
        self.assertEqual(ev.scope, "global")
        self.assertEqual(ev.duration_days, 1) # default

if __name__ == '__main__':
    unittest.main()
