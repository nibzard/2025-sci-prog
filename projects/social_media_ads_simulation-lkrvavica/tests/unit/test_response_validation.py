
import unittest
from src.simulation.core import validate_and_sanitize_interaction

class TestResponseValidation(unittest.TestCase):
    
    def test_basic_sanitize(self):
        # Missing fields should be filled with defaults
        data = {"click": True}
        cleaned = validate_and_sanitize_interaction(data)
        self.assertFalse(cleaned["ignore"])
        self.assertTrue(cleaned["click"])
        self.assertEqual(cleaned["share"], 0)
        self.assertEqual(cleaned["reaction_description"], "")

    def test_ignore_conflicts(self):
        # Ignore + Click -> Ignore should win, Click should be False
        data = {"ignore": True, "click": True, "reaction_description": "Ignoring this"}
        cleaned = validate_and_sanitize_interaction(data)
        self.assertTrue(cleaned["ignore"])
        self.assertFalse(cleaned["click"])

    def test_ignore_conflicts_share(self):
        # Ignore + Share -> Ignore should win
        data = {"ignore": True, "share": 1}
        cleaned = validate_and_sanitize_interaction(data)
        self.assertTrue(cleaned["ignore"])
        self.assertEqual(cleaned["share"], 0)

    def test_no_action_fallback(self):
        # No action selected -> Default to Ignore
        data = {"reaction_description": "Meh"}
        cleaned = validate_and_sanitize_interaction(data)
        self.assertTrue(cleaned["ignore"])

    def test_share_type_correction(self):
        # Share as boolean -> int
        data = {"share": True, "click": True}
        cleaned = validate_and_sanitize_interaction(data)
        self.assertEqual(cleaned["share"], 1)

    def test_valid_complex_interaction(self):
        # Click + Like + Share -> Allowed
        data = {"click": True, "like": True, "share": 1}
        cleaned = validate_and_sanitize_interaction(data)
        self.assertTrue(cleaned["click"])
        self.assertTrue(cleaned["like"])
        self.assertEqual(cleaned["share"], 1)
        self.assertFalse(cleaned["ignore"])

if __name__ == '__main__':
    unittest.main()
