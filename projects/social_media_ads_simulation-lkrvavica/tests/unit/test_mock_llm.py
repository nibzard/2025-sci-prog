
import unittest
from unittest.mock import patch, MagicMock
from src.services.llm import LLMService

class TestLLMService(unittest.TestCase):
    def setUp(self):
        self.service = LLMService()

    @patch("src.services.llm.requests.post")
    def test_query_openai(self, mock_post):
        # Mock response
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"test": "ok"}'}}]
        }
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        # Call
        with patch("src.services.llm.config.openai_api_key", "sk-test"):
            res = self.service.query("Hello", backend="openai", model_name="gpt-4o")
            
        self.assertEqual(res, '{"test": "ok"}')
        mock_post.assert_called_once()

    def test_clean_json(self):
        dirty = "```json\n{\"a\": 1}\n```"
        clean = self.service.clean_json_response(dirty)
        self.assertEqual(clean, '{"a": 1}')
        
    def test_clean_json_empty(self):
        res = self.service.clean_json_response("")
        self.assertEqual(res, "{}")

if __name__ == '__main__':
    unittest.main()
