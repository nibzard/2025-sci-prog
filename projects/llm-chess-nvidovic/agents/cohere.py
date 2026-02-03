import cohere
from utils.prompts import get_move_prompt

class CohereAgent:

    """Agent which uses Cohere API to generate chess moves."""
    def __init__(self, api_key):
        self.API_key = api_key
        self.cohere = cohere.Client(api_key=api_key)

    def get_move(self, fen: str, color: str, forced_prompt: str = None, player_name: str = None) -> str:
        try:
            if forced_prompt:
                prompt = forced_prompt
            else:
                from utils.prompts import get_move_prompt
                prompt = get_move_prompt(fen, color)
            response = self.cohere.chat(model="command-a-03-2025", message=prompt, max_tokens=10).text
            print(f"{player_name} raw response: {response}")
            return response
        except Exception as e:
            return f"ERROR from Cohere: {e}"
        
    def check_is_alive(self):
        return "I'm alive!"
