from reka.client import Reka
from utils.prompts import get_move_prompt

class RekaAgent:

    """Agent which uses REKA API to generate chess moves."""
    def __init__(self, api_key=None):
        self.API_key = api_key
        self.reka = Reka(api_key=api_key)

    def get_move(self, fen: str, color: str, forced_prompt: str = None, player_name: str = None) -> str:
        try:
            if forced_prompt:
                prompt = forced_prompt
            else:
                from utils.prompts import get_move_prompt
                prompt = get_move_prompt(fen, color)
            response = self.reka.chat.create(messages=[{"content":f"{prompt}", "role":"user"}], model="reka-core-20240501")
            raw_response = response.responses[0].message.content
            print(f"{player_name} raw response: {raw_response}")
            return raw_response
        except Exception as e:
            return f"ERROR from REKA: {e}"
        
    def check_is_alive(self):
        return "I'm alive!"
