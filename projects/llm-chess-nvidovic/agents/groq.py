from ai4free import GROQ
from utils.prompts import get_move_prompt

class GroqAgent:

    """Agent which uses Groq API to generate chess moves."""
    def __init__(self, api_key=None):
        self.groq = GROQ(api_key=api_key, model="llama-3.3-70b-versatile") # openai/gpt-oss-20b
        
    def get_move(self, fen: str, color: str, forced_prompt: str = None, player_name: str = None) -> str:
        try:
            if forced_prompt:
                prompt = forced_prompt
            else:
                from utils.prompts import get_move_prompt
                prompt = get_move_prompt(fen, color)
            response = self.groq.chat(prompt)
            print(f"{player_name} raw response: {response}")
            return response
        except Exception as e:
            return f"ERROR from Groq: {e}"
        
    def check_is_alive(self):
        return "I'm alive!"
