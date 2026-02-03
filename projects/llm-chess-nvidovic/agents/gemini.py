import google.generativeai as genai
from utils.prompts import get_move_prompt

class GeminiAgent:

    """Agent which uses Google Gemini API to generate chess moves."""
    def __init__(self, api_key=None):
        self.API_key = api_key
        
    def get_move(self, fen: str, color: str, forced_prompt: str = None, player_name: str = None) -> str:
        if not self.API_key:
            return "ERROR: GOOGLE_API_KEY not set."
        try:
            genai.configure(api_key=self.API_key)

            if forced_prompt:
                prompt = forced_prompt
                use_tools = False
            else:
                from utils.prompts import get_move_prompt
                prompt = get_move_prompt(fen, color)
                use_tools = True

            if use_tools:
                # For now, since tools are complex, just use text
                model = genai.GenerativeModel("gemma-3-27b-it")
                resp = model.generate_content(prompt)
                move = resp.text.strip()
                print(f"{player_name} raw response: {move}")
                return move
            else:
                model = genai.GenerativeModel("gemma-3-27b-it")
                resp = model.generate_content(prompt)
                move = resp.text.strip()
                print(f"{player_name} raw response: {move}")
                return move

        except Exception as e:
            return f"ERROR from Gemini: {e}"
        
    def check_is_alive(self):
        return "I'm alive!"

