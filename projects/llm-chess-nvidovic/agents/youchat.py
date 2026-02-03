from ai4free import YouChat
from utils.prompts import get_move_prompt

class YouChatAgent:
    """Agent which uses YouChat API to generate chess moves."""
    
    
    def __init__(self):
        self.youchat = YouChat()
        self.name = "YouChat Agent"

    def get_move(self, fen: str, color: str, forced_prompt: str = None, player_name: str = None) -> str:
        try:
            if forced_prompt:
                prompt = forced_prompt
            else:
                from utils.prompts import get_move_prompt
                prompt = get_move_prompt(fen, color)
            
            response = self.youchat.chat(prompt)
            print(f"{player_name} raw response: {response}")
            return response
        except Exception as e:
            return f"ERROR from YouChat: {e}"
        
    def check_is_alive(self):
        return "I'm alive!"