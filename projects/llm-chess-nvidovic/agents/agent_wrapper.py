from agents.cohere import CohereAgent
from agents.gemini import GeminiAgent
from agents.groq import GroqAgent
from agents.reka import RekaAgent
from agents.youchat import YouChatAgent

class AgentWrapper:
    def __init__(self, agent_name, api_key=None):
        self.agent_name = agent_name
        self.api_key = api_key
        self.agent = self._get_agent()
        self.name = agent_name

    def _get_agent(self):
        if self.agent_name == "gemini":
            return GeminiAgent(api_key=self.api_key)
        elif self.agent_name == "cohere":
            return CohereAgent(api_key=self.api_key)
        elif self.agent_name == "groq":
            return GroqAgent(api_key=self.api_key)
        elif self.agent_name == "reka":
            return RekaAgent(api_key=self.api_key)
        elif self.agent_name == "youchat":
            return YouChatAgent()
        else:
            raise ValueError(f"Unknown agent: {self.agent_name}")

    def get_move(self, fen, color, forced_prompt=None, player_name=None):
        return self.agent.get_move(fen, color, forced_prompt=forced_prompt, player_name=player_name or self.name)
