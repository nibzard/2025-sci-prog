from pathlib import Path
import os
from dotenv import load_dotenv
from agents.cohere import CohereAgent

# Load .env file
env_path = Path(__file__).resolve().parent.parent / "enviroment_setup/.env"
load_dotenv(env_path)

# Get API key
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ COHERE_API_KEY nije učitan!")

# Test
print("Running CohereAgent tests...")
agent = CohereAgent(api_key=api_key)
print(agent.check_is_alive())

fen = "r1bqk2r/2pp1ppp/p1n2n2/1pb1p3/4P3/1B1P1N2/PPP2PPP/RNBQ1RK1 b KQkq - 0 1"
move = agent.get_move(fen, "white")
print(f"Generated move for FEN '{fen}': {move}")
print("CohereAgent tests completed.")
