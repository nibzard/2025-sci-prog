from pathlib import Path
import os
from dotenv import load_dotenv
from agents.gemini import GeminiAgent

# Load .env file
env_path = Path(__file__).resolve().parent.parent / "enviroment_setup/.env"
load_dotenv(env_path)

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ GOOGLE_API_KEY nije učitan!")

# Test
print("Running GeminiAgent tests...")
agent = GeminiAgent(api_key=api_key)
print(agent.check_is_alive())

fen = "r1bqk2r/2pp1ppp/p1n2n2/1pb1p3/4P3/1B1P1N2/PPP2PPP/RNBQ1RK1 b KQkq - 0 1"
move = agent.get_move(fen, "white")
print(f"Generated move for FEN '{fen}': {move}")
print("GeminiAgent tests completed.")
