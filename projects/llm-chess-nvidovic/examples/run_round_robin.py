import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tournament import Tournament

# Load .env file
env_path = Path(__file__).resolve().parent.parent / "enviroment_setup/.env"
load_dotenv(env_path)

def main():
    # API keys are loaded from the .env file
    api_keys = {
        "gemini": os.getenv("GOOGLE_API_KEY"),
        "cohere": os.getenv("COHERE_API_KEY"),
        "groq": os.getenv("GROQ_API_KEY"),
        "reka": os.getenv("REKA_API_KEY"),
    }
    
    # List of all agents to participate in the tournament
    all_agents = ["gemini", "cohere", "groq", "reka", "youchat"]
    agents_with_keys = []
    
    for agent_name in all_agents:
        if agent_name == "youchat": # youchat doesn't need a key
            agents_with_keys.append(agent_name)
        elif api_keys.get(agent_name):
            agents_with_keys.append(agent_name)
        else:
            print(f"Warning: Skipping {agent_name} agent because its API key is not set.")

    if len(agents_with_keys) < 2:
        print("Error: At least two agents with API keys are required to run a tournament.")
        return
        
    tournament = Tournament(agents_with_keys, api_keys)
    
    print("--- LLM Chess Deathmatch ---")
    print("1. Start a new tournament")
    print("2. Resume a tournament")
    print("3. Display schedule")
    print("4. Display standings")
    
    choice = input("Enter your choice: ")
    
    if choice == "1":
        tournament.run()
    elif choice == "2":
        tournament.load_state(api_keys=api_keys)
        tournament.run()
    elif choice == "3":
        # In case the tournament has started, load the state to show the correct status
        tournament.load_state(api_keys=api_keys)
        tournament.display_schedule()
    elif choice == "4":
        tournament.load_state(api_keys=api_keys)
        tournament.display_standings()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
