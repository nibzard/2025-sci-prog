import itertools
import json
import os
from agents.agent_wrapper import AgentWrapper
from core.match import Match

class Tournament:
    def __init__(self, agents, api_keys=None):
        self.agents = [AgentWrapper(agent, api_key=api_keys.get(agent) if api_keys else None) for agent in agents]
        self.schedule = self._generate_schedule()
        self.scores = {agent.name: 0 for agent in self.agents}
        self.tournament_dir = "gamedata/tournament"
        os.makedirs(self.tournament_dir, exist_ok=True)

    def _generate_schedule(self):
        schedule = []
        match_id = 0
        # Quadruple round-robin
        for _ in range(2):
            for player1, player2 in itertools.permutations(self.agents, 2):
                schedule.append({"id": match_id, "white": player1, "black": player2, "played": False, "result": None})
                match_id += 1
        return schedule

    def run(self):
        for match_info in self.schedule:
            if not match_info["played"]:
                white_player = match_info["white"]
                black_player = match_info["black"]
                
                print(f"Playing match: {white_player.name} (white) vs {black_player.name} (black)")
                existing_pgn = match_info.get("pgn_filename")
                match = Match(white_player, black_player, pgn_filename=existing_pgn)
                result = match.test_play()
                pgn_filename = match.pgn_filename # Retrieve pgn_filename
                
                self._update_scores(result, white_player, black_player)
                
                match_info["played"] = True
                match_info["result"] = result
                match_info["pgn_filename"] = pgn_filename # Store pgn_filename
                
                self.save_state()

    def _update_scores(self, result, white_player, black_player):
        if result == "1-0":
            self.scores[white_player.name] += 1
        elif result == "0-1":
            self.scores[black_player.name] += 1
        elif result == "1/2-1/2":
            self.scores[white_player.name] += 0.5
            self.scores[black_player.name] += 0.5

    def save_state(self, filename="tournament_state.json"):
        state = {
            "agents": [agent.name for agent in self.agents],
            "schedule": [
                {
                    "id": m["id"],
                    "white": m["white"].name,
                    "black": m["black"].name,
                    "played": m["played"],
                    "result": m["result"],
                    "pgn_filename": m.get("pgn_filename"), # Include pgn_filename
                }
                for m in self.schedule
            ],
            "scores": self.scores,
        }
        filepath = os.path.join(self.tournament_dir, filename)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=4)

    def load_state(self, filename="tournament_state.json", api_keys=None):
        filepath = os.path.join(self.tournament_dir, filename)
        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            state = json.load(f)

        self.agents = [AgentWrapper(name, api_key=api_keys.get(name) if api_keys else None) for name in state["agents"]]
        agent_map = {agent.name: agent for agent in self.agents}

        self.schedule = []
        for m_data in state["schedule"]:
            white = agent_map.get(m_data["white"])
            black = agent_map.get(m_data["black"])
            if white and black:
                self.schedule.append({
                    "id": m_data["id"],
                    "white": white,
                    "black": black,
                    "played": m_data["played"],
                    "result": m_data["result"],
                    "pgn_filename": m_data.get("pgn_filename"), # Load pgn_filename
                })
        
        self.scores = state["scores"]

    def display_schedule(self):
        print("--- Tournament Schedule ---")
        for match_info in self.schedule:
            status = "Played" if match_info["played"] else "To be played"
            result = f"({match_info['result']})" if match_info['result'] else ""
            pgn_info = f" (PGN: {match_info['pgn_filename']})" if match_info.get("pgn_filename") else ""
            print(
                f"Match {match_info['id']}: {match_info['white'].name} vs {match_info['black'].name} - {status} {result}{pgn_info}"
            )

    def display_played_games(self):
        print("--- Played Games ---")
        for match_info in self.schedule:
            if match_info["played"]:
                pgn_info = f" (PGN: {match_info['pgn_filename']})" if match_info.get("pgn_filename") else ""
                print(
                    f"Match {match_info['id']}: {match_info['white'].name} vs {match_info['black'].name} - Result: {match_info['result']}{pgn_info}"
                )

    def display_standings(self):
        print("--- Tournament Standings ---")
        # Sort scores in descending order
        sorted_scores = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        for agent_name, score in sorted_scores:
            print(f"{agent_name}: {score}")
