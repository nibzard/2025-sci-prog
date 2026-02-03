import os
import json
import datetime

def log_illegal_move(log_file, player_name, opponent_name, board_fen, attempted_move, move_number):
    """Logs illegal move data to a file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "player": player_name,
        "opponent": opponent_name,
        "move_number": move_number,
        "board_fen": board_fen,
        "attempted_move_uci": str(attempted_move) if attempted_move is not None else None,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
