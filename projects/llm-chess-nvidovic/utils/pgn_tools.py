import os

def save_pgn(filename, game):
    """Saves the game to a PGN file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as pgn_file:
        print(game, file=pgn_file, end="\n\n")
