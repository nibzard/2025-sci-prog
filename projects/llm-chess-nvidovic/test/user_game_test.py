from core.match import Match
from agents.youchat import YouChatAgent

# A simple class for the human player.
# The play_with_user method identifies the human player
# by checking for the absence of a 'get_move' method.
class HumanPlayer:
    def __init__(self, name="Human"):
        self.name = name

# Set up the players. Human can be white or black.
# To play as black, swap the players in the Match constructor.
human_player = HumanPlayer()
ai_player = YouChatAgent()

# Create the match with Human as White.
match = Match(player_white=human_player, player_black=ai_player)

# Start the game.
print("You are playing as White against the AI. Enter moves in UCI format (e.g. e2e4).")
match.play_with_user()

print(f"Game over! Result: {match.result}")
