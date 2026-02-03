from core.match import Match
from agents.youchat import YouChatAgent

youchat1 = YouChatAgent()

youchat2 = YouChatAgent()

match = Match(youchat1, youchat2)

match.test_play()

print(f"Game over! Result: {match.result}")

with open(f"youchat_vs_youchat_game.txt", "w") as f:
    f.write("PGN of the game between YouChat Agent 1 and YouChat Agent 2\n\n")
    f.write(match.board.move_stack.__str__())
    f.close()