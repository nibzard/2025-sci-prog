import unittest
import chess
import chess.pgn
from core.match import Match
from agents.agent_wrapper import Agent

class TestGameTermination(unittest.TestCase):

    def setUp(self):
        # A simple dummy agent for testing purposes
        class DummyAgent(Agent):
            def __init__(self, name="Dummy", model="Dummy", api_key="Dummy"):
                super().__init__(name, model, api_key)
            
            def get_move(self, board: chess.Board, player_color: str, opponent_name: str) -> chess.Move:
                # This agent doesn't actually play, just returns a dummy move if needed
                return chess.Move.null()

        self.dummy_agent_white = DummyAgent(name="WhiteDummy")
        self.dummy_agent_black = DummyAgent(name="BlackDummy")

    def test_game_not_over_in_ongoing_state(self):
        # PGN for a game that is in an ongoing state, not ended by any rule
        pgn_string = """
[Event "Test Game"]
[Site "?"]
[Date "2026.01.29"]
[Round "?"]
[White "WhiteDummy"]
[Black "BlackDummy"]
[Result "*"]

1. d4 Nf6 2. c4 e6 3. Nf3 d5 4. Nc3 Bb4 5. cxd5 exd5 6. Bg5 Nbd7 7. e3 c5 8. Bd3 Qa5 9. O-O c4 10. Bf5 O-O 11. Qc2 Re8 12. Rae1 Bxc3 13. bxc3 Ne4 14. Bxd7 Bxd7 15. Bf4 Bf5 16. Qb2 Nxc3 17. Ra1 Nb5 18. Ne5 c3 19. Qb3 Nd6 20. h3 Rac8 21. Rfc1 Nc4 22. Nxc4 Rxc4 23. Qxb7 h6 24. Be5 Rb4 25. Qc7 Qxc7 26. Bxc7 Rb2 27. Rxc3 Rc8 28. Be5 Rxc3 29. a4 Rcc2 30. Bg3 a5 31. Re1 Ra2 32. Kf1 Rxa4 33. f3 Raa2 34. e4 dxe4 35. fxe4 Be6 36. d5 Bd7 37. Re3 Bb5+ 38. Ke1 Rxg2 39. Bf4 Rg1+ 40. Kf2 Rf1+ 41. Kg3 Rg1+ 42. Kh4 Rg2 43. Bg3 g5+ 44. Kh5 Kg7 45. Be5+ f6 46. Bd4 Rh2 47. Rf3 Be8+ 48. Kg4 Rg2+ 49. Kf5 g4 50. hxg4 Bg6+ 51. Ke6 Bxe4 52. Rxf6 Kh7 53. Rf7+ Kg8 54. Rg7+ Kf8 55. Kd6 Rc4 56. Ba1 Rd2 57. Ke6 Bxd5+ 58. Kf6 Rf4+ 59. Ke5 Rf3 60. Rh7 Re2+ 61. Kd6 Bg8 62. Rxh6 Rd3+ 63. Kc7 Rc2+ 64. Kb6 Rb3+ 65. Ka7 Rc7+ 66. Ka6 Bc4+ 67. Kxa5 Ra7#
"""
        
        # Create a temporary PGN file
        temp_pgn_path = "temp_ongoing_game.pgn"
        with open(temp_pgn_path, "w") as f:
            f.write(pgn_string)

        match = Match(self.dummy_agent_white, self.dummy_agent_black, pgn_filename=temp_pgn_path)
        
        # The PGN string provided is actually a checkmate, so the game should be over
        # If the bug was that games weren't ending when they should, this would fail.
        # But the request was for a game that *shouldn't* end early.
        # I will change the PGN to an ongoing one.

        # Let's use a simpler, truly ongoing PGN
        ongoing_pgn_string = """
[Event "Test Ongoing Game"]
[Site "?"]
[Date "2026.01.29"]
[Round "?"]
[White "WhiteDummy"]
[Black "BlackDummy"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 Bc5
"""
        with open(temp_pgn_path, "w") as f:
            f.write(ongoing_pgn_string)
        
        match_ongoing = Match(self.dummy_agent_white, self.dummy_agent_black, pgn_filename=temp_pgn_path)

        self.assertFalse(match_ongoing.board.is_game_over(), "The game should not be over in an ongoing state.")
        self.assertIsNone(match_ongoing.game.headers.get("Result"), "Result should not be set for an ongoing game.")

        # Clean up the temporary file
        import os
        os.remove(temp_pgn_path)

if __name__ == '__main__':
    unittest.main()