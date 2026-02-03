import chess
import chess.pgn
import datetime
import re
import time
import os
from utils.logging import log_illegal_move
from utils.move_parsing import get_move_with_recovery, parse_and_validate_move
from utils.pgn_tools import save_pgn

class Match:
    def __init__(self, player_white, player_black, pgn_filename=None):
        self.player_white = player_white
        self.player_black = player_black
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.moves = []
        self.result = None

        self.game.headers["Event"] = "AI Chess Match"
        self.game.headers["White"] = player_white.name
        self.game.headers["Black"] = player_black.name
        self.game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")

        # Sanitize player names for filename and add a timestamp
        white_name = re.sub(r'[^\w\-.]', '_', player_white.name)
        black_name = re.sub(r'[^\w\-.]', '_', player_black.name)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.illegal_move_log_file = f"gamedata/logs/{white_name}_vs_{black_name}_{timestamp}_illegal_moves.log"
        self.pgn_filename = pgn_filename or f"gamedata/pgns/{white_name}_vs_{black_name}_{timestamp}.pgn"

        if pgn_filename and os.path.exists(pgn_filename):
            self.load_from_pgn(pgn_filename)
        
        self.current_node = self.game

    def load_from_pgn(self, pgn_filename):
        with open(pgn_filename, 'r') as f:
            self.game = chess.pgn.read_game(f)
        self.board = self.game.board()
        self.current_node = self.game # Initialize current_node to the root of the loaded game
        for move in self.game.mainline_moves():
            self.board.push(move)
            self.moves.append(move)
            self.current_node = self.current_node.add_variation(move) # Advance current_node along with the moves
        self.result = self.game.headers.get("Result", None)

    def board_str(self):
        return str(self.board)

    def game_state(self):
        return self.board.fen()

    def is_move_legal(self, move):
        return move in self.board.legal_moves

    def play_move(self, move):
        self.board.push(move)
        self.moves.append(move)
        
        self.current_node = self.current_node.add_variation(move)

        if self.board.is_stalemate():
            self.game.headers["Result"] = "1/2-1/2"
            print("Game over: Stalemate!")
        elif self.board.can_claim_threefold_repetition():
            self.game.headers["Result"] = "1/2-1/2"
            print("Game over: Threefold repetition!")
        elif self.board.is_insufficient_material():
            self.game.headers["Result"] = "1/2-1/2"
            print("Game over: Insufficient material!")
        
        return self.board.is_game_over()

    def test_play(self):
        current_player = self.player_white
        while not self.board.is_game_over(claim_draw=True):
            player_color_str = "white" if current_player == self.player_white else "black"
            opponent_name = self.player_black.name if current_player == self.player_white else self.player_white.name
            move = get_move_with_recovery(self.board, current_player, player_color_str, self.illegal_move_log_file, opponent_name)

            self.board.push(move)
            self.current_node = self.current_node.add_variation(move)
            print(f"{current_player.name} plays: {move.uci()}\n{self.board_str()}\n")

            current_player = self.player_black if current_player == self.player_white else self.player_white
            time.sleep(15)
        self.result = self.board.result()
        save_pgn(self.pgn_filename, self.game)
        return self.result

    def play_with_user(self):
        current_player = self.player_white

        while not self.board.is_game_over(claim_draw=True):
            player_color_str = "white" if current_player == self.player_white else "black"
            move = None

            if not hasattr(current_player, 'get_move'):
                valid_input = False
                while not valid_input:
                    move_uci = input(f"Enter your move for {player_color_str} (in UCI format, e.g. e2e4): ")
                    move = parse_and_validate_move(self.board, move_uci)
                    if move:
                        valid_input = True
                    else:
                        print("Invalid or illegal move. Try again.")
            else: # AI player
                opponent_name = self.player_black.name if current_player == self.player_white else self.player_white.name
                move = get_move_with_recovery(self.board, current_player, player_color_str, self.illegal_move_log_file, opponent_name)
            
            self.board.push(move)
            self.current_node = self.current_node.add_variation(move)
            print(f"{current_player.name} plays: {move.uci()}\n{self.board_str()}\n")

            current_player = self.player_black if current_player == self.player_white else self.player_white
        
        self.result = self.board.result()
        print(f"Game over. Result: {self.result}")
        user_pgn_filename = self.pgn_filename.replace(".pgn", "_with_user.pgn")
        save_pgn(user_pgn_filename, self.game)
