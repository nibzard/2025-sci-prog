import chess

class GameAnalyzer:
    def __init__(self, board):
        self.board = board

    def evaluate_position(self):
        # Simple material evaluation
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        white_score = 0
        black_score = 0

        for piece_type in piece_values:
            white_pieces = len(self.board.pieces(piece_type, chess.WHITE))
            black_pieces = len(self.board.pieces(piece_type, chess.BLACK))
            white_score += white_pieces * piece_values[piece_type]
            black_score += black_pieces * piece_values[piece_type]

        return white_score - black_score

    def is_checkmate(self):
        return self.board.is_checkmate()

    def is_stalemate(self):
        return self.board.is_stalemate()