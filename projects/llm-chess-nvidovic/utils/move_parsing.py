import time
import chess
import random
from utils.logging import log_illegal_move

MAX_ASSISTED_ATTEMPTS = 3
MAX_LEGAL_MOVES_IN_PROMPT = 30

def parse_and_validate_move(board, move_uci):
    """
    Parses a move in UCI format and validates if it is a legal move.
    Returns the move object if it is valid, otherwise returns None.
    """
    if not move_uci:
        return None
    try:
        candidate_move = chess.Move.from_uci(move_uci.strip())
        if candidate_move in board.legal_moves:
            return candidate_move
    except (ValueError, TypeError):
        return None
    return None

def get_move_with_recovery(board, player, player_color, log_file=None, opponent_name=None):
    """
    Gets a legal move from an LLM player using a 2-phase approach.
    Phase 1: Free attempt without hints.
    Phase 2: Assisted attempt with a list of legal moves.
    Retries Phase 2 up to MAX_ASSISTED_ATTEMPTS times.
    Raises an exception if the LLM fails to produce a valid move.
    """
    # Phase 1: Free attempt
    board_fen = board.fen()
    move_uci = player.get_move(board_fen, player_color, player_name=player.name)
    if move_uci and move_uci.startswith("ERROR"):
        raise RuntimeError(f"{player.name} encountered an error: {move_uci}")
    move = parse_and_validate_move(board, move_uci)

    if move:
        return move
    else:
        if log_file and opponent_name:
            move_number = board.ply() // 2 + 1
            log_illegal_move(log_file, player.name, opponent_name, board_fen, move_uci, move_number)

    # Phase 2: Assisted attempt
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [m.uci() for m in legal_moves]

    # Limit the number of moves in the prompt to save tokens
    if len(legal_moves_uci) > MAX_LEGAL_MOVES_IN_PROMPT:
        legal_moves_uci_prompt = legal_moves_uci[:MAX_LEGAL_MOVES_IN_PROMPT]
    else:
        legal_moves_uci_prompt = legal_moves_uci
    
    legal_moves_str = ", ".join(legal_moves_uci_prompt)
    
    for i in range(MAX_ASSISTED_ATTEMPTS):
        prompt = (
            f"The previous move was illegal. Choose exactly one move from this list: {legal_moves_str}. "
            "Do not provide any other text, only the move in UCI format."
        )
        time.sleep(12)  # Brief pause to avoid rate limiting
        move_uci = player.get_move(board.fen(), player_color, forced_prompt=prompt, player_name=player.name)
        if move_uci and move_uci.startswith("ERROR"):
            raise RuntimeError(f"{player.name} encountered an error during assisted attempt: {move_uci}")
        move = parse_and_validate_move(board, move_uci)

        if move and move.uci() in legal_moves_uci:
            return move
        else:
            if log_file and opponent_name:
                move_number = board.ply() // 2 + 1
                log_illegal_move(log_file, player.name, opponent_name, board.fen(), move_uci, move_number)

    # If all attempts fail, return a random legal move
    print(f"{player.name} failed to produce a legal move after {MAX_ASSISTED_ATTEMPTS} assisted attempts. Making a random move.")
    return random.choice(list(board.legal_moves))
