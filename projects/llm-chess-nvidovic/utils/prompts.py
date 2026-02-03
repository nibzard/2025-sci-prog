def get_move_prompt(fen: str, color: str) -> str:
    """
    Generates a standardized prompt for the LLM to generate a chess move.
    """
    return f"""You are a chess engine.

                Given the position in FEN format:
                {fen}

                Your task:
                - Output ONLY ONE legal chess move.
                - Format MUST be exactly UCI (e.g. "e2e4", "g8f6").
                - NO words, NO sentences, NO punctuation, NO commentary.
                - Output MUST contain ONLY 4 characters (or 5 if promotion, e.g. "e7e8q").

                If you output anything except a single valid UCI move, the response is considered INVALID.
                
                You are playing as {color}.

                Now output the move:
"""
