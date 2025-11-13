# Tinker Implementation Guide for LLM Chess Battle

## Overview

This guide provides a complete implementation plan for integrating Tinker into the LLM Chess Battle project to create a specialized chess-playing model using Reinforcement Learning with Verifiable Rewards (RLVR).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Chess Battle + Tinker                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │   Phase 1:   │ ───> │   Phase 2:   │                   │
│  │ Supervised   │      │   RLVR with  │                   │
│  │  Learning    │      │  Stockfish   │                   │
│  └──────────────┘      └──────────────┘                   │
│         │                      │                            │
│         │                      │                            │
│  ┌──────▼──────────────────────▼─────┐                    │
│  │   Fine-tuned Chess Model          │                    │
│  │   (Qwen3-8B with LoRA)           │                    │
│  └──────────────────────────────────┘                     │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────┐                     │
│  │   Steel Browser Integration      │                     │
│  │   (Existing chess.com interface) │                     │
│  └──────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Supervised Learning Setup

### 1.1 Install Dependencies

```bash
# Install Tinker SDK
pip install git+https://github.com/thinking-machines/tinker-sdk
pip install git+https://github.com/thinking-machines/tinker-cookbook

# Install chess libraries
pip install python-chess
pip install chess  # PGN parsing
pip install stockfish  # For evaluation

# Set API key
export TINKER_API_KEY="your-tinker-api-key-here"
```

### 1.2 Data Collection

Download high-quality chess games in PGN format:

```python
# datasets/collect_chess_data.py

import chess.pgn
import requests
from pathlib import Path

def download_lichess_games(rating_threshold=2000, num_games=10000):
    """
    Download chess games from Lichess database
    Focus on high-rated players (>2000 ELO)
    """
    # Lichess has free database dumps
    # https://database.lichess.org/

    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-11.pgn.zst"

    # Download and filter games
    games = []

    # TODO: Implement download and filtering
    # Filter criteria:
    # - Both players > 2000 ELO
    # - Time control: Standard/Classical
    # - Termination: Normal (not by time forfeit)

    return games

def parse_pgn_to_training_data(pgn_file):
    """
    Convert PGN games to training format for Tinker
    """
    training_examples = []

    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()

            for move in game.mainline_moves():
                # Create training example
                position_fen = board.fen()
                move_uci = move.uci()

                # Format as chat conversation
                training_examples.append({
                    "messages": [
                        {
                            "role": "user",
                            "content": f"You are playing chess. Current position (FEN): {position_fen}\n\nWhat is your next move? Respond with only the move in UCI format (e.g., 'e2e4')."
                        },
                        {
                            "role": "assistant",
                            "content": move_uci
                        }
                    ]
                })

                board.push(move)

    return training_examples

if __name__ == "__main__":
    # Download games
    games = download_lichess_games(rating_threshold=2200, num_games=50000)

    # Parse to training data
    training_data = parse_pgn_to_training_data("games.pgn")

    # Save for Tinker
    import json
    with open("chess_training_data.jsonl", "w") as f:
        for example in training_data:
            f.write(json.dumps(example) + "\n")

    print(f"Created {len(training_data)} training examples")
```

### 1.3 Supervised Fine-tuning

```python
# train/supervised_finetune.py

import tinker
from tinker import AsyncClient
import json

# Initialize Tinker client
client = AsyncClient()

# Load training data
def load_chess_data():
    with open("chess_training_data.jsonl", "r") as f:
        for line in f:
            yield json.loads(line)

# Configure training
async def train_chess_model():
    # Choose model: Qwen3-8B (good balance of performance/cost)
    model_name = "qwen3-8b-instruct"

    # Create training configuration
    training_config = {
        "model": model_name,
        "lora_rank": 32,  # LoRA rank (higher = more capacity)
        "lora_alpha": 64,
        "learning_rate": 2e-4,  # LoRA can use higher LR
        "batch_size": 8,
        "num_epochs": 3,
    }

    # Training loop
    data_iterator = load_chess_data()

    step = 0
    for batch in batched(data_iterator, training_config["batch_size"]):
        # Forward and backward pass
        loss = await client.forward_backward(
            messages=batch,
            loss_type="cross_entropy"
        )

        # Update weights
        await client.optim_step()

        step += 1

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss}")

            # Save checkpoint
            await client.save_state(f"checkpoints/chess_sl_step_{step}")

        if step >= 10000:  # ~80k examples with batch size 8
            break

    # Save final model
    await client.save_weights_for_sampler("models/chess_sl_final")
    print("Supervised training complete!")

def batched(iterable, n):
    """Batch data into chunks of size n"""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

if __name__ == "__main__":
    import asyncio
    asyncio.run(train_chess_model())
```

---

## Phase 2: RLVR with Stockfish

### 2.1 Reward Function Design

```python
# reward/chess_rewards.py

import chess
import chess.engine
from pathlib import Path

class ChessRewardFunction:
    def __init__(self, stockfish_path="/usr/games/stockfish"):
        """
        Initialize reward function with Stockfish engine
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Threads": 4, "Hash": 2048})

    def evaluate_position(self, fen, depth=15):
        """
        Evaluate position using Stockfish
        Returns centipawn score from current player's perspective
        """
        board = chess.Board(fen)

        info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].relative.score(mate_score=10000)

        return score / 100.0  # Convert to pawns

    def evaluate_move(self, fen_before, move_uci):
        """
        Evaluate the quality of a move
        Returns reward signal for RLVR
        """
        board = chess.Board(fen_before)

        # Check if move is legal
        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                return -10.0  # Heavy penalty for illegal moves
        except:
            return -10.0

        # Evaluate position before move
        score_before = self.evaluate_position(fen_before)

        # Apply move
        board.push(move)
        fen_after = board.fen()

        # Evaluate position after move (from opponent's perspective, so negate)
        score_after = -self.evaluate_position(fen_after)

        # Reward is improvement in position
        improvement = score_after - score_before

        # Bonus for checkmate
        if board.is_checkmate():
            improvement += 50.0

        # Bonus for check
        elif board.is_check():
            improvement += 0.5

        # Penalty for stalemate (draw when winning)
        if board.is_stalemate() and score_before > 2.0:
            improvement -= 5.0

        return improvement

    def get_best_move(self, fen, depth=20):
        """
        Get Stockfish's best move (for comparison)
        """
        board = chess.Board(fen)
        result = self.engine.play(board, chess.engine.Limit(depth=depth))
        return result.move.uci()

    def close(self):
        self.engine.quit()
```

### 2.2 RLVR Training Loop

```python
# train/rlvr_train.py

import tinker
from tinker import AsyncClient
import chess
import random
from reward.chess_rewards import ChessRewardFunction

async def rlvr_training():
    client = AsyncClient()
    reward_fn = ChessRewardFunction()

    # Load supervised model checkpoint
    await client.load_state("models/chess_sl_final")

    # Training configuration
    config = {
        "learning_rate": 5e-5,  # Lower LR for RL fine-tuning
        "episodes_per_batch": 16,
        "max_moves_per_game": 80,
    }

    episode = 0

    while episode < 1000:  # 1000 training episodes
        batch_rewards = []
        batch_trajectories = []

        for _ in range(config["episodes_per_batch"]):
            # Play one game
            trajectory, total_reward = await play_chess_episode(client, reward_fn)

            batch_trajectories.append(trajectory)
            batch_rewards.append(total_reward)

        # Update model using policy gradient
        await client.forward_backward(
            trajectories=batch_trajectories,
            rewards=batch_rewards,
            loss_type="importance_sampling"  # PPO-style update
        )

        await client.optim_step()

        episode += config["episodes_per_batch"]

        avg_reward = sum(batch_rewards) / len(batch_rewards)
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        # Save checkpoint every 100 episodes
        if episode % 100 == 0:
            await client.save_state(f"checkpoints/chess_rlvr_ep_{episode}")

    # Save final model
    await client.save_weights_for_sampler("models/chess_rlvr_final")
    reward_fn.close()
    print("RLVR training complete!")

async def play_chess_episode(client, reward_fn, max_moves=80):
    """
    Play one chess game using the model
    Returns trajectory and cumulative reward
    """
    board = chess.Board()
    trajectory = []
    total_reward = 0.0

    for move_num in range(max_moves):
        fen = board.fen()

        # Generate move from model
        prompt = f"You are playing chess. Current position (FEN): {fen}\n\nWhat is your next move? Respond with only the move in UCI format (e.g., 'e2e4')."

        response = await client.sample(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.7  # Some randomness for exploration
        )

        move_uci = response.strip()

        # Evaluate move and get reward
        reward = reward_fn.evaluate_move(fen, move_uci)

        # Store trajectory
        trajectory.append({
            "state": fen,
            "action": move_uci,
            "reward": reward
        })

        total_reward += reward

        # Apply move if legal
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                # Illegal move ends game with penalty
                break
        except:
            # Parse error ends game
            break

        # Check game termination
        if board.is_game_over():
            # Bonus/penalty for game outcome
            if board.is_checkmate():
                total_reward += 100.0  # Win!
            elif board.is_stalemate():
                total_reward += 0.0  # Draw
            break

        # Opponent move (random or Stockfish)
        if random.random() < 0.3:
            # 30% random (for diversity)
            opponent_move = random.choice(list(board.legal_moves))
        else:
            # 70% Stockfish (strong opponent)
            opponent_move_uci = reward_fn.get_best_move(board.fen(), depth=10)
            opponent_move = chess.Move.from_uci(opponent_move_uci)

        board.push(opponent_move)

        if board.is_game_over():
            if board.is_checkmate():
                total_reward -= 100.0  # Loss
            break

    return trajectory, total_reward

if __name__ == "__main__":
    import asyncio
    asyncio.run(rlvr_training())
```

---

## Phase 3: Integration with Existing Steel Browser System

### 3.1 Model Inference Wrapper

```python
# inference/chess_agent.py

import tinker
from tinker import AsyncClient
import chess

class TinkerChessAgent:
    def __init__(self, model_path="models/chess_rlvr_final"):
        """
        Chess agent using Tinker fine-tuned model
        """
        self.client = AsyncClient()
        self.model_path = model_path
        self.loaded = False

    async def initialize(self):
        """Load model weights"""
        await self.client.load_weights(self.model_path)
        self.loaded = True
        print(f"Loaded Tinker chess model from {self.model_path}")

    async def get_move(self, fen, temperature=0.3):
        """
        Get best move for given position

        Args:
            fen: Position in FEN notation
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            move in UCI format (e.g., 'e2e4')
        """
        if not self.loaded:
            await self.initialize()

        prompt = f"""You are playing chess. Current position (FEN): {fen}

What is your next move? Respond with only the move in UCI format (e.g., 'e2e4')."""

        response = await self.client.sample(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=temperature
        )

        # Extract UCI move from response
        move_uci = response.strip().split()[0]  # Take first token

        # Validate
        board = chess.Board(fen)
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                return move_uci
            else:
                # Fallback to random legal move
                return random.choice(list(board.legal_moves)).uci()
        except:
            # Parse error - fallback
            return random.choice(list(board.legal_moves)).uci()

    async def get_move_with_reasoning(self, fen):
        """
        Get move with explanation (for analysis)
        """
        prompt = f"""You are playing chess. Current position (FEN): {fen}

Analyze the position and suggest your next move. Explain your reasoning briefly, then provide the move in UCI format."""

        response = await self.client.sample(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.5
        )

        return response
```

### 3.2 Steel Browser Integration

```python
# integration/steel_adapter.py

from inference.chess_agent import TinkerChessAgent
import asyncio

class SteelChessIntegration:
    """
    Adapter to use Tinker model with existing Steel browser automation
    """
    def __init__(self):
        self.agent = TinkerChessAgent(model_path="models/chess_rlvr_final")
        self.initialized = False

    async def setup(self):
        await self.agent.initialize()
        self.initialized = True

    async def make_move_on_chesscom(self, steel_session, game_state):
        """
        Integration point for Steel browser automation

        Args:
            steel_session: Steel browser session
            game_state: Current game state from chess.com

        Returns:
            move executed
        """
        # Extract FEN from game state
        fen = game_state.get("fen") or self._parse_board_to_fen(game_state)

        # Get move from Tinker model
        move_uci = await self.agent.get_move(fen, temperature=0.2)

        # Convert UCI to chess.com format if needed
        move_formatted = self._format_move_for_chesscom(move_uci)

        # Execute move via Steel browser
        # (Keep existing Steel automation code)
        # steel_session.click_square(move_formatted['from'])
        # steel_session.click_square(move_formatted['to'])

        return move_uci

    def _parse_board_to_fen(self, game_state):
        """Convert game state to FEN notation"""
        # Implementation depends on game_state format
        pass

    def _format_move_for_chesscom(self, move_uci):
        """Convert UCI move to chess.com coordinates"""
        from_square = move_uci[:2]
        to_square = move_uci[2:4]

        return {
            "from": from_square,
            "to": to_square,
            "promotion": move_uci[4] if len(move_uci) > 4 else None
        }

# Usage example
async def main():
    integration = SteelChessIntegration()
    await integration.setup()

    # Use with existing Steel automation
    # move = await integration.make_move_on_chesscom(steel_session, game_state)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Evaluation & Comparison

### Benchmark Script

```python
# evaluation/benchmark.py

import chess
import chess.engine
from inference.chess_agent import TinkerChessAgent
import asyncio

async def benchmark_agent(agent, num_games=100):
    """
    Play against Stockfish and measure performance
    """
    stockfish = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    stockfish.configure({"Skill Level": 10})  # Mid-level

    results = {"wins": 0, "losses": 0, "draws": 0}

    for game_num in range(num_games):
        board = chess.Board()

        while not board.is_game_over():
            # Agent's turn
            fen = board.fen()
            move_uci = await agent.get_move(fen)

            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    results["losses"] += 1
                    break
            except:
                results["losses"] += 1
                break

            if board.is_game_over():
                break

            # Stockfish's turn
            stockfish_move = stockfish.play(board, chess.engine.Limit(time=0.1))
            board.push(stockfish_move.move)

        # Record result
        if board.is_checkmate():
            if board.turn:  # Black to move = White (agent) won
                results["wins"] += 1
            else:
                results["losses"] += 1
        else:
            results["draws"] += 1

        print(f"Game {game_num + 1}/{num_games}: {results}")

    stockfish.quit()

    win_rate = results["wins"] / num_games
    print(f"\nFinal Results: {results}")
    print(f"Win Rate: {win_rate:.1%}")

    return results

if __name__ == "__main__":
    agent = TinkerChessAgent("models/chess_rlvr_final")
    asyncio.run(benchmark_agent(agent, num_games=50))
```

---

## Expected Results

### Performance Metrics

| Metric | Before (GPT-4/Claude) | After (Tinker Fine-tuned) |
|--------|----------------------|---------------------------|
| **Cost per move** | $0.003-0.01 | $0.0001 (99% reduction) |
| **Latency** | 1-3 seconds | 0.1-0.5 seconds |
| **ELO Rating** | ~1400-1600 | Target: ~1800-2000 |
| **Legal Move Rate** | ~95% | Target: >99% |
| **Opening Knowledge** | General | Specialized |

### Timeline

- **Week 1**: Data collection and supervised fine-tuning (Phase 1)
- **Week 2**: RLVR implementation and training (Phase 2)
- **Week 3**: Integration with Steel browser and testing
- **Week 4**: Evaluation, benchmarking, and iteration

---

## Next Steps

1. **Set up Tinker account** and obtain API key
2. **Download chess dataset** (50k+ high-quality games from Lichess)
3. **Run supervised fine-tuning** (Phase 1)
4. **Install Stockfish** for reward evaluation
5. **Implement RLVR training** (Phase 2)
6. **Integrate with existing Steel browser code**
7. **Benchmark against baseline** (GPT-4/Claude)
8. **Iterate and improve**

---

## Resources

- **Tinker Docs**: https://tinker-docs.thinkingmachines.ai/
- **Lichess Database**: https://database.lichess.org/
- **Python Chess Library**: https://python-chess.readthedocs.io/
- **Stockfish**: https://stockfishchess.org/
- **Chess Programming Wiki**: https://www.chessprogramming.org/

---

## Troubleshooting

**Issue**: Model generates illegal moves
- **Solution**: Increase supervised training data, add move legality filtering, use constrained decoding

**Issue**: Model plays too defensively
- **Solution**: Adjust reward function to reward aggressive play, train against weaker opponents initially

**Issue**: High API costs during training
- **Solution**: Use smaller model (Qwen3-4B) for experimentation, batch requests efficiently

**Issue**: Slow training
- **Solution**: Reduce LoRA rank, use smaller batch sizes, reduce training data size initially

---

*Implementation Guide v1.0*
*Last Updated: 2025-11-13*
