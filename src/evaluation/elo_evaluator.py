import chess
import math
import random
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass
from ..models.chess_ai import ChessAI
from ..models.neural_ai import NeuralAI
from ..models.mcts_ai import MCTSAI
from ..utils.logger import setup_logger
import sys
from tqdm import tqdm

logger = setup_logger("chess_ai.evaluation")

@dataclass
class GlickoPlayer:
    rating: float
    rd: float  # Rating deviation
    
    def __init__(self, rating: float = 1500, rd: float = 350):
        self.rating = rating
        self.rd = rd
    
    def update_rd(self, games_played: int):
        """Update rating deviation based on number of games played"""
        self.rd = max(30, min(350, self.rd * (0.9 ** games_played)))

class EloEvaluator:
    def __init__(self, move_time_limit: float = 0.2):  # Reduced to 0.2s
        self.move_time_limit = move_time_limit
        self.players: Dict[str, GlickoPlayer] = {}
        
        # AI variants with different strengths
        self.ai_players = {
            "Minimax-1": lambda: ChessAI(depth=1),
            "Minimax-2": lambda: ChessAI(depth=2),
            "Neural-1": lambda: NeuralAI(depth=1),
            "MCTS-1000": lambda: MCTSAI(simulation_limit=1000, num_threads=4),
            "MCTS-2000": lambda: MCTSAI(simulation_limit=2000, num_threads=4)
        }
        
        # Initialize ratings with different starting points
        self.players["Minimax-1"] = GlickoPlayer(1200)  # Weakest
        self.players["Minimax-2"] = GlickoPlayer(1400)
        self.players["Neural-1"] = GlickoPlayer(1500)
        self.players["MCTS-1000"] = GlickoPlayer(1600)
        self.players["MCTS-2000"] = GlickoPlayer(1700)  # Potentially strongest

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using ELO formula"""
        return 1 / (1 + 10**((rating_b - rating_a) / 400))

    def update_rating(self, rating: float, expected: float, actual: float, k: float = 32) -> float:
        """Update rating using simplified ELO formula"""
        return rating + k * (actual - expected)

    def play_game(self, white_ai, black_ai) -> float:
        """Play a game between two AIs with timeout"""
        board = chess.Board()
        moves = 0
        max_moves = 40  # Reduced max moves for faster games
        
        while not board.is_game_over() and moves < max_moves:
            try:
                start_time = time.time()
                if board.turn:  # White's turn
                    move = white_ai.get_best_move(board)
                else:  # Black's turn
                    move = black_ai.get_best_move(board)
                
                # Check timeout with more lenient limit for neural network
                elapsed = time.time() - start_time
                if isinstance(white_ai if board.turn else black_ai, NeuralAI):
                    timeout = self.move_time_limit * 1.5  # Slightly more time for neural
                else:
                    timeout = self.move_time_limit
                    
                if elapsed > timeout:
                    return 0.0 if board.turn else 1.0  # Timeout loses
                
                if move is None:
                    return 0.5  # Draw if no move found
                    
                board.push(move)
                moves += 1
                
            except Exception as e:
                logger.error(f"Error in game: {e}")
                return 0.5  # Draw on error
        
        # Game result
        if board.is_checkmate():
            return 1.0 if not board.turn else 0.0
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.5
        elif moves >= max_moves:
            # Evaluate position if max moves reached
            if isinstance(white_ai, NeuralAI):
                eval = white_ai.evaluate_board(board)
                return 1.0 if eval > 0.5 else 0.0 if eval < -0.5 else 0.5
            else:
                return 0.5  # Draw if no neural evaluation available
        return 0.5

    def evaluate_elo(self, num_games: int = 12):  # Reduced number of games
        """Thorough ELO evaluation"""
        logger.info("Starting ELO evaluation...")
        logger.info(f"Settings: {num_games} games, {self.move_time_limit}s per move")
        
        # Track games played for rating deviation
        games_played = {name: 0 for name in self.ai_players.keys()}
        
        # Progress bar
        pbar = tqdm(range(num_games), desc="Playing games")
        
        for i in pbar:
            # Select players, ensuring each plays similar number of games
            ai_names = list(self.ai_players.keys())
            min_games = min(games_played.values())
            
            # Find candidates that haven't played too many games
            candidates = [n for n, g in games_played.items() if g <= min_games + 2]
            if len(candidates) < 2:
                # If not enough candidates, use all AIs
                candidates = ai_names
            
            white_name = random.choice(candidates)
            black_candidates = [n for n in candidates if n != white_name]
            if not black_candidates:
                black_candidates = [n for n in ai_names if n != white_name]
            black_name = random.choice(black_candidates)
            
            pbar.set_description(f"Game {i+1}: {white_name} vs {black_name}")
            
            # Create AIs
            white_ai = self.ai_players[white_name]()
            black_ai = self.ai_players[black_name]()
            
            # Play game
            result = self.play_game(white_ai, black_ai)
            
            # Update ratings
            expected_white = self.expected_score(
                self.players[white_name].rating,
                self.players[black_name].rating
            )
            
            self.players[white_name].rating = self.update_rating(
                self.players[white_name].rating,
                expected_white,
                result
            )
            
            self.players[black_name].rating = self.update_rating(
                self.players[black_name].rating,
                1 - expected_white,
                1 - result
            )
            
            # Update games played and rating deviations
            games_played[white_name] += 1
            games_played[black_name] += 1
            self.players[white_name].update_rd(games_played[white_name])
            self.players[black_name].update_rd(games_played[black_name])
            
            # Log result
            result_str = 'White wins' if result == 1 else 'Black wins' if result == 0 else 'Draw'
            logger.info(f"Game {i+1} Result: {white_name} vs {black_name}: {result_str}")
        
        # Print final ratings
        logger.info("\nFinal ELO Ratings:")
        logger.info("-" * 40)
        sorted_players = sorted(self.players.items(), key=lambda x: x[1].rating, reverse=True)
        for name, player in sorted_players:
            logger.info(f"{name:12} ELO: {player.rating:.1f} ± {2*player.rd:.1f} ({games_played[name]} games)")

def main():
    evaluator = EloEvaluator(move_time_limit=0.2)
    evaluator.evaluate_elo(num_games=12)

if __name__ == "__main__":
    main()
