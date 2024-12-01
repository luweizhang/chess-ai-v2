import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Optional, Tuple, Dict
import math

class ChessConvNet(nn.Module):
    def __init__(self):
        super(ChessConvNet, self).__init__()
        # Input: 12 channels (6 piece types * 2 colors)
        # Residual blocks
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        # Residual blocks
        self.num_residual = 4  # Reduced from 8 for faster training
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(self.num_residual)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_fc = nn.Linear(128 * 8 * 8, 20480)  # 4096 moves * 5 promotion options
        
        # Value head
        self.value_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial convolution block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 128 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class NeuralAI:
    def __init__(self, depth: int = 3, model_path: str = '../../models/chess_model.pth'):
        self.depth = depth
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChessConvNet().to(self.device)
        
        # Piece values for basic evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Piece-square tables for positional evaluation
        self.pst = {
            chess.PAWN: [
                0,  0,  0,  0,  0,  0,  0,  0,
                50, 50, 50, 50, 50, 50, 50, 50,
                10, 10, 20, 30, 30, 20, 10, 10,
                5,  5, 10, 25, 25, 10,  5,  5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5, -5,-10,  0,  0,-10, -5,  5,
                5, 10, 10,-20,-20, 10, 10,  5,
                0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ],
            chess.BISHOP: [
                -20,-10,-10,-10,-10,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5, 10, 10,  5,  0,-10,
                -10,  5,  5, 10, 10,  5,  5,-10,
                -10,  0, 10, 10, 10, 10,  0,-10,
                -10, 10, 10, 10, 10, 10, 10,-10,
                -10,  5,  0,  0,  0,  0,  5,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ],
            chess.ROOK: [
                0,  0,  0,  0,  0,  0,  0,  0,
                5, 10, 10, 10, 10, 10, 10,  5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                0,  0,  0,  5,  5,  0,  0,  0
            ],
            chess.QUEEN: [
                -20,-10,-10, -5, -5,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5,  5,  5,  5,  0,-10,
                -5,  0,  5,  5,  5,  5,  0, -5,
                0,  0,  5,  5,  5,  5,  0, -5,
                -10,  5,  5,  5,  5,  5,  0,-10,
                -10,  0,  5,  0,  0,  0,  0,-10,
                -20,-10,-10, -5, -5,-10,-10,-20
            ],
            chess.KING: [
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -20,-30,-30,-40,-40,-30,-30,-20,
                -10,-20,-20,-20,-20,-20,-20,-10,
                20, 20,  0,  0,  0,  0, 20, 20,
                20, 30, 10,  0,  0, 10, 30, 20
            ]
        }
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model.eval()
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using random weights")
        else:
            print(f"No pre-trained model found at {model_path}")
            print("Using random weights")
        
        # Transposition table for caching
        self.transposition_table = {}
        
    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert board to 12-channel tensor representation"""
        # Initialize 12 channels (6 piece types * 2 colors)
        tensor = torch.zeros(12, 8, 8, dtype=torch.float32, device=self.device)
        
        # Map pieces to channels
        piece_to_channel = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        # Fill tensor with piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = square // 8
                file = square % 8
                channel = piece_to_channel[piece.piece_type]
                if not piece.color:  # Black pieces
                    channel += 6
                tensor[channel][rank][file] = 1.0
        
        return tensor

    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate board position using neural network"""
        if board.is_checkmate():
            return -float('inf') if board.turn else float('inf')
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
            
        with torch.no_grad():
            tensor = self.board_to_tensor(board)
            _, value = self.model(tensor.unsqueeze(0))
            return float(value.item()) * (1 if board.turn else -1)

    def evaluate_position(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """Evaluate a position using either neural network or basic evaluation"""
        # If we have a trained model, use it
        if hasattr(self, 'model') and self.model is not None:
            try:
                with torch.no_grad():
                    x = self.board_to_tensor(board).unsqueeze(0)
                    policy, value = self.model(x)
                    # Convert policy to move probabilities
                    policy_dict = self._policy_to_moves(policy[0], board)
                    return policy_dict, value.item()
            except Exception as e:
                print(f"Error using neural network: {e}")
                # Fall back to basic evaluation
                pass
        
        # Use basic evaluation
        return self._basic_evaluation(board)
    
    def _basic_evaluation(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """Basic chess evaluation function"""
        if board.is_game_over():
            outcome = board.outcome()
            if outcome is None:
                return {}, 0.0
            if outcome.winner is None:
                return {}, 0.0
            return {}, 1.0 if outcome.winner == board.turn else -1.0

        # Material score
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            value = self.piece_values[piece.piece_type]
            # Add positional bonus
            pos_value = self.pst[piece.piece_type][square if piece.color else 63 - square]
            
            if piece.color == board.turn:
                score += value + pos_value
            else:
                score -= value + pos_value
        
        # Normalize score to [-1, 1] range
        normalized_score = 2 / (1 + math.exp(-score / 1000)) - 1
        
        # Generate move probabilities based on basic principles
        policy_dict = {}
        for move in board.legal_moves:
            # Capture moves
            if board.is_capture(move):
                policy_dict[move] = 0.8
            # Center control
            elif move.to_square in [27, 28, 35, 36]:
                policy_dict[move] = 0.7
            # Development moves
            elif move.from_square in [1, 6, 57, 62] and not board.is_capture(move):
                policy_dict[move] = 0.6
            else:
                policy_dict[move] = 0.4
                
        # Normalize policy
        total = sum(policy_dict.values())
        if total > 0:
            policy_dict = {move: prob/total for move, prob in policy_dict.items()}
            
        return policy_dict, normalized_score

    def move_to_index(self, move: chess.Move) -> int:
        """Convert a chess move to a policy index"""
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion if move.promotion else 0
        # Base index from regular moves (64 * 64 = 4096 possible from-to combinations)
        # Add promotion piece type offset (5 possible values: None, N, B, R, Q)
        return from_square * 64 + to_square + (promotion * 4096 if promotion else 0)
        
    def index_to_move(self, index: int) -> chess.Move:
        """Convert a policy index back to a chess move"""
        promotion = index // 4096
        base_index = index % 4096
        from_square = base_index // 64
        to_square = base_index % 64
        return chess.Move(from_square, to_square, promotion if promotion > 0 else None)

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool) -> tuple[float, Optional[chess.Move]]:
        """Minimax search with alpha-beta pruning and transposition table"""
        # Check transposition table
        board_hash = hash(str(board))
        if board_hash in self.transposition_table:
            cached_depth, cached_value, cached_move = self.transposition_table[board_hash]
            if cached_depth >= depth:
                return cached_value, cached_move
        
        if depth == 0 or board.is_game_over():
            value = self.evaluate_board(board)
            self.transposition_table[board_hash] = (depth, value, None)
            return value, None
            
        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            moves = list(board.legal_moves)
            # Move ordering: captures first
            moves.sort(key=lambda move: board.is_capture(move), reverse=True)
            
            for move in moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
                    
            self.transposition_table[board_hash] = (depth, max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            moves = list(board.legal_moves)
            # Move ordering: captures first
            moves.sort(key=lambda move: board.is_capture(move), reverse=True)
            
            for move in moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
                    
            self.transposition_table[board_hash] = (depth, min_eval, best_move)
            return min_eval, best_move

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get best move using iterative deepening"""
        best_move = None
        
        # Iterative deepening
        for current_depth in range(1, self.depth + 1):
            try:
                _, move = self.minimax(board, current_depth, float('-inf'), float('inf'), board.turn)
                if move:
                    best_move = move
            except Exception:
                break
                
        return best_move
