import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ..models.neural_ai import ChessConvNet, NeuralAI
from ..models.chess_ai import ChessAI
import random
from typing import List, Tuple
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from ..utils.logger import setup_logger

logger = setup_logger("chess_ai.training")

class ChessDataset(Dataset):
    def __init__(self, positions: List[Tuple[chess.Board, float]]):
        self.positions = positions
        self.neural_ai = NeuralAI()  # For move encoding

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        board, score = self.positions[idx]
        tensor = self._board_to_tensor(board)
        
        # Get best move and encode it
        best_move = None
        legal_moves = list(board.legal_moves)
        if legal_moves:
            # For training data, just use first legal move as target
            best_move = legal_moves[0]
            move_idx = self.neural_ai.move_to_index(best_move)
        else:
            move_idx = 0  # Default to 0 if no legal moves
            
        return tensor, torch.tensor([score, move_idx], dtype=torch.float32)

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        pieces = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        
        tensor = torch.zeros(12, 8, 8)
        
        for color in [chess.WHITE, chess.BLACK]:
            for piece_idx, piece_type in enumerate(pieces):
                channel_idx = piece_idx if color else piece_idx + 6
                for square in board.pieces(piece_type, color):
                    rank, file = chess.square_rank(square), chess.square_file(square)
                    tensor[channel_idx][7-rank][file] = 1
                    
        return tensor

def generate_game(ai1, ai2, max_moves=100) -> List[Tuple[chess.Board, float]]:
    """Generate a game between two AIs and return positions with scores"""
    positions = []
    board = chess.Board()
    moves = 0
    
    try:
        while not board.is_game_over() and moves < max_moves:
            # Store position before move
            positions.append((board.copy(), ai1.evaluate_board(board)))
            
            # Get and make move
            if board.turn:
                move = ai1.get_best_move(board)
            else:
                move = ai2.get_best_move(board)
                
            if move is None:
                break
                
            board.push(move)
            moves += 1
            
        # Add final position
        if board.is_checkmate():
            final_score = float('inf') if not board.turn else -float('inf')
        else:
            final_score = 0.0
        positions.append((board.copy(), final_score))
        
    except Exception as e:
        logger.error(f"Error in game generation: {e}")
        
    return positions

def generate_positions(num_positions: int, max_moves: int) -> List[Tuple[chess.Board, float]]:
    """Generate positions from a single game"""
    logger.info("Initializing AI models...")
    minimax_ai = ChessAI(depth=3)
    neural_ai = NeuralAI(depth=2)
    
    positions = []
    games_played = 0
    logger.info(f"Generating {num_positions} positions...")
    
    with tqdm(total=num_positions, desc="Generating positions") as pbar:
        while len(positions) < num_positions:
            # Alternate between AI vs AI and AI vs itself
            if len(positions) % 2 == 0:
                game_positions = generate_game(minimax_ai, neural_ai, max_moves)
            else:
                game_positions = generate_game(minimax_ai, minimax_ai, max_moves)
            
            new_positions = len(game_positions[:num_positions - len(positions)])
            positions.extend(game_positions[:num_positions - len(positions)])
            games_played += 1
            pbar.update(new_positions)
            
    logger.info(f"Generated {len(positions)} positions from {games_played} games")
    return positions[:num_positions]

def train_network(model: nn.Module, train_loader: DataLoader, num_epochs: int = 10):
    """Train the neural network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    value_criterion = nn.MSELoss()
    policy_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    
    logger.info(f"\nTraining on {device}")
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_value_loss = 0
        total_policy_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            value_targets = targets[:, 0].to(device)  # Value target
            policy_targets = targets[:, 1].long().to(device)  # Policy target (move index)
            
            optimizer.zero_grad()
            policy, value = model(inputs)
            
            # Compute both value and policy losses
            value_loss = value_criterion(value, value_targets.unsqueeze(1))
            policy_loss = policy_criterion(policy, policy_targets)
            
            # Combined loss (weighted sum)
            loss = value_loss + policy_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1
            
            current_loss = total_loss/num_batches
            current_value_loss = total_value_loss/num_batches
            current_policy_loss = total_policy_loss/num_batches
            
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'value_loss': f"{current_value_loss:.4f}",
                'policy_loss': f"{current_policy_loss:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Average Value Loss: {total_value_loss/num_batches:.4f}")
        logger.info(f"Average Policy Loss: {total_policy_loss/num_batches:.4f}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            logger.info(f"New best loss achieved! Saving model...")
            torch.save(model.state_dict(), '../../models/chess_model_best.pth')
        else:
            patience_counter += 1
            logger.info(f"Loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

def main():
    # Parameters
    num_positions = 10000  # Target number of positions
    max_moves = 100
    batch_size = 64
    num_epochs = 20
    
    logger.info("Generating training data...")
    all_positions = generate_positions(num_positions, max_moves)
    logger.info(f"Generated {len(all_positions)} positions")
    
    # Create dataset and dataloader
    dataset = ChessDataset(all_positions)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Create and train model
    model = ChessConvNet()
    train_network(model, train_loader, num_epochs)
    
    # Load best model before saving
    model.load_state_dict(torch.load('../../models/chess_model_best.pth'))
    torch.save(model.state_dict(), '../../models/chess_model.pth')
    logger.info("Best model saved to models/chess_model.pth")

if __name__ == "__main__":
    main()
