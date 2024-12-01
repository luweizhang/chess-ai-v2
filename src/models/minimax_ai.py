import chess
import random
from typing import Optional, Tuple, Dict, List
from .base_ai import ChessAI
import time

class MinimaxAI(ChessAI):
    def __init__(self, depth: int = 3):
        self.depth = depth
        self._pv_line: List[chess.Move] = []
        self.max_time = 0.5
        self.start_time = 0
        
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
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
                -40,-20,  0,  5,  5,  0,-20,-40,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50,
            ]
        }
        
        self.transposition_table = {}
        self.table_size = 100000
        
    @property
    def pv_line(self) -> List[chess.Move]:
        return self._pv_line
        
    def evaluate_position(self, board: chess.Board) -> float:
        if board.is_game_over():
            if board.is_checkmate():
                return -20000 if board.turn else 20000
            return 0
            
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            value = self.piece_values[piece.piece_type]
            
            if piece.piece_type in [chess.PAWN, chess.KNIGHT]:
                pos_value = self.pst[piece.piece_type][square if piece.color else 63 - square]
                value += pos_value
                
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
                
        if not board.is_check():
            score += len(list(board.legal_moves)) * 5 if board.turn else -len(list(board.legal_moves)) * 5
            
        return score if board.turn == chess.WHITE else -score
        
    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[chess.Move]]:
        if time.time() - self.start_time > self.max_time:
            return self.evaluate_position(board), None
            
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None
            
        board_hash = board.fen()
        if board_hash in self.transposition_table:
            score, stored_depth, best_move = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return score, best_move
                
        best_move = None
        
        moves = self.order_moves(board)
        
        if maximizing:
            max_eval = float('-inf')
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
                    
            if len(self.transposition_table) >= self.table_size:
                self.transposition_table.clear()
            self.transposition_table[board_hash] = (max_eval, depth, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
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
                    
            if len(self.transposition_table) >= self.table_size:
                self.transposition_table.clear()
            self.transposition_table[board_hash] = (min_eval, depth, best_move)
            return min_eval, best_move
            
    def order_moves(self, board: chess.Board) -> List[chess.Move]:
        """Fast move ordering"""
        moves = list(board.legal_moves)
        scored_moves = []
        
        for move in moves:
            score = 0
            # Captures
            if board.is_capture(move):
                victim_piece = board.piece_at(move.to_square)
                if victim_piece:
                    score = 10 * self.piece_values[victim_piece.piece_type]
            
            # Promotions
            if move.promotion:
                score += self.piece_values[move.promotion]
                
            # Center control (simplified)
            if move.to_square in [27, 28, 35, 36]:
                score += 30
                
            # Use move string as secondary sort key to ensure stable sort
            scored_moves.append((score, str(move), move))
            
        # Sort by score first, then by move string for stable sorting
        scored_moves.sort(key=lambda x: (-x[0], x[1]))
        return [move for _, _, move in scored_moves]
        
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None
            
        self.start_time = time.time()
        print(f"\nSearching position: {board.fen()}")
        
        self._pv_line = []
        
        best_move = None
        for current_depth in range(1, self.depth + 1):
            if time.time() - self.start_time > self.max_time:
                break
                
            score, move = self.minimax(board, current_depth, float('-inf'), float('inf'), True)
            if move:
                best_move = move
                self._pv_line = self.get_pv_line(board, current_depth)
                print(f"Depth {current_depth}: Best move {move}, Score: {score}, Time: {time.time() - self.start_time:.2f}s")
                
        if best_move:
            print(f"Final best move: {best_move}")
            print(f"Total time: {time.time() - self.start_time:.2f}s")
            
        return best_move
        
    def get_pv_line(self, board: chess.Board, depth: int) -> List[chess.Move]:
        if depth == 0 or board.is_game_over():
            return []
            
        board_hash = board.fen()
        if board_hash in self.transposition_table:
            _, stored_depth, best_move = self.transposition_table[board_hash]
            if stored_depth >= depth and best_move:
                board.push(best_move)
                pv_line = [best_move] + self.get_pv_line(board, depth - 1)
                board.pop()
                return pv_line
                
        return []
