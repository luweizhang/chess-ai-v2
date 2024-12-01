import chess
import time
from typing import Optional, Tuple

class ChessAI:
    def __init__(self, depth: int = 3):
        self.depth = depth
        # Standard piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Piece-square tables for positional evaluation
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        self.rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        self.queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        self.king_middle_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        
        self.king_endgame_table = [
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,  0,  0,-10,-20,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-30,  0,  0,  0,  0,-30,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
        ]
        
        # Transposition table for caching positions
        self.transposition_table = {}
        
    def is_endgame(self, board: chess.Board) -> bool:
        """Determine if the position is in the endgame"""
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        minors = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) +
                 len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)))
        return queens == 0 or (queens == 2 and minors <= 2)

    def evaluate_board(self, board: chess.Board) -> float:
        """Enhanced evaluation function"""
        if board.is_checkmate():
            return -float('inf') if board.turn else float('inf')
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        is_endgame = self.is_endgame(board)
        
        # Material and position evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = self.piece_values[piece.piece_type]
                
                # Position bonus based on piece-square tables
                if piece.piece_type == chess.PAWN:
                    value += self.pawn_table[square if piece.color else chess.square_mirror(square)] * 0.1
                elif piece.piece_type == chess.KNIGHT:
                    value += self.knight_table[square if piece.color else chess.square_mirror(square)] * 0.1
                elif piece.piece_type == chess.BISHOP:
                    value += self.bishop_table[square if piece.color else chess.square_mirror(square)] * 0.1
                elif piece.piece_type == chess.ROOK:
                    value += self.rook_table[square if piece.color else chess.square_mirror(square)] * 0.1
                elif piece.piece_type == chess.QUEEN:
                    value += self.queen_table[square if piece.color else chess.square_mirror(square)] * 0.1
                elif piece.piece_type == chess.KING:
                    if is_endgame:
                        value += self.king_endgame_table[square if piece.color else chess.square_mirror(square)] * 0.1
                    else:
                        value += self.king_middle_table[square if piece.color else chess.square_mirror(square)] * 0.1
                
                if piece.color:
                    score += value
                else:
                    score -= value
        
        # Mobility (number of legal moves)
        if board.turn:
            score += len(list(board.legal_moves)) * 0.1
        else:
            board.turn = not board.turn
            score -= len(list(board.legal_moves)) * 0.1
            board.turn = not board.turn
        
        # Pawn structure evaluation
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Doubled pawns penalty
        for file in range(8):
            white_pawns_in_file = sum(1 for sq in white_pawns if chess.square_file(sq) == file)
            black_pawns_in_file = sum(1 for sq in black_pawns if chess.square_file(sq) == file)
            if white_pawns_in_file > 1:
                score -= 20 * (white_pawns_in_file - 1)
            if black_pawns_in_file > 1:
                score += 20 * (black_pawns_in_file - 1)
        
        # Isolated pawns penalty
        for file in range(8):
            white_isolated = True
            black_isolated = True
            for adjacent_file in [file-1, file+1]:
                if 0 <= adjacent_file < 8:
                    if any(chess.square_file(sq) == adjacent_file for sq in white_pawns):
                        white_isolated = False
                    if any(chess.square_file(sq) == adjacent_file for sq in black_pawns):
                        black_isolated = False
            if white_isolated and any(chess.square_file(sq) == file for sq in white_pawns):
                score -= 10
            if black_isolated and any(chess.square_file(sq) == file for sq in black_pawns):
                score += 10
        
        return score

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[float, Optional[chess.Move]]:
        """Enhanced Minimax with alpha-beta pruning and transposition table"""
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
