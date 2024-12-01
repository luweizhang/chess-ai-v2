from abc import ABC, abstractmethod
import chess
from typing import Optional, List

class ChessAI(ABC):
    """Base class for chess AI implementations"""
    
    @abstractmethod
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get the best move for the current position"""
        pass
        
    @property
    def pv_line(self) -> List[chess.Move]:
        """Get the principal variation (best line of play)"""
        return []
        
    @property
    def name(self) -> str:
        """Get the name of this AI implementation"""
        return self.__class__.__name__
