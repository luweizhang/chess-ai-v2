import chess
import math
import random
import numpy as np
from typing import Optional, Dict, Tuple, List, Set
from collections import defaultdict, deque
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from src.models.neural_ai import NeuralAI
import threading
import time

class TranspositionTable:
    def __init__(self, size: int = 1000000):
        self.size = size
        self.table: Dict[str, Tuple[float, int]] = {}
        self.lock = threading.Lock()
    
    def store(self, board: chess.Board, value: float, depth: int):
        """Store position value with replacement strategy"""
        with self.lock:
            key = board.fen()
            if len(self.table) >= self.size:
                # Remove random entry if full
                del self.table[random.choice(list(self.table.keys()))]
            self.table[key] = (value, depth)
    
    def lookup(self, board: chess.Board) -> Optional[Tuple[float, int]]:
        """Lookup position value"""
        return self.table.get(board.fen())

class Node:
    def __init__(self, board: chess.Board, parent: Optional['Node'] = None, last_move: Optional[chess.Move] = None, evaluator: Optional[NeuralAI] = None):
        self.board = board.copy()
        self.parent = parent
        self.last_move = last_move
        self.children: List['Node'] = []
        self.visits = 0
        self.value = 0.0
        self.prior = 1.0 if parent is None else 0.0
        self.evaluator = evaluator
        
    def expand(self):
        """Expand the node by creating all possible child nodes"""
        if not self.children:  # Only expand if not already expanded
            for move in self.board.legal_moves:
                new_board = self.board.copy()
                new_board.push(move)
                child = Node(new_board, parent=self, last_move=move, evaluator=self.evaluator)
                self.children.append(child)
                
    def simulate(self) -> float:
        """Simulate a game from this position and return the result"""
        if self.board.is_game_over():
            outcome = self.board.outcome()
            if outcome is None:
                return 0.0
            if outcome.winner is None:
                return 0.0
            return 1.0 if outcome.winner == self.board.turn else -1.0
            
        # Use neural network for evaluation
        if self.evaluator is None:
            print("Warning: No evaluator provided to Node")
            return 0.0
            
        policy_dict, value = self.evaluator.evaluate_position(self.board)
        return float(value)
        
    def backpropagate(self, value: float):
        """Update node statistics"""
        node = self
        while node is not None:
            node.visits += 1
            node.value += value
            value = -value  # Flip value for opponent
            node = node.parent
            
    def ucb_score(self, exploration: float = 1.41) -> float:
        """Calculate the UCB score for this node"""
        if self.visits == 0:
            return float('inf')
        
        # Q-value
        q_value = self.value / self.visits if self.visits > 0 else 0.0
        
        # U-value (exploration term)
        parent_visits = math.log(self.parent.visits) if self.parent is not None else 0.0
        u_value = exploration * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        
        return q_value + u_value

class MCTSAI:
    def __init__(self, simulation_limit: int = 1000):
        self.simulation_limit = simulation_limit
        self.neural_evaluator = NeuralAI()
        self.pv_line: List[chess.Move] = []
        self.transposition_table = TranspositionTable()
        self.exploration_constant = 1.4  # UCB exploration parameter
        self.time_limit = 5.0  # Time limit in seconds
        self.name = "MCTS"  # Add name attribute
        
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get the best move for the current position using MCTS"""
        if board.is_game_over():
            return None
            
        # Create root node
        root = Node(board, evaluator=self.neural_evaluator)
        
        # Run MCTS for specified number of simulations or until time limit
        start_time = time.time()
        num_simulations = 0
        
        while num_simulations < self.simulation_limit and (time.time() - start_time) < self.time_limit:
            node = root
            
            # Selection
            while node.children and not node.board.is_game_over():
                # Select child with highest UCB score
                node = max(node.children, key=lambda n: n.ucb_score(self.exploration_constant))
            
            # Expansion
            if not node.board.is_game_over():
                node.expand()
                if node.children:
                    node = random.choice(node.children)
            
            # Simulation/Evaluation
            value = node.simulate()
            
            # Backpropagation
            node.backpropagate(value)
            
            num_simulations += 1
        
        # Select best move based on visit counts
        if not root.children:
            return None
            
        # Choose move with highest visit count
        best_child = max(root.children, key=lambda n: n.visits)
        
        # Store principal variation
        self.pv_line = []
        current = best_child
        while current.children:
            self.pv_line.append(current.last_move)
            current = max(current.children, key=lambda n: n.visits)
        
        # Print some debug info
        print(f"Completed {num_simulations} simulations in {time.time() - start_time:.2f} seconds")
        print(f"Best move: {best_child.last_move}, Value: {best_child.value/best_child.visits:.3f}, Visits: {best_child.visits}")
        
        return best_child.last_move

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None, prior: float = 0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.wins = 0
        self.visits = 0
        self.virtual_loss = 0  # For parallel MCTS
        self.untried_moves = list(board.legal_moves)
        self.prior = prior
        self.value = 0.0
        self.best_value = float('-inf')
        self.pv_line: List[chess.Move] = []  # Principal variation
        self.history_score = 0.0  # For move ordering
        self.lock = threading.Lock()
