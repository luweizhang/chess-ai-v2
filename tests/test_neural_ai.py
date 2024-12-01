"""Tests for neural network chess AI."""

import unittest
import chess
import torch
from src.models.neural_ai import ChessConvNet, NeuralAI

class TestChessConvNet(unittest.TestCase):
    def setUp(self):
        self.model = ChessConvNet()
        
    def test_model_output_shape(self):
        # Create a random board tensor (batch_size=1, channels=12, height=8, width=8)
        board_tensor = torch.randn(1, 12, 8, 8)
        
        # Get model outputs
        policy, value = self.model(board_tensor)
        
        # Check policy output shape (1968 possible moves)
        self.assertEqual(policy.shape, (1, 1968))
        
        # Check value output shape (single scalar)
        self.assertEqual(value.shape, (1, 1))
        
    def test_value_output_range(self):
        # Value head should output between -1 and 1 (tanh activation)
        board_tensor = torch.randn(1, 12, 8, 8)
        _, value = self.model(board_tensor)
        
        self.assertTrue(torch.all(value >= -1))
        self.assertTrue(torch.all(value <= 1))

class TestNeuralAI(unittest.TestCase):
    def setUp(self):
        self.ai = NeuralAI(depth=1)  # Use depth=1 for faster tests
        
    def test_board_conversion(self):
        board = chess.Board()
        tensor = self.ai.board_to_tensor(board)
        
        # Check tensor shape
        self.assertEqual(tensor.shape, (1, 12, 8, 8))
        
        # Initial position should have 32 pieces (32 ones in the tensor)
        self.assertEqual(torch.sum(tensor).item(), 32)
        
    def test_evaluation(self):
        board = chess.Board()
        
        # Starting position should be roughly equal
        eval_score = self.ai.evaluate_board(board)
        self.assertTrue(-0.5 <= eval_score <= 0.5)
        
        # Checkmate should be infinite
        board = chess.Board("8/8/8/8/8/5K2/7Q/7k w - - 0 1")  # White to move and mate in one
        board.push_san("Qh1#")
        eval_score = self.ai.evaluate_board(board)
        self.assertEqual(eval_score, float('inf'))

if __name__ == '__main__':
    unittest.main()
