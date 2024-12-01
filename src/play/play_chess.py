import chess
from ..models.chess_ai import ChessAI
from ..models.neural_ai import NeuralAI
from ..models.mcts_ai import MCTSAI

def print_board(board):
    print("\n" + str(board) + "\n")

def main():
    # Select AI type
    print("Choose your opponent:")
    print("1. Classic Minimax AI")
    print("2. Neural Network AI")
    print("3. Monte Carlo Tree Search AI")
    
    while True:
        choice = input("Enter your choice (1-3): ")
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Initialize the selected AI
    if choice == '1':
        ai = ChessAI(depth=3)
        print("Playing against Classic Minimax AI")
    elif choice == '2':
        ai = NeuralAI(depth=3)
        print("Playing against Neural Network AI")
    else:
        ai = MCTSAI(simulation_limit=1000)
        print("Playing against Monte Carlo Tree Search AI")
    
    board = chess.Board()
    
    print("\nWelcome to Chess AI!")
    print("Enter moves in UCI format (e.g., 'e2e4')")
    print("Enter 'quit' to exit")
    
    while not board.is_game_over():
        print_board(board)
        
        if board.turn:  # Human's turn (White)
            move_str = input("Your move: ")
            if move_str.lower() == 'quit':
                break
                
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move! Try again.")
                    continue
            except ValueError:
                print("Invalid move format! Try again.")
                continue
        else:  # AI's turn (Black)
            print("AI is thinking...")
            move = ai.get_best_move(board)
            if move:
                print(f"AI moves: {move.uci()}")
                board.push(move)
            else:
                print("AI couldn't find a move!")
                break
    
    print_board(board)
    print("\nGame Over!")
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw due to insufficient material!")

if __name__ == "__main__":
    main()
