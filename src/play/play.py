import chess
import sys
from ..models.chess_ai import ChessAI

def print_board(board):
    print("\n" + str(board) + "\n")

def main():
    board = chess.Board()
    ai = ChessAI(depth=3)  # Increase depth for stronger play, but slower computation
    
    print("Welcome to Chess AI!")
    print("Enter moves in UCI format (e.g., 'e2e4')")
    print("Enter 'quit' to exit")
    
    # Process the first move
    first_move = sys.argv[1] if len(sys.argv) > 1 else input("Your move: ")
    try:
        move = chess.Move.from_uci(first_move)
        if move in board.legal_moves:
            print_board(board)
            board.push(move)
        else:
            print("Illegal move!")
            return
    except ValueError:
        print("Invalid move format!")
        return
    
    while not board.is_game_over():
        if not board.turn:  # AI's turn (Black)
            print("AI is thinking...")
            move = ai.get_best_move(board)
            if move:
                print(f"AI moves: {move.uci()}")
                board.push(move)
                print_board(board)
            else:
                print("AI couldn't find a move!")
                break
        else:  # Human's turn (White)
            try:
                move_str = input("Your move: ")
                if move_str.lower() == 'quit':
                    break
                    
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move! Try again.")
                    continue
            except ValueError:
                print("Invalid move format! Try again.")
                continue
            except EOFError:
                break
    
    print("\nGame Over!")
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw due to insufficient material!")

if __name__ == "__main__":
    main()
