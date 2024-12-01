from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
from ..models.mcts_ai import MCTSAI
from ..models.minimax_ai import MinimaxAI
from ..models.base_ai import ChessAI
from typing import Optional, Dict, List
import traceback

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AIs
ai_implementations = {
    "minimax": MinimaxAI(depth=4),  # 4-ply search depth
    "mcts": MCTSAI(simulation_limit=1000)  # 1000 simulations
}

# Current AI
current_ai: ChessAI = ai_implementations["minimax"]  # Default to minimax

class MoveRequest(BaseModel):
    fen: str
    move: Optional[str] = None
    ai_type: Optional[str] = None  # Optional field to switch AI

class GameState(BaseModel):
    fen: str
    legal_moves: List[str]
    is_game_over: bool
    result: Optional[str]
    last_move: Optional[str]
    principal_variation: List[str]
    evaluation: float
    ai_type: str  # Added to show which AI is being used

@app.post("/move")
async def make_move(request: MoveRequest) -> GameState:
    """Make a move in the game"""
    try:
        global current_ai
        
        print(f"\n=== Received move request ===")
        print(f"FEN: {request.fen}")
        print(f"Move: {request.move}")
        print(f"AI Type: {request.ai_type}")
        
        # Switch AI if requested
        if request.ai_type and request.ai_type in ai_implementations:
            current_ai = ai_implementations[request.ai_type]
            print(f"Switched to {current_ai.name}")
        
        if not isinstance(request.move, str):
            raise HTTPException(status_code=400, detail=f"Move must be a string, got {type(request.move)}")
        
        # Create board from FEN
        board = chess.Board(request.fen)
        print(f"Board created with FEN: {board.fen()}")
        print(f"Legal moves: {[m.uci() for m in board.legal_moves]}")
        
        # If move is provided, make player's move
        if request.move:
            print(f"\nAttempting player move: {request.move}")
            try:
                # Parse move, handling potential promotion
                move_str = str(request.move)  # Ensure it's a string
                move = chess.Move.from_uci(move_str)
                print(f"Parsed UCI move: {move}")
                
                if move not in board.legal_moves:
                    print(f"Illegal move: {move} not in {[m.uci() for m in board.legal_moves]}")
                    raise HTTPException(status_code=400, detail=f"Illegal move: {move}")
                
                board.push(move)
                print(f"Player move successful, new position: {board.fen()}")
            except ValueError as e:
                print(f"Error parsing move: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid move format: {e}")
            
            # Check if game is over after player's move
            if board.is_game_over():
                print("Game over after player's move")
                return GameState(
                    fen=board.fen(),
                    legal_moves=[],
                    is_game_over=True,
                    result=board.outcome().result(),
                    last_move=request.move,
                    principal_variation=[],
                    evaluation=0.0,
                    ai_type=current_ai.name
                )
        
        # Get AI's move
        print("\nGetting AI's move...")
        ai_move = current_ai.get_best_move(board)
        if ai_move:
            print(f"AI chose move: {ai_move.uci()}")
            board.push(ai_move)
        
        # Get evaluation
        print("\nGetting evaluation...")
        evaluation = 0.0
        if isinstance(current_ai, MinimaxAI):
            evaluation = current_ai.evaluate_position(board)
        elif isinstance(current_ai, MCTSAI):
            policy_dict, evaluation = current_ai.neural_evaluator.evaluate_position(board)
        print(f"Evaluation: {evaluation}")
        
        response = GameState(
            fen=board.fen(),
            legal_moves=[m.uci() for m in board.legal_moves],
            is_game_over=board.is_game_over(),
            result=board.outcome().result() if board.is_game_over() else None,
            last_move=ai_move.uci() if ai_move else None,
            principal_variation=[m.uci() for m in current_ai.pv_line],
            evaluation=float(evaluation),
            ai_type=current_ai.name
        )
        print(f"\nSending response: {response}")
        return response
        
    except Exception as e:
        print(f"\nError in make_move: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/new_game")
async def new_game() -> GameState:
    """Start a new game"""
    try:
        print("\n=== Starting new game ===")
        board = chess.Board()
        print(f"Initial position: {board.fen()}")
        
        # Get initial evaluation
        evaluation = 0.0
        if isinstance(current_ai, MinimaxAI):
            evaluation = current_ai.evaluate_position(board)
        elif isinstance(current_ai, MCTSAI):
            policy_dict, evaluation = current_ai.neural_evaluator.evaluate_position(board)
        print(f"Initial evaluation: {evaluation}")
        
        response = GameState(
            fen=board.fen(),
            legal_moves=[move.uci() for move in board.legal_moves],
            is_game_over=False,
            result=None,
            last_move=None,
            principal_variation=[],
            evaluation=evaluation,
            ai_type=current_ai.name
        )
        print(f"Sending response: {response}")
        return response
        
    except Exception as e:
        print(f"Error in new_game: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
