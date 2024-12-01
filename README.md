# Chess AI

A modern chess application with multiple AI implementations and a web-based interface.

## Features

- Web-based chess interface with beautiful piece designs
- Multiple AI implementations:
  - Minimax with Alpha-Beta Pruning
    - Depth: 3 plies
    - Time limit: 0.5 seconds per move
    - Move ordering with captures and center control
    - Transposition table for position caching
  - Monte Carlo Tree Search (MCTS)
    - 1000 simulations per move
    - UCB1 exploration (constant: 1.4)
    - 5-second time limit
    - Position caching with transposition table
- Real-time position evaluation
- Principal variation display
- Dynamic AI switching during gameplay

## Project Structure

```
chess_ai/
├── frontend/             # React TypeScript frontend
│   ├── src/             # Frontend source code
│   ├── package.json     # Frontend dependencies
│   └── vite.config.ts   # Vite configuration
├── src/                 # Python backend
│   ├── models/         # AI implementations
│   │   ├── base_ai.py  # Abstract base class
│   │   ├── minimax_ai.py # Minimax with alpha-beta
│   │   ├── mcts_ai.py  # Monte Carlo Tree Search
│   │   └── neural_ai.py # Neural network evaluator
│   └── server/        # FastAPI server
│       └── main.py    # Server implementation
└── requirements.txt    # Python dependencies
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Node.js dependencies:
```bash
cd frontend
npm install
```

## Running the Application

1. Start the backend server:
```bash
python -m uvicorn src.server.main:app --reload --port 8001
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Open http://localhost:5173 in your browser

## Usage

1. The game starts with you playing as White against the AI
2. Choose your preferred AI opponent from the dropdown (Minimax or MCTS)
3. Make moves by clicking and dragging pieces
4. Use the "New Game" button to start a fresh game
5. The info box shows:
   - Current position evaluation
   - Principal variation (best predicted line of play)
   - Game status and results

## Technical Details

### Frontend
- React with TypeScript
- react-chessboard for the chess interface
- Vite for fast development and building
- Axios for API communication

### Backend
- FastAPI for the web server
- python-chess for game logic and move validation
- Custom AI implementations with modular design
- Real-time move generation and evaluation

## Future Improvements

1. Neural network training integration
2. Opening book support
3. Game history and analysis
4. Adjustable AI difficulty levels
5. Multi-game performance statistics
# chess-ai-v2
# chess-ai-v2
