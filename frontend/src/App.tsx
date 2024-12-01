import { useState, useCallback, useEffect } from 'react'
import { Chessboard } from 'react-chessboard'
import { Chess } from 'chess.js'
import axios from 'axios'

// API URL
const API_URL = 'http://localhost:8001'

interface GameState {
  fen: string
  legal_moves: string[]
  is_game_over: boolean
  result: string | null
  last_move: string | null
  principal_variation: string[]
  evaluation: number
}

function App() {
  const [game, setGame] = useState<Chess>(new Chess())
  const [currentPosition, setCurrentPosition] = useState<string>(game.fen())
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [aiType, setAiType] = useState<string>("minimax")  // Default to minimax
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Styles for the container and controls
  const containerStyle = {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    padding: '20px',
    gap: '20px'
  };
  
  const controlsStyle = {
    display: 'flex',
    gap: '10px',
    marginBottom: '20px'
  };

  const buttonStyle = {
    padding: '10px 20px',
    fontSize: '16px',
    cursor: 'pointer',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '4px'
  };

  const selectStyle = {
    padding: '10px',
    fontSize: '16px',
    borderRadius: '4px'
  };

  // Start new game
  const startNewGame = useCallback(async () => {
    console.log('Starting new game...')
    try {
      setLoading(true)
      setError(null)
      const response = await axios.get(`${API_URL}/new_game`)
      console.log('New game response:', response.data)
      const newState = response.data
      setGameState(newState)
      setGame(new Chess(newState.fen))
    } catch (error: any) {
      console.error('Error starting new game:', error)
      setError('Failed to start new game')
    } finally {
      setLoading(false)
    }
  }, [])

  // Handle piece movement
  const onDrop = useCallback(async (sourceSquare: string, targetSquare: string) => {
    console.log('Move attempt:', { sourceSquare, targetSquare })
    
    try {
      // Try the move locally first
      const gameCopy = new Chess(game.fen())
      const move = gameCopy.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q'
      })

      if (!move) {
        console.log('Invalid move locally')
        return false
      }

      // Get the move in UCI format
      const uciMove = `${sourceSquare}${targetSquare}${move.promotion || ''}`
      console.log('Move details:', {
        currentFen: game.fen(),
        move: uciMove,
        newFen: gameCopy.fen()
      })

      // Update UI immediately with player's move
      setGame(gameCopy)
      setLoading(true)
      setError(null)

      // Send move to backend
      const response = await axios.post(`${API_URL}/move`, {
        fen: game.fen(),
        move: uciMove,
        ai_type: aiType  // Send current AI type
      })

      // Update state with AI's response
      const newState = response.data
      console.log('AI response:', newState)
      setGameState(newState)
      const finalGame = new Chess(newState.fen)
      setGame(finalGame)
      return true

    } catch (error: any) {
      console.error('Move error:', error.response?.data || error)
      setError(error.response?.data?.detail || 'Failed to make move')
      // Revert to previous state
      setGame(new Chess(gameState?.fen || game.fen()))
      return false
    } finally {
      setLoading(false)
    }
  }, [game, gameState, aiType])

  // Initialize game on mount
  useEffect(() => {
    startNewGame()
  }, [startNewGame])

  return (
    <div style={containerStyle}>
      <h1>Chess AI</h1>
      
      <div style={{ 
        display: 'flex', 
        gap: '20px',
        alignItems: 'flex-start',
        width: '1000px'
      }}>
        <div style={{ width: '600px' }}>
          <Chessboard 
            position={game.fen()} 
            onPieceDrop={onDrop}
            customBoardStyle={{
              borderRadius: '4px',
              boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)',
            }}
            customLightSquareStyle={{ backgroundColor: '#e8f5e9' }}
            customDarkSquareStyle={{ backgroundColor: '#a5d6a7' }}
            areArrowsAllowed={true}
            customPieces={{
              bP: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bp.png" alt="Black Pawn" />,
              bN: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bn.png" alt="Black Knight" />,
              bB: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bb.png" alt="Black Bishop" />,
              bR: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/br.png" alt="Black Rook" />,
              bQ: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png" alt="Black Queen" />,
              bK: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bk.png" alt="Black King" />,
              wP: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wp.png" alt="White Pawn" />,
              wN: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wn.png" alt="White Knight" />,
              wB: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wb.png" alt="White Bishop" />,
              wR: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wr.png" alt="White Rook" />,
              wQ: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wq.png" alt="White Queen" />,
              wK: () => <img style={{ width: '100%', height: '100%' }} src="https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wk.png" alt="White King" />
            }}
          />
        </div>

        <div style={{
          padding: '20px',
          backgroundColor: '#f5f5f5',
          borderRadius: '8px',
          width: '300px',
          flexShrink: 0
        }}>
          <div style={{ marginBottom: '15px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <button 
              onClick={startNewGame}
              style={{
                padding: '8px 16px',
                fontSize: '14px',
                cursor: 'pointer',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px'
              }}
            >
              New Game
            </button>
            
            <select 
              value={aiType} 
              onChange={(e) => setAiType(e.target.value)}
              style={{
                padding: '8px',
                fontSize: '14px',
                borderRadius: '4px',
                border: '1px solid #ddd'
              }}
            >
              <option value="minimax">Minimax AI</option>
              <option value="mcts">MCTS AI</option>
            </select>
          </div>

          <div style={{ 
            display: 'flex',
            flexDirection: 'column',
            gap: '10px'
          }}>
            {loading && <div>AI is thinking...</div>}
            {error && <div style={{ color: 'red' }}>{error}</div>}
            {gameState && (
              <>
                <div>Evaluation: {gameState.evaluation.toFixed(2)}</div>
                <div>Principal variation: {gameState.principal_variation.join(', ')}</div>
                {gameState.is_game_over && (
                  <div style={{ fontWeight: 'bold' }}>
                    Game Over! {gameState.result}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
