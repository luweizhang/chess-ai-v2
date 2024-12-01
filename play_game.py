import requests
import json
import time
import sys

API_URL = "http://localhost:8001"

def play_move(fen, move):
    print(f"\nPlaying move: {move}")
    response = requests.post(f"{API_URL}/move", json={
        "fen": fen,
        "move": move
    })
    if response.status_code != 200:
        print(f"Error: {response.text}")
        sys.exit(1)
    return response.json()

def print_board_state(state):
    print(f"\nPosition: {state['fen']}")
    print(f"Last move: {state['last_move']}")
    print(f"Evaluation: {state['evaluation']}")
    print(f"Principal variation: {state['principal_variation']}")
    if state['is_game_over']:
        print(f"Game over! Result: {state['result']}")

# Start new game
print("Starting new game...")
state = requests.get(f"{API_URL}/new_game").json()
print_board_state(state)

# Play the moves
moves = [
    "d2d4",   # 1. d4
    "b1c3",   # 2. Nc3
    "g2g3",   # 3. g3
    "f1g2",   # 4. Bg2
    "g1f3",   # 5. Nf3
    "e1g1",   # 6. O-O
    "c1f4",   # 7. Bf4
    "d1d2",   # 8. Qd2
    "a1d1",   # 9. Rad1
    "f3e5",   # 10. Ne5
    "e2e4",   # 11. e4
    "e5f7",   # 12. Nxf7! (attacking the king)
    "d2e2",   # 13. Qe2 (threatening Qxe4+)
    "g2h3",   # 14. Bh3 (pinning the queen)
    "c3e4",   # 15. Ne4 (threatening Nf6+)
]

for move in moves:
    state = play_move(state['fen'], move)
    print_board_state(state)
    time.sleep(1)  # Wait a bit between moves

print("\nGame complete!")
