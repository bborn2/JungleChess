"""Play against a trained AlphaZero model.

Usage:
    python play_alphazero.py --model models/iter_100.pt
    python play_alphazero.py --model models/iter_100.pt --you red
"""

import argparse
import numpy as np
import torch

from main import JungleChess
from alphazero.network import JungleChessNet
from alphazero.mcts import MCTS, select_action
from alphazero.config import NUM_SIMULATIONS_EVAL, MAX_GAME_MOVES


def get_human_move(game, player):
    while True:
        try:
            raw = input("Your move (fromRow fromCol toRow toCol): ").strip().split()
            move = [int(x) for x in raw]
            if len(move) == 4 and game.make_move(move):
                return move
            # make_move already applied the move if valid; undo by re-reading
            print("Illegal move, try again.")
            # Re-init game state is tricky here; just re-prompt
        except (ValueError, IndexError):
            print("Enter 4 numbers separated by spaces.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--you', type=str, default='red', choices=['blue', 'red'],
                        help='Which side you play (default: red)')
    parser.add_argument('--sims', type=int, default=NUM_SIMULATIONS_EVAL)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    net = JungleChessNet().to(device)
    ckpt = torch.load(args.model, map_location=device)
    net.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    net.eval()
    print(f"Loaded model from {args.model}")

    mcts = MCTS(net, args.sims, device=device_str, add_noise=False)

    human_player = 1 if args.you == 'blue' else -1
    ai_player = -human_player

    game = JungleChess()
    player = 1   # blue moves first
    move_number = 0

    while game.isGameOver() == 0 and move_number < MAX_GAME_MOVES:
        game.showBoard()

        if player == human_player:
            print(f"Your turn ({'blue' if player == 1 else 'red'})")
            # We need to handle make_move inside get_human_move carefully
            while True:
                try:
                    raw = input("Your move (fromRow fromCol toRow toCol): ").strip().split()
                    move = [int(x) for x in raw]
                    if len(move) == 4:
                        # Validate without applying
                        from alphazero.utils import get_legal_action_mask, encode_action
                        legal = game.get_all_legal_moves(player)
                        if move in legal or tuple(move) in [tuple(m) for m in legal]:
                            game.make_move(move)
                            break
                    print("Illegal move, try again.")
                except (ValueError, IndexError):
                    print("Enter 4 numbers separated by spaces.")
        else:
            print(f"AI thinking ({'blue' if player == 1 else 'red'}, {args.sims} sims)...")
            action_probs = mcts.search(game, player)
            action = select_action(action_probs, move_number, deterministic=True)
            move = [action // 1000, (action % 1000) // 100, (action % 100) // 10, action % 10]
            print(f"AI plays: {move}")
            if not game.make_move(move):
                legal = game.get_all_legal_moves(player)
                if legal:
                    game.make_move(legal[0])

        player *= -1
        move_number += 1

    game.showBoard()
    result = game.isGameOver()
    if result == 1:
        winner = 'Blue'
    elif result == -1:
        winner = 'Red'
    else:
        winner = 'Draw'

    print(f"\nGame over! {winner} wins.")
    if (result == 1 and human_player == 1) or (result == -1 and human_player == -1):
        print("You win!")
    elif result == 0:
        print("Draw.")
    else:
        print("AI wins!")


if __name__ == '__main__':
    main()
