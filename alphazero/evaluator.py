"""Arena evaluation: pit two models against each other."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from main import JungleChess
from alphazero.config import NUM_SIMULATIONS_EVAL, EVAL_GAMES, WIN_RATE_THRESHOLD, MAX_GAME_MOVES
from alphazero.mcts import MCTS, select_action


def play_match(net_blue, net_red, device, num_simulations=NUM_SIMULATIONS_EVAL):
    """Play one game. net_blue plays as blue (player 1), net_red as red (-1).

    Returns: +1 if blue wins, -1 if red wins, 0 if draw.
    """
    mcts_blue = MCTS(net_blue, num_simulations, device=device, add_noise=False)
    mcts_red = MCTS(net_red, num_simulations, device=device, add_noise=False)

    game = JungleChess()
    player = 1
    move_number = 0

    while game.isGameOver() == 0 and move_number < MAX_GAME_MOVES:
        mcts = mcts_blue if player == 1 else mcts_red
        action_probs = mcts.search(game, player)
        action = select_action(action_probs, move_number, deterministic=True)
        move = _aid_to_move(action)
        if not game.make_move(move):
            legal = game.get_all_legal_moves(player)
            if not legal:
                break
            game.make_move(legal[0])
        player *= -1
        move_number += 1

    return game.isGameOver()


def evaluate(new_net, best_net, device, num_games=EVAL_GAMES):
    """Run arena matches between new_net and best_net.

    Returns win_rate of new_net (wins / total non-draw games).
    """
    new_net.eval()
    best_net.eval()

    wins = 0
    losses = 0
    draws = 0
    half = num_games // 2

    # new_net plays blue for first half
    for _ in range(half):
        result = play_match(new_net, best_net, device)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    # new_net plays red for second half
    for _ in range(half):
        result = play_match(best_net, new_net, device)
        if result == -1:    # red wins = new_net wins
            wins += 1
        elif result == 1:
            losses += 1
        else:
            draws += 1

    total = wins + losses
    win_rate = wins / total if total > 0 else 0.5
    print(f"  Arena: wins={wins} losses={losses} draws={draws} win_rate={win_rate:.3f}")
    return win_rate


def evaluate_vs_minimax(net, device, minimax_depth=4, num_games=20):
    """Evaluate network against minimax AI. Returns win rate."""
    from alphazero.mcts import MCTS, select_action

    net.eval()
    mcts = MCTS(net, NUM_SIMULATIONS_EVAL, device=device, add_noise=False)

    wins = 0
    for game_idx in range(num_games):
        game = JungleChess()
        # Alternate: net plays blue for even games, red for odd
        net_player = 1 if game_idx % 2 == 0 else -1
        player = 1
        move_number = 0

        while game.isGameOver() == 0 and move_number < MAX_GAME_MOVES:
            if player == net_player:
                action_probs = mcts.search(game, player)
                action = select_action(action_probs, move_number, deterministic=True)
                move = _aid_to_move(action)
                if not game.make_move(move):
                    legal = game.get_all_legal_moves(player)
                    if legal:
                        game.make_move(legal[0])
            else:
                maximizing = (player == -1)
                _, move = game.minimax2(minimax_depth, float('-inf'), float('inf'), maximizing)
                if move:
                    game.make_move(move)
            player *= -1
            move_number += 1

        result = game.isGameOver()
        if result == net_player:
            wins += 1

    win_rate = wins / num_games
    print(f"  vs minimax depth={minimax_depth}: {wins}/{num_games} = {win_rate:.2f}")
    return win_rate


def _aid_to_move(aid):
    fr = aid // 1000
    fc = (aid % 1000) // 100
    tr = (aid % 100) // 10
    tc = aid % 10
    return [fr, fc, tr, tc]
