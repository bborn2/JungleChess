"""Self-play game generation for AlphaZero training."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Pool

from main import JungleChess
from alphazero.config import (
    NUM_SIMULATIONS_TRAIN, NUM_WORKERS, SELF_PLAY_GAMES, MAX_GAME_MOVES, ACTION_SPACE
)
from alphazero.mcts import MCTS, select_action
from alphazero.utils import encode_state, flip_board_horizontal, flip_policy_horizontal


def play_one_game(args):
    """Play a single self-play game. Runs in a worker process.

    Args:
        args: (model_state_dict, device_str)

    Returns:
        list of (state_planes, mcts_policy, outcome) tuples
        outcome is from blue's perspective: +1 blue wins, -1 red wins
    """
    model_state, device_str = args

    # Import here to avoid issues with multiprocessing
    from alphazero.network import JungleChessNet

    device = torch.device(device_str)
    net = JungleChessNet().to(device)
    net.load_state_dict(model_state)
    net.eval()

    mcts = MCTS(net, NUM_SIMULATIONS_TRAIN, device=device_str, add_noise=True)

    game = JungleChess()
    player = 1   # blue starts
    history = []  # (state_planes, mcts_policy, player)
    move_number = 0

    while game.isGameOver() == 0 and move_number < MAX_GAME_MOVES:
        action_probs = mcts.search(game, player)
        state_planes = encode_state(game.board, player)

        history.append((state_planes, action_probs, player))

        action = select_action(action_probs, move_number, deterministic=False)
        move = _aid_to_move(action)
        if not game.make_move(move):
            # Fallback: pick any legal move
            legal = game.get_all_legal_moves(player)
            if not legal:
                break
            game.make_move(legal[0])

        player *= -1
        move_number += 1

    terminal = game.isGameOver()
    # terminal: +1 = blue wins, -1 = red wins, 0 = draw (treat as 0)
    outcome = float(terminal)

    samples = []
    for state_planes, policy, p in history:
        # outcome from this player's perspective
        z = outcome if p == 1 else -outcome
        samples.append((state_planes, policy, z))

        # Data augmentation: horizontal flip
        flipped_board = flip_board_horizontal(
            [[int(v) for v in row] for row in
             _planes_to_board_approx(state_planes)]
        )
        # We store the original board for flipping, not reconstructed
        # Just flip the policy vector
        flipped_policy = flip_policy_horizontal(policy)
        flipped_state = np.flip(state_planes, axis=2).copy()  # flip cols
        samples.append((flipped_state, flipped_policy, z))

    return samples


def generate_self_play_data(network, num_games, device):
    """Generate self-play games using parallel workers.

    Returns:
        list of (state_planes, mcts_policy, outcome) tuples
    """
    model_state = {k: v.cpu() for k, v in network.state_dict().items()}
    # Use 'cpu' for workers to avoid CUDA multiprocessing issues
    worker_device = 'cpu'
    args = [(model_state, worker_device)] * num_games

    num_workers = min(NUM_WORKERS, num_games, mp.cpu_count())

    all_samples = []
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = pool.map(play_one_game, args)
        for game_samples in results:
            all_samples.extend(game_samples)
    else:
        # Single-process fallback (easier to debug)
        for arg in args:
            all_samples.extend(play_one_game(arg))

    return all_samples


def _aid_to_move(aid):
    fr = aid // 1000
    fc = (aid % 1000) // 100
    tr = (aid % 100) // 10
    tc = aid % 10
    return [fr, fc, tr, tc]


def _planes_to_board_approx(planes):
    """Approximate board reconstruction from planes (for augmentation reference only)."""
    # Not used for actual board reconstruction; augmentation is done on planes directly
    return [[0] * 7 for _ in range(9)]
