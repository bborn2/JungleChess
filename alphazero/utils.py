import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from alphazero.config import (
    BOARD_ROWS, BOARD_COLS, INPUT_CHANNELS, ACTION_SPACE
)

# Piece type index: piece_value (2-9) → channel index (0-7)
PIECE_TO_IDX = {v: i for i, v in enumerate([2, 3, 4, 5, 6, 7, 8, 9])}


def encode_state(board, current_player):
    """Encode board into (INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS) float32 tensor.

    Channels:
      0-7  : blue pieces (rat, cat, dog, wolf, leopard, tiger, lion, elephant)
      8-15 : red pieces (same order)
      16   : current player (all 1s = blue to move, all 0s = red to move)
    """
    planes = np.zeros((INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            cell = board[r][c]
            if 10 < cell < 100:          # blue piece
                pt = cell // 10
                idx = PIECE_TO_IDX.get(pt)
                if idx is not None:
                    planes[idx, r, c] = 1.0
            elif cell > 100:             # red piece
                pt = cell // 100
                idx = PIECE_TO_IDX.get(pt)
                if idx is not None:
                    planes[8 + idx, r, c] = 1.0
    if current_player == 1:
        planes[16, :, :] = 1.0
    return planes


def encode_action(move):
    """[from_row, from_col, to_row, to_col] → int action id."""
    fr, fc, tr, tc = move
    return fr * 1000 + fc * 100 + tr * 10 + tc


def decode_action(action_id):
    """int action id → [from_row, from_col, to_row, to_col]."""
    fr = action_id // 1000
    fc = (action_id % 1000) // 100
    tr = (action_id % 100) // 10
    tc = action_id % 10
    return [fr, fc, tr, tc]


def get_legal_action_mask(game, player):
    """Return bool array of shape (ACTION_SPACE,) with True for legal actions."""
    mask = np.zeros(ACTION_SPACE, dtype=bool)
    for move in game.get_all_legal_moves(player):
        aid = encode_action(move)
        if 0 <= aid < ACTION_SPACE:
            mask[aid] = True
    return mask


def flip_board_horizontal(board):
    """Mirror board left-right (data augmentation)."""
    return [row[::-1] for row in board]


def flip_policy_horizontal(policy):
    """Mirror policy vector to match horizontally flipped board."""
    flipped = np.zeros_like(policy)
    for aid in range(ACTION_SPACE):
        move = decode_action(aid)
        fr, fc, tr, tc = move
        fc_f = BOARD_COLS - 1 - fc
        tc_f = BOARD_COLS - 1 - tc
        new_aid = encode_action([fr, fc_f, tr, tc_f])
        if 0 <= new_aid < ACTION_SPACE:
            flipped[new_aid] = policy[aid]
    return flipped
