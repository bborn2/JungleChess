"""Gymnasium environment for training a Jungle Chess player."""
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jungle_chess import COLS, DEN, RIVER, ROWS, TRAPS, Game

BOARD_CELLS = COLS * ROWS
ACTION_COUNT = BOARD_CELLS * BOARD_CELLS
OBSERVATION_PLANES = 21
ACTION_ENCODING = "player-relative-v1"


def validate_model_encoding(model) -> None:
    if getattr(model, "jungle_action_encoding", None) != ACTION_ENCODING:
        raise ValueError(
            "Incompatible Jungle Chess action encoding. Retrain with train_rl.py; "
            "legacy checkpoints (including experiment_10k) use absolute actions."
        )


def _to_agent_view(col: int, row: int, player: int) -> tuple[int, int]:
    if player == 1:
        return col, row
    return COLS - 1 - col, ROWS - 1 - row


def encode_observation(game: Game, player: int) -> np.ndarray:
    """Encode a game from the chosen player's canonical point of view."""
    if player not in (1, 2):
        raise ValueError("player must be 1 or 2")

    observation = np.zeros((OBSERVATION_PLANES, ROWS, COLS), dtype=np.int8)
    for piece in game.pieces:
        col, row = _to_agent_view(piece.col, piece.row, player)
        side = 0 if piece.player == player else 1
        observation[side * 8 + piece.rank - 1, row, col] = 1

    opponent = 2 if player == 1 else 1
    _mark_cells(observation[16], (DEN[player],), player)
    _mark_cells(observation[17], (DEN[opponent],), player)
    _mark_cells(observation[18], RIVER, player)
    _mark_cells(observation[19], TRAPS[player], player)
    _mark_cells(observation[20], TRAPS[opponent], player)
    return observation


def _mark_cells(plane: np.ndarray, cells, player: int) -> None:
    for col, row in cells:
        view_col, view_row = _to_agent_view(col, row, player)
        plane[view_row, view_col] = 1


def legal_action_mask(game: Game, player: int) -> np.ndarray:
    """Build a MaskablePPO action mask for a player's current turn."""
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    if game.winner is not None or game.turn != player:
        return mask

    for piece, (nc, nr, _captured) in game.all_moves(player):
        mask[encode_action((piece.col, piece.row, nc, nr), player)] = True
    return mask


def encode_action(move: tuple[int, int, int, int], player: int = 1) -> int:
    """Encode a board move in the chosen player's coordinate system."""
    from_col, from_row = _to_agent_view(move[0], move[1], player)
    to_col, to_row = _to_agent_view(move[2], move[3], player)
    return (from_col * ROWS + from_row) * BOARD_CELLS + to_col * ROWS + to_row


def decode_action(action: int, player: int = 1) -> tuple[int, int, int, int]:
    """Decode a player-relative action back to absolute board coordinates."""
    if not 0 <= action < ACTION_COUNT:
        raise ValueError("action is outside the action space")
    from_idx, to_idx = divmod(action, BOARD_CELLS)
    from_col, from_row = divmod(from_idx, ROWS)
    to_col, to_row = divmod(to_idx, ROWS)
    from_col, from_row = _to_agent_view(from_col, from_row, player)
    to_col, to_row = _to_agent_view(to_col, to_row, player)
    return from_col, from_row, to_col, to_row


def predict_rl_move(game: Game, player: int, model) -> tuple[int, int, int, int]:
    """Predict and validate a legal move using a trained masked policy."""
    validate_model_encoding(model)
    mask = legal_action_mask(game, player)
    if not mask.any():
        raise RuntimeError("No legal actions are available for the PPO player")

    action, _state = model.predict(
        encode_observation(game, player),
        action_masks=mask,
        deterministic=True,
    )
    action = int(action)
    if not 0 <= action < ACTION_COUNT or not mask[action]:
        raise RuntimeError("PPO predicted an illegal action despite the mask")
    return decode_action(action, player)


class JungleChessEnv(gym.Env):
    """Train one player against a random or caller-provided opponent.

    Each environment step contains one learner move and, when the game is
    ongoing, one opponent reply. Opponent policies receive the current Game
    and return an absolute-board action encoded as ``from_idx * 63 + to_idx``.
    Learner actions instead use the same player-relative view as observations.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent_policy: Callable[[Game], int] | None = None,
        max_episode_steps: int = 300,
        render_mode: str | None = None,
    ):
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")

        self.opponent_policy = opponent_policy
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(OBSERVATION_PLANES, ROWS, COLS),
            dtype=np.int8,
        )
        self.game: Game | None = None
        self.agent_player: int | None = None
        self.episode_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        player = options.get("player")
        if player is None:
            player = int(self.np_random.integers(1, 3))
        if player not in (1, 2):
            raise ValueError("options['player'] must be 1 or 2")

        self.game = Game()
        self.agent_player = player
        self.episode_steps = 0
        if self.game.turn != self.agent_player:
            self._play_opponent_turn()

        return self._observation(), {"agent_player": self.agent_player}

    def step(self, action: int):
        if self.game is None or self.agent_player is None:
            raise RuntimeError("Call reset() before step()")
        if self.game.winner is not None:
            raise RuntimeError("The game has ended; call reset()")

        mask = self.action_masks()
        if not self.action_space.contains(action) or not mask[int(action)]:
            self.game.winner = 2 if self.agent_player == 1 else 1
            return self._observation(), -1.0, True, False, {
                "agent_player": self.agent_player,
                "winner": self.game.winner,
                "invalid_action": True,
            }

        self._apply_action(self.agent_player, int(action), canonical=True)
        if self.game.winner is None:
            self._play_opponent_turn()

        reward = 0.0
        terminated = self.game.winner is not None
        if terminated:
            reward = 1.0 if self.game.winner == self.agent_player else -1.0
        elif not self.game.all_moves(self.agent_player):
            self.game.winner = 2 if self.agent_player == 1 else 1
            reward = -1.0
            terminated = True

        self.episode_steps += 1
        truncated = not terminated and self.episode_steps >= self.max_episode_steps
        info = {
            "agent_player": self.agent_player,
            "winner": self.game.winner,
        }
        return self._observation(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask compatible with sb3-contrib MaskablePPO."""
        if (
            self.game is None
            or self.agent_player is None
        ):
            return np.zeros(ACTION_COUNT, dtype=np.bool_)
        return legal_action_mask(self.game, self.agent_player)

    def render(self):
        if self.render_mode == "human" and self.game is not None:
            self.game.display()

    def _apply_action(self, player: int, action: int, *, canonical: bool = False) -> bool:
        if self.game is None or self.game.turn != player:
            return False

        from_col, from_row, to_col, to_row = decode_action(
            action, player if canonical else 1
        )
        piece = self.game.piece_at(from_col, from_row)
        if piece is None or piece.player != player:
            return False

        for nc, nr, captured in self.game.get_moves(piece):
            if (nc, nr) == (to_col, to_row):
                self.game.apply_move(piece, nc, nr, captured)
                return True
        return False

    def _play_opponent_turn(self):
        if self.game is None or self.agent_player is None:
            return

        opponent = 2 if self.agent_player == 1 else 1
        legal_moves = self.game.all_moves(opponent)
        if not legal_moves:
            self.game.winner = self.agent_player
            return

        if self.opponent_policy is None:
            piece, (nc, nr, captured) = legal_moves[
                int(self.np_random.integers(len(legal_moves)))
            ]
            self.game.apply_move(piece, nc, nr, captured)
            return

        action = self.opponent_policy(self.game)
        if not self._apply_action(opponent, int(action)):
            raise ValueError("opponent_policy returned an illegal action")

    def _observation(self) -> np.ndarray:
        if self.game is None or self.agent_player is None:
            return np.zeros((OBSERVATION_PLANES, ROWS, COLS), dtype=np.int8)
        return encode_observation(self.game, self.agent_player)