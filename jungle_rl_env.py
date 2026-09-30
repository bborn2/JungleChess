"""Gymnasium environment for training a Jungle Chess player."""
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jungle_chess import COLS, DEN, RIVER, ROWS, TRAPS, Game

BOARD_CELLS = COLS * ROWS
ACTION_COUNT = BOARD_CELLS * BOARD_CELLS
OBSERVATION_PLANES = 21


class JungleChessEnv(gym.Env):
    """Train one player against a random or caller-provided opponent.

    Each environment step contains one learner move and, when the game is
    ongoing, one opponent reply. Opponent policies receive the current Game
    and return an action encoded as ``from_idx * 63 + to_idx``.
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

        self._apply_action(self.agent_player, int(action))
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
        mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        if (
            self.game is None
            or self.agent_player is None
            or self.game.winner is not None
            or self.game.turn != self.agent_player
        ):
            return mask

        for piece, (nc, nr, _captured) in self.game.all_moves(self.agent_player):
            from_idx = piece.col * ROWS + piece.row
            to_idx = nc * ROWS + nr
            mask[from_idx * BOARD_CELLS + to_idx] = True
        return mask

    def render(self):
        if self.render_mode == "human" and self.game is not None:
            self.game.display()

    def _apply_action(self, player: int, action: int) -> bool:
        if self.game is None or self.game.turn != player:
            return False

        from_idx, to_idx = divmod(action, BOARD_CELLS)
        from_col, from_row = divmod(from_idx, ROWS)
        to_col, to_row = divmod(to_idx, ROWS)
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
        observation = np.zeros(
            (OBSERVATION_PLANES, ROWS, COLS), dtype=np.int8
        )
        if self.game is None or self.agent_player is None:
            return observation

        for piece in self.game.pieces:
            col, row = self._to_agent_view(piece.col, piece.row)
            side = 0 if piece.player == self.agent_player else 1
            observation[side * 8 + piece.rank - 1, row, col] = 1

        opponent = 2 if self.agent_player == 1 else 1
        self._mark_cells(observation[16], (DEN[self.agent_player],))
        self._mark_cells(observation[17], (DEN[opponent],))
        self._mark_cells(observation[18], RIVER)
        self._mark_cells(observation[19], TRAPS[self.agent_player])
        self._mark_cells(observation[20], TRAPS[opponent])
        return observation

    def _mark_cells(self, plane: np.ndarray, cells):
        for col, row in cells:
            view_col, view_row = self._to_agent_view(col, row)
            plane[view_row, view_col] = 1

    def _to_agent_view(self, col: int, row: int) -> tuple[int, int]:
        if self.agent_player == 1:
            return col, row
        return COLS - 1 - col, ROWS - 1 - row