"""带奖励塑形的 Jungle Chess 环境。"""
import gymnasium as gym
import numpy as np

from jungle_chess import RANK, Game
from jungle_rl_env import (
    ACTION_COUNT, JungleChessEnv, OBSERVATION_PLANES, ROWS, COLS,
    encode_observation, legal_action_mask, decode_action
)


class ShapedRewardJungleEnv(JungleChessEnv):
    """加入中间奖励的环境，引导模型学习战术。"""

    def step(self, action: int):
        if self.game is None or self.agent_player is None:
            raise RuntimeError("Call reset() before step()")
        if self.game.winner is not None:
            raise RuntimeError("The game has ended; call reset()")

        # 记录行动前的状态
        before_material = self._material_balance()
        before_territory = self._territory_score()

        # 执行原有 step 逻辑
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

        # 计算塑形后的奖励
        reward = 0.0
        terminated = self.game.winner is not None

        if terminated:
            # 终局奖励保持不变
            reward = 1.0 if self.game.winner == self.agent_player else -1.0
        elif not self.game.all_moves(self.agent_player):
            self.game.winner = 2 if self.agent_player == 1 else 1
            reward = -1.0
            terminated = True
        else:
            # 中间奖励：子力变化 + 领地优势
            after_material = self._material_balance()
            after_territory = self._territory_score()

            material_gain = (after_material - before_material) * 0.01  # 吃子奖励
            territory_gain = (after_territory - before_territory) * 0.005  # 位置奖励
            reward = material_gain + territory_gain

        self.episode_steps += 1
        truncated = not terminated and self.episode_steps >= self.max_episode_steps
        info = {
            "agent_player": self.agent_player,
            "winner": self.game.winner,
        }
        return self._observation(), reward, terminated, truncated, info

    def _material_balance(self) -> float:
        """计算子力平衡（己方 - 对方）。"""
        if self.game is None or self.agent_player is None:
            return 0.0

        agent_material = sum(
            RANK[p.name] for p in self.game.pieces if p.player == self.agent_player
        )
        opponent = 2 if self.agent_player == 1 else 1
        opponent_material = sum(
            RANK[p.name] for p in self.game.pieces if p.player == opponent
        )
        return agent_material - opponent_material

    def _territory_score(self) -> float:
        """计算领地优势（棋子靠近对方巢穴的距离）。"""
        if self.game is None or self.agent_player is None:
            return 0.0

        from jungle_chess import DEN

        opponent = 2 if self.agent_player == 1 else 1
        enemy_den_col, enemy_den_row = DEN[opponent]

        score = 0.0
        for piece in self.game.pieces:
            if piece.player == self.agent_player:
                # 曼哈顿距离，越近越好
                dist = abs(piece.col - enemy_den_col) + abs(piece.row - enemy_den_row)
                score -= dist  # 距离越小，分数越高
            else:
                # 对手棋子远离我方巢穴更好
                my_den_col, my_den_row = DEN[self.agent_player]
                dist = abs(piece.col - my_den_col) + abs(piece.row - my_den_row)
                score += dist * 0.5  # 权重小一些

        return score
