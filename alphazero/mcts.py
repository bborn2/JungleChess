import math
import numpy as np
import torch

from alphazero.config import (
    C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON,
    TEMPERATURE_THRESHOLD, TEMP_HIGH, TEMP_LOW, ACTION_SPACE
)
from alphazero.utils import encode_state, encode_action, get_legal_action_mask


class MCTSNode:
    __slots__ = ('parent', 'action', 'prior', 'children',
                 'visit_count', 'total_value', 'is_expanded')

    def __init__(self, parent, action, prior):
        self.parent = parent
        self.action = action          # action that led to this node
        self.prior = prior            # P(s, a) from network
        self.children = {}            # action_id → MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, parent_visits):
        """PUCT formula."""
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u


class MCTS:
    def __init__(self, network, num_simulations, device='cpu', add_noise=False):
        self.network = network
        self.num_simulations = num_simulations
        self.device = device
        self.add_noise = add_noise   # True during self-play, False during eval

    def search(self, game, player):
        """Run MCTS from the current game state.

        Returns:
            action_probs: (ACTION_SPACE,) numpy array
            move_number : int (used for temperature schedule)
        """
        root = MCTSNode(parent=None, action=None, prior=1.0)
        self._expand(root, game, player)

        if self.add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            sim_game = game.copy()
            sim_player = player
            path = [node]

            # --- Selection ---
            while node.is_expanded and sim_game.isGameOver() == 0:
                node = self._select_child(node)
                sim_game.make_move(node.action)
                sim_player *= -1
                path.append(node)

            # --- Expansion & Evaluation ---
            terminal = sim_game.isGameOver()
            if terminal != 0:
                value = float(terminal)   # +1 blue wins, -1 red wins
            else:
                value = self._expand(node, sim_game, sim_player)

            # --- Backpropagation ---
            # value is from the perspective of sim_player after the last move,
            # so we flip sign as we go up the tree
            self._backpropagate(path, value, sim_player, player)

        return self._action_probs(root)

    # ------------------------------------------------------------------
    def _expand(self, node, game, player):
        """Query network, create child nodes. Returns value estimate."""
        state = encode_state(game.board, player)
        state_t = torch.FloatTensor(state).to(self.device)
        mask = torch.BoolTensor(get_legal_action_mask(game, player)).to(self.device)

        policy, value = self.network.predict(state_t, mask)

        for aid in np.where(mask.cpu().numpy())[0]:
            node.children[int(aid)] = MCTSNode(
                parent=node,
                action=self._aid_to_move(int(aid)),
                prior=float(policy[aid]),
            )
        node.is_expanded = True
        return value

    def _select_child(self, node):
        parent_visits = node.visit_count
        return max(node.children.values(), key=lambda c: c.ucb_score(parent_visits))

    def _backpropagate(self, path, value, last_player, root_player):
        """Propagate value up the path, flipping sign at each level."""
        # value is from last_player's perspective
        # root_player is the player at the root
        sign = 1 if last_player == root_player else -1
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value * sign
            sign *= -1

    def _add_dirichlet_noise(self, root):
        n = len(root.children)
        if n == 0:
            return
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * n)
        for child, eta in zip(root.children.values(), noise):
            child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * eta

    def _action_probs(self, root):
        probs = np.zeros(ACTION_SPACE, dtype=np.float32)
        for aid, child in root.children.items():
            probs[aid] = child.visit_count
        total = probs.sum()
        if total > 0:
            probs /= total
        return probs

    @staticmethod
    def _aid_to_move(aid):
        fr = aid // 1000
        fc = (aid % 1000) // 100
        tr = (aid % 100) // 10
        tc = aid % 10
        return [fr, fc, tr, tc]


def select_action(action_probs, move_number, deterministic=False):
    """Sample or argmax action from MCTS policy."""
    if deterministic:
        return int(np.argmax(action_probs))
    temp = TEMP_HIGH if move_number < TEMPERATURE_THRESHOLD else TEMP_LOW
    if temp < 0.01:
        return int(np.argmax(action_probs))
    # Apply temperature
    probs = action_probs ** (1.0 / temp)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
