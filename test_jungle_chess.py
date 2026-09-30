import random
import unittest

from jungle_chess import Game, Piece, _fast_from_game, _fast_moves
from jungle_rl_env import JungleChessEnv


def _game_moves(game):
    return {
        (piece.col * 9 + piece.row, nc * 9 + nr)
        for piece, (nc, nr, _captured) in game.all_moves(game.turn)
    }


class FastMoveParityTests(unittest.TestCase):
    def test_attacker_in_enemy_trap_has_reduced_rank(self):
        game = Game()
        game.pieces = [Piece(3, 1, 1, '豹'), Piece(4, 1, 2, '猫')]
        game.turn = 1

        board, locs, turn = _fast_from_game(game)
        fast_moves = set(_fast_moves(board, locs, turn))

        self.assertEqual(_game_moves(game), fast_moves)
        self.assertNotIn((3 * 9 + 1, 4 * 9 + 1), fast_moves)

    def test_rat_capture_rules_at_river_edge(self):
        game = Game()
        game.pieces = [Piece(6, 3, 1, '鼠'), Piece(5, 3, 2, '鼠')]
        game.turn = 1

        board, locs, turn = _fast_from_game(game)
        self.assertEqual(_game_moves(game), set(_fast_moves(board, locs, turn)))

        game.pieces = [Piece(5, 3, 1, '鼠'), Piece(6, 3, 2, '鼠')]
        board, locs, turn = _fast_from_game(game)
        self.assertEqual(_game_moves(game), set(_fast_moves(board, locs, turn)))

        game.pieces = [Piece(5, 3, 1, '鼠'), Piece(4, 3, 2, '猫')]
        board, locs, turn = _fast_from_game(game)
        self.assertEqual(_game_moves(game), set(_fast_moves(board, locs, turn)))

    def test_moves_match_during_reachable_random_games(self):
        rng = random.Random(0)

        for _ in range(10):
            game = Game()
            for _ in range(200):
                expected = _game_moves(game)
                board, locs, turn = _fast_from_game(game)
                self.assertEqual(expected, set(_fast_moves(board, locs, turn)))
                if game.winner is not None or not expected:
                    break
                piece, (nc, nr, captured) = rng.choice(game.all_moves(game.turn))
                game.apply_move(piece, nc, nr, captured)


class JungleChessEnvTests(unittest.TestCase):
    def test_observation_and_action_mask_follow_gymnasium_spaces(self):
        env = JungleChessEnv()
        observation, info = env.reset(seed=7, options={"player": 1})

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["agent_player"], 1)
        mask = env.action_masks()
        self.assertEqual(int(mask.sum()), len(env.game.all_moves(1)))

        action = next(index for index, legal in enumerate(mask) if legal)
        observation, reward, terminated, truncated, _info = env.step(action)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIn(reward, (-1.0, 0.0, 1.0))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertGreater(env.action_masks().sum(), 0)

    def test_invalid_action_loses_without_mutating_the_board(self):
        env = JungleChessEnv()
        env.reset(seed=7, options={"player": 1})
        before = [(p.col, p.row, p.player, p.name) for p in env.game.pieces]
        invalid_action = next(
            index for index, legal in enumerate(env.action_masks()) if not legal
        )

        _observation, reward, terminated, truncated, info = env.step(invalid_action)

        self.assertEqual(reward, -1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(before, [(p.col, p.row, p.player, p.name) for p in env.game.pieces])

    def test_episode_limit_truncates_a_nonterminal_game(self):
        env = JungleChessEnv(max_episode_steps=1)
        env.reset(seed=7, options={"player": 1})
        action = next(
            index for index, legal in enumerate(env.action_masks()) if legal
        )

        _observation, _reward, terminated, truncated, _info = env.step(action)

        self.assertFalse(terminated)
        self.assertTrue(truncated)


if __name__ == '__main__':
    unittest.main()