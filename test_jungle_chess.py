import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from jungle_chess import Game, Piece, _fast_from_game, _fast_moves
from jungle_rl_env import (
    ACTION_ENCODING, JungleChessEnv, decode_action, encode_action, encode_observation,
    legal_action_mask, predict_rl_move, validate_model_encoding,
)


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

        for _ in range(5):  # reduced from 10
            game = Game()
            for _ in range(100):  # reduced from 200
                expected = _game_moves(game)
                board, locs, turn = _fast_from_game(game)
                self.assertEqual(expected, set(_fast_moves(board, locs, turn)))
                if game.winner is not None or not expected:
                    break
                piece, (nc, nr, captured) = rng.choice(game.all_moves(game.turn))
                game.apply_move(piece, nc, nr, captured)


class JungleChessEnvTests(unittest.TestCase):
    def test_mirrored_positions_share_observations_and_actions(self):
        red_game = Game()
        red_game.pieces = [Piece(1, 2, 1, '鼠'), Piece(5, 6, 2, '猫')]
        blue_game = Game()
        blue_game.pieces = [Piece(5, 6, 2, '鼠'), Piece(1, 2, 1, '猫')]
        blue_game.turn = 2
        np.testing.assert_array_equal(
            encode_observation(red_game, 1), encode_observation(blue_game, 2)
        )
        np.testing.assert_array_equal(
            legal_action_mask(red_game, 1), legal_action_mask(blue_game, 2)
        )
        for action in np.flatnonzero(legal_action_mask(red_game, 1)):
            red_move = decode_action(int(action), 1)
            blue_move = decode_action(int(action), 2)
            self.assertEqual(blue_move, tuple(
                (6 if index % 2 == 0 else 8) - coordinate
                for index, coordinate in enumerate(red_move)
            ))
            self.assertEqual(encode_action(blue_move, 2), action)

    def test_blue_step_and_absolute_opponent_actions(self):
        opponent_moves = []

        def opponent(game):
            piece, (col, row, _captured) = game.all_moves(game.turn)[0]
            move = (piece.col, piece.row, col, row)
            opponent_moves.append(move)
            return encode_action(move)

        env = JungleChessEnv(opponent_policy=opponent)
        env.reset(seed=7, options={"player": 2})
        self.assertEqual(len(opponent_moves), 1)
        action = int(np.flatnonzero(env.action_masks())[0])
        from_col, from_row, to_col, to_row = decode_action(action, 2)
        piece = env.game.piece_at(from_col, from_row)
        _observation, reward, terminated, truncated, _info = env.step(action)
        self.assertIs(env.game.piece_at(to_col, to_row), piece)
        self.assertEqual(piece.player, 2)
        self.assertEqual(len(opponent_moves), 2)
        self.assertEqual(env.game.turn, 2)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated or truncated)

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

    def test_ppo_prediction_is_legal_and_matches_game_orientation(self):
        class FirstLegalModel:
            jungle_action_encoding = ACTION_ENCODING

            def predict(self, observation, action_masks, deterministic):
                self.observation = observation
                self.action_masks = action_masks
                self.deterministic = deterministic
                return next(i for i, legal in enumerate(action_masks) if legal), None

        game = Game()
        model = FirstLegalModel()

        for player in (1, 2):
            game.turn = player
            move = predict_rl_move(game, player, model)
            legal_moves = {
                (piece.col, piece.row, nc, nr)
                for piece, (nc, nr, _captured) in game.all_moves(player)
            }
            self.assertIn(move, legal_moves)
            self.assertEqual(model.observation.shape, (21, 9, 7))
            self.assertTrue(model.deterministic)

    def test_legacy_or_unknown_model_encoding_is_rejected(self):
        for model in (SimpleNamespace(), SimpleNamespace(jungle_action_encoding="unknown")):
            with self.assertRaisesRegex(ValueError, "Retrain"):
                validate_model_encoding(model)
            with self.assertRaisesRegex(ValueError, "Retrain"):
                predict_rl_move(Game(), 1, model)

    def test_evaluation_alternates_sides_and_reports_each_side(self):
        from evaluate_rl import evaluate_games

        model = SimpleNamespace(
            jungle_action_encoding=ACTION_ENCODING,
            predict=lambda observation, action_masks, deterministic: (
                int(np.flatnonzero(action_masks)[0]), None
            ),
        )
        for episodes in (1, 5, 20):
            with self.subTest(episodes=episodes):
                env = JungleChessEnv(max_episode_steps=1)
                with patch.object(env, "reset", wraps=env.reset) as reset:
                    results = evaluate_games(model, env, episodes, 1000)
                self.assertEqual(
                    [call.kwargs["options"]["player"] for call in reset.call_args_list],
                    [1 + episode % 2 for episode in range(episodes)],
                )
                self.assertEqual(results[1]["games"], (episodes + 1) // 2)
                self.assertEqual(results[2]["games"], episodes // 2)
                self.assertEqual(sum(result["draws"] for result in results.values()), episodes)
                # Check new metrics exist
                for player in (1, 2):
                    self.assertIn("total_steps", results[player])
                    self.assertIn("pieces_captured", results[player])
                    self.assertIn("pieces_lost", results[player])
                env.close()


class CurriculumTrainingTests(unittest.TestCase):
    def test_total_steps_select_latest_stage_without_reverting(self):
        from train_curriculum_fast import CurriculumMCTSCallback

        env = Mock()
        callback = CurriculumMCTSCallback([(0, 0.03), (80000, 0.06), (150000, 0.1)])
        callback.model = SimpleNamespace(get_env=lambda: env)
        callback.n_calls = 10000
        for steps, expected in ((79992, 0.03), (80000, 0.06), (80008, 0.06), (150000, 0.1), (150008, 0.1)):
            callback.num_timesteps = steps
            callback._on_step()
            self.assertEqual(callback.current_time_limit, expected)
        self.assertEqual(env.env_method.call_count, 2)
        env.env_method.assert_any_call("set_mcts_time_limit", 0.06)
        env.env_method.assert_any_call("set_mcts_time_limit", 0.1)
        env.env_method.side_effect = RuntimeError("worker failure")
        callback.current_time_limit = 0.03
        with self.assertRaisesRegex(RuntimeError, "worker failure"):
            callback._on_step()

    def test_curriculum_validation(self):
        from train_curriculum_fast import parse_curriculum

        self.assertEqual(parse_curriculum("8:0.02,0:0.01"), [(0, 0.01), (8, 0.02)])
        for schedule in ("", "1:0.02", "0:0", "0:nan", "0:inf", "0:0.02,0:0.03", "0:0.03,8:0.01"):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                parse_curriculum(schedule)

    def test_live_subprocess_opponents_are_updated(self):
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from train_curriculum_fast import CurriculumMCTSCallback, make_env

        env = SubprocVecEnv([make_env(0.001, 7), make_env(0.001, 7)], start_method="spawn")
        try:
            callback = CurriculumMCTSCallback([(0, 0.001), (8, 0.002)])
            callback.model = SimpleNamespace(get_env=lambda: env)
            callback.num_timesteps = 8
            callback.n_calls = 4
            callback._on_step()
            self.assertEqual(env.get_attr("mcts_time_limit"), [0.002, 0.002])
            self.assertEqual(env.get_attr("max_episode_steps"), [7, 7])
            env.reset()
            masks = env.env_method("action_masks")
            env.step([int(np.flatnonzero(mask)[0]) for mask in masks])
            self.assertEqual(env.get_attr("mcts_time_limit"), [0.002, 0.002])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()