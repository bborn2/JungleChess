"""使用 MCTS 作为初始对手的训练脚本。"""
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from jungle_chess import ai_best_move
from jungle_rl_env import ACTION_ENCODING, JungleChessEnv


def make_mcts_opponent(time_limit: float = 0.1):
    """创建 MCTS 对手策略。"""
    from jungle_rl_env import encode_action

    def opponent_policy(game):
        """使用 MCTS 选择动作。"""
        move = ai_best_move(game, time_limit=time_limit, verbose=False)
        if move is None:
            # MCTS 失败时随机选择
            legal_moves = game.all_moves(game.turn)
            if not legal_moves:
                raise RuntimeError("No legal moves available")
            piece, (nc, nr, _) = legal_moves[0]
            move = (piece.col, piece.row, nc, nr)
        return encode_action(move, player=1)

    return opponent_policy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("models/mcts_trained/jungle_ppo"))
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--mcts-time-limit", type=float, default=0.1,
                        help="MCTS 每步思考时间（秒）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if args.timesteps < 1 or args.max_episode_steps < 1:
        parser.error("timesteps and max-episode-steps must be positive")
    if args.n_steps < 2 or args.batch_size < 1 or args.batch_size > args.n_steps:
        parser.error("require 2 <= n-steps and 1 <= batch-size <= n-steps")
    if args.eval_freq < 1 or args.eval_episodes < 1:
        parser.error("eval-freq and eval-episodes must be positive")
    return args


def main():
    args = parse_args()
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n训练对手: MCTS (每步 {args.mcts_time_limit} 秒)")
    print(f"这将比对随机对手慢很多，但能学到真正的策略\n")

    # 创建环境
    mcts_opponent = make_mcts_opponent(args.mcts_time_limit)
    training_env = Monitor(
        JungleChessEnv(
            opponent_policy=mcts_opponent,
            max_episode_steps=args.max_episode_steps,
        )
    )
    evaluation_env = Monitor(
        JungleChessEnv(
            opponent_policy=mcts_opponent,
            max_episode_steps=args.max_episode_steps,
        )
    )

    evaluation_callback = MaskableEvalCallback(
        evaluation_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "evaluations"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=args.verbose,
    )

    try:
        model = MaskablePPO(
            "MlpPolicy",
            training_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            policy_kwargs={"net_arch": [256, 256]},  # 更大的网络
            verbose=args.verbose,
        )
        model.jungle_action_encoding = ACTION_ENCODING

        model.learn(
            total_timesteps=args.timesteps,
            callback=evaluation_callback,
        )

        model.save(str(args.output))
        print(f"\n保存最终模型到 {args.output}.zip")

    finally:
        training_env.close()
        evaluation_env.close()


if __name__ == "__main__":
    main()
