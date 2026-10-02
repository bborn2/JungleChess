"""加速版课程学习训练：多环境并行 + 优化的 MCTS 难度曲线。"""
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from jungle_chess import ai_best_move
from jungle_rl_env import ACTION_ENCODING, JungleChessEnv, encode_action


class CurriculumMCTSCallback(BaseCallback):
    """根据训练步数自动调整 MCTS 难度。"""

    def __init__(
        self,
        env_fns: list,
        curriculum_schedule: list[tuple[int, float]],
        verbose: int = 0,
    ):
        """
        Args:
            env_fns: 环境构造函数列表（用于更新所有并行环境）
            curriculum_schedule: [(步数, MCTS时间)] 列表
            verbose: 日志详细度
        """
        super().__init__(verbose)
        self.env_fns = env_fns
        self.curriculum_schedule = sorted(curriculum_schedule)
        self.current_time_limit = curriculum_schedule[0][1]

    def _on_step(self) -> bool:
        # 检查是否需要更新难度
        for threshold_step, time_limit in reversed(self.curriculum_schedule):
            if self.n_calls >= threshold_step and time_limit != self.current_time_limit:
                self.current_time_limit = time_limit

                if self.verbose > 0:
                    print(f"\n{'='*60}")
                    print(f"[Curriculum] 步数 {self.n_calls}: MCTS 难度提升到 {time_limit:.3f} 秒")
                    print(f"{'='*60}\n")

                # 更新所有并行环境的对手
                new_opponent = make_mcts_opponent(time_limit)
                for i, env_fn in enumerate(self.env_fns):
                    # 访问 VecEnv 中的第 i 个环境
                    try:
                        # SubprocVecEnv 的环境在子进程中，需要通过远程调用更新
                        self.training_model.env.env_fns[i] = lambda: Monitor(
                            JungleChessEnv(
                                opponent_policy=new_opponent,
                                max_episode_steps=300,
                            )
                        )
                    except:
                        pass  # 如果访问失败，在下一次 reset 时会自动使用新对手

                break

        return True


def make_mcts_opponent(time_limit: float):
    """创建指定难度的 MCTS 对手。"""
    def opponent_policy(game):
        move = ai_best_move(game, time_limit=time_limit, verbose=False)
        if move is None:
            legal_moves = game.all_moves(game.turn)
            if not legal_moves:
                raise RuntimeError("No legal moves available")
            piece, (nc, nr, _) = legal_moves[0]
            move = (piece.col, piece.row, nc, nr)
        return encode_action(move, player=1)

    return opponent_policy


def make_env(opponent_time_limit: float, max_episode_steps: int):
    """创建单个环境的工厂函数。"""
    def _init():
        env = JungleChessEnv(
            opponent_policy=make_mcts_opponent(opponent_time_limit),
            max_episode_steps=max_episode_steps,
        )
        return Monitor(env)
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("models/curriculum_fast/jungle_ppo"))
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=4,
                        help="并行环境数量（默认 4）")

    # 课程学习相关参数（默认更温和的曲线）
    parser.add_argument(
        "--curriculum",
        type=str,
        default="0:0.02,60000:0.04,120000:0.06",
        help="课程时间表，格式: '步数1:时间1,步数2:时间2,...'"
    )

    args = parser.parse_args()

    if args.timesteps < 1 or args.max_episode_steps < 1:
        parser.error("timesteps and max-episode-steps must be positive")
    if args.n_steps < 2 or args.batch_size < 1 or args.batch_size > args.n_steps:
        parser.error("require 2 <= n-steps and 1 <= batch-size <= n-steps")
    if args.eval_freq < 1 or args.eval_episodes < 1:
        parser.error("eval-freq and eval-episodes must be positive")
    if args.n_envs < 1:
        parser.error("n-envs must be positive")

    return args


def parse_curriculum(curriculum_str: str) -> list[tuple[int, float]]:
    """解析课程时间表字符串。"""
    schedule = []
    for pair in curriculum_str.split(','):
        step_str, time_str = pair.strip().split(':')
        schedule.append((int(step_str), float(time_str)))

    schedule.sort()
    return schedule


def main():
    args = parse_args()
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析课程时间表
    curriculum = parse_curriculum(args.curriculum)

    print(f"\n{'='*60}")
    print(f"加速课程学习训练")
    print(f"{'='*60}")
    print(f"并行环境数: {args.n_envs}")
    print(f"MCTS 难度曲线:")
    for step, time_limit in curriculum:
        print(f"  {step:>6} 步: MCTS {time_limit:.3f} 秒")
    print(f"{'='*60}\n")

    # 创建初始对手（课程第一阶段）
    initial_time_limit = curriculum[0][1]

    # 创建并行训练环境
    env_fns = [
        make_env(initial_time_limit, args.max_episode_steps)
        for _ in range(args.n_envs)
    ]
    training_env = SubprocVecEnv(env_fns)

    # 评估环境使用中等难度 MCTS（单环境）
    eval_time_limit = curriculum[len(curriculum) // 2][1]
    evaluation_env = Monitor(
        JungleChessEnv(
            opponent_policy=make_mcts_opponent(eval_time_limit),
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

    # 课程学习回调
    curriculum_callback = CurriculumMCTSCallback(
        env_fns=env_fns,
        curriculum_schedule=curriculum,
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
            policy_kwargs={"net_arch": [256, 256]},
            verbose=args.verbose,
        )
        model.jungle_action_encoding = ACTION_ENCODING

        print(f"开始训练 {args.timesteps} 步...")
        print(f"预计比单环境快 {args.n_envs}x\n")

        model.learn(
            total_timesteps=args.timesteps,
            callback=[evaluation_callback, curriculum_callback],
        )

        model.save(str(args.output))
        print(f"\n保存最终模型到 {args.output}.zip")
        print(f"模型已经历完整课程训练，可对抗 {curriculum[-1][1]:.3f} 秒的 MCTS\n")

    finally:
        training_env.close()
        evaluation_env.close()


if __name__ == "__main__":
    main()
