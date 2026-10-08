"""加速版课程学习训练：多环境并行 + 优化的 MCTS 难度曲线。"""
import argparse
import math
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from jungle_chess import ai_best_move
from jungle_rl_env import ACTION_ENCODING, JungleChessEnv, encode_action, validate_model_encoding


class CurriculumMCTSCallback(BaseCallback):
    """根据训练步数自动调整 MCTS 难度。"""

    def __init__(
        self,
        curriculum_schedule: list[tuple[int, float]],
        verbose: int = 0,
    ):
        """
        Args:
            curriculum_schedule: [(步数, MCTS时间)] 列表
            verbose: 日志详细度
        """
        super().__init__(verbose)
        self.curriculum_schedule = sorted(curriculum_schedule)
        self.current_time_limit = self.curriculum_schedule[0][1]

    def _on_step(self) -> bool:
        for threshold_step, time_limit in reversed(self.curriculum_schedule):
            if self.num_timesteps >= threshold_step:
                if time_limit != self.current_time_limit:
                    self.training_env.env_method("set_mcts_time_limit", time_limit)
                    self.current_time_limit = time_limit
                    if self.verbose > 0:
                        print(f"[Curriculum] 总采样步数 {self.num_timesteps}: MCTS {time_limit:.3f} 秒")
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


class CurriculumChessEnv(JungleChessEnv):
    """Allow VecEnv RPC to update the live opponent in each worker."""

    def __init__(self, time_limit: float, max_episode_steps: int):
        super().__init__(max_episode_steps=max_episode_steps)
        self.set_mcts_time_limit(time_limit)

    def set_mcts_time_limit(self, time_limit: float):
        if not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("MCTS time limit must be finite and positive")
        self.opponent_policy = make_mcts_opponent(time_limit)
        self.mcts_time_limit = time_limit


def make_env(opponent_time_limit: float, max_episode_steps: int):
    """创建单个环境的工厂函数。"""
    def _init():
        env = CurriculumChessEnv(
            time_limit=opponent_time_limit,
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
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, help="加载兼容检查点，新增训练步数及课程从零计数")
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
    if args.n_steps < 2 or args.batch_size < 2 or args.batch_size > args.n_steps * args.n_envs:
        parser.error("require n-steps >= 2 and 2 <= batch-size <= n-steps * n-envs")
    if args.eval_freq < 1 or args.eval_episodes < 1:
        parser.error("eval-freq and eval-episodes must be positive")
    if args.n_envs < 1:
        parser.error("n-envs must be positive")
    try:
        parse_curriculum(args.curriculum)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def parse_curriculum(curriculum_str: str) -> list[tuple[int, float]]:
    """解析课程时间表字符串。"""
    schedule = []
    for pair in curriculum_str.split(','):
        step_str, time_str = pair.strip().split(':')
        schedule.append((int(step_str), float(time_str)))

    schedule.sort()
    if not schedule or schedule[0][0] != 0:
        raise ValueError("curriculum must start at step 0")
    if len({step for step, _ in schedule}) != len(schedule):
        raise ValueError("curriculum thresholds must be unique")
    if any(step < 0 or not math.isfinite(limit) or limit <= 0 for step, limit in schedule):
        raise ValueError("curriculum steps must be nonnegative and times finite and positive")
    if any(current[1] < previous[1] for previous, current in zip(schedule, schedule[1:])):
        raise ValueError("curriculum time limits must not decrease")
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
        eval_freq=max(1, math.ceil(args.eval_freq / args.n_envs)),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=args.verbose,
    )

    # 课程学习回调
    curriculum_callback = CurriculumMCTSCallback(
        curriculum_schedule=curriculum,
        verbose=args.verbose,
    )

    try:
        if args.resume:
            model = MaskablePPO.load(
                args.resume,
                env=training_env,
                device=args.device,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                seed=args.seed,
                verbose=args.verbose,
            )
            validate_model_encoding(model)
            print(f"已加载 {args.resume}；历史步数 {model.num_timesteps}，本次课程从零开始")
        else:
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
        print(f"每轮采样 {args.n_steps * args.n_envs} 步；并行加速取决于 CPU 和对手耗时\n")

        model.learn(
            total_timesteps=args.timesteps,
            callback=[evaluation_callback, curriculum_callback],
            reset_num_timesteps=True,
        )

        model.save(str(args.output))
        print(f"\n保存最终模型到 {args.output}.zip")
        print(f"本次实际采样 {model.num_timesteps} 步，最后训练难度 {curriculum_callback.current_time_limit:.3f} 秒；棋力需独立评估\n")

    finally:
        training_env.close()
        evaluation_env.close()


if __name__ == "__main__":
    main()
