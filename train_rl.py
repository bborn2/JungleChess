"""Train a masked PPO policy for Jungle Chess."""
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from jungle_rl_env import ACTION_ENCODING, JungleChessEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("models/jungle_ppo"))
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
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

    training_env = Monitor(
        JungleChessEnv(max_episode_steps=args.max_episode_steps)
    )
    evaluation_env = Monitor(
        JungleChessEnv(max_episode_steps=args.max_episode_steps)
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
            policy_kwargs={"net_arch": [128, 128]},
            verbose=args.verbose,
        )
        model.jungle_action_encoding = ACTION_ENCODING
        model.learn(
            total_timesteps=args.timesteps,
            callback=evaluation_callback,
        )
        model.save(str(args.output))
        print(f"Saved final policy to {args.output}.zip")
    finally:
        training_env.close()
        evaluation_env.close()


if __name__ == "__main__":
    main()