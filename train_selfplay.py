"""Train a masked PPO policy using self-play against frozen snapshots."""
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from jungle_rl_env import ACTION_ENCODING, JungleChessEnv, encode_action, predict_rl_move


def make_model_opponent(model_path: Path):
    """Create an opponent policy from a frozen model checkpoint."""
    frozen_model = MaskablePPO.load(str(model_path))

    def opponent_policy(game):
        """Return absolute-coordinate action for the current game state."""
        player = game.turn
        move = predict_rl_move(game, player, frozen_model)
        return encode_action(move, player=1)  # absolute coordinates

    return opponent_policy


class SelfPlayCallback(BaseCallback):
    """Periodically update the opponent to the latest trained policy."""

    def __init__(
        self,
        opponent_env: JungleChessEnv,
        update_freq: int,
        snapshot_dir: Path,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.opponent_env = opponent_env
        self.update_freq = update_freq
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            # Save current model as opponent snapshot
            snapshot_path = self.snapshot_dir / f"snapshot_{self.snapshot_count}.zip"
            self.model.save(str(snapshot_path))
            self.snapshot_count += 1

            if self.verbose > 0:
                print(f"\n[SelfPlay] Saved snapshot {snapshot_path.name} at {self.n_calls} steps")
                print(f"[SelfPlay] Updating opponent policy...")

            # Update opponent environment with new policy
            self.opponent_env.opponent_policy = make_model_opponent(snapshot_path)

            if self.verbose > 0:
                print(f"[SelfPlay] Opponent updated to snapshot {self.snapshot_count - 1}\n")

        return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("models/selfplay/jungle_ppo"))
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--selfplay-update-freq", type=int, default=25_000,
                        help="Steps between opponent policy updates")
    parser.add_argument("--initial-opponent", type=Path, default=None,
                        help="Path to initial opponent model (default: random)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if args.timesteps < 1 or args.max_episode_steps < 1:
        parser.error("timesteps and max-episode-steps must be positive")
    if args.n_steps < 2 or args.batch_size < 1 or args.batch_size > args.n_steps:
        parser.error("require 2 <= n-steps and 1 <= batch-size <= n-steps")
    if args.eval_freq < 1 or args.eval_episodes < 1:
        parser.error("eval-freq and eval-episodes must be positive")
    if args.selfplay_update_freq < args.n_steps:
        parser.error("selfplay-update-freq must be >= n-steps")
    return args


def main():
    args = parse_args()
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Set up initial opponent policy
    if args.initial_opponent:
        if args.verbose > 0:
            print(f"Loading initial opponent from {args.initial_opponent}")
        initial_opponent_policy = make_model_opponent(args.initial_opponent)
    else:
        if args.verbose > 0:
            print("Starting with random opponent")
        initial_opponent_policy = None

    # Create training and evaluation environments
    training_env = Monitor(
        JungleChessEnv(
            opponent_policy=initial_opponent_policy,
            max_episode_steps=args.max_episode_steps,
        )
    )
    evaluation_env = Monitor(
        JungleChessEnv(max_episode_steps=args.max_episode_steps)
    )

    # Set up evaluation callback (evaluates against random opponent)
    evaluation_callback = MaskableEvalCallback(
        evaluation_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "evaluations"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=args.verbose,
    )

    # Set up self-play callback
    selfplay_callback = SelfPlayCallback(
        opponent_env=training_env.env,
        update_freq=args.selfplay_update_freq,
        snapshot_dir=snapshot_dir,
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

        if args.verbose > 0:
            print(f"\nStarting self-play training for {args.timesteps} steps")
            print(f"Opponent updates every {args.selfplay_update_freq} steps\n")

        model.learn(
            total_timesteps=args.timesteps,
            callback=[evaluation_callback, selfplay_callback],
        )

        model.save(str(args.output))
        print(f"\nSaved final policy to {args.output}.zip")
        print(f"Snapshots saved to {snapshot_dir}")
        print(f"Total opponent updates: {selfplay_callback.snapshot_count}")

    finally:
        training_env.close()
        evaluation_env.close()


if __name__ == "__main__":
    main()
