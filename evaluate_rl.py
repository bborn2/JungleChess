"""Evaluate a trained policy against random play or the existing MCTS AI."""
import argparse

from sb3_contrib import MaskablePPO

from jungle_chess import COLS, ROWS, ai_best_move
from jungle_rl_env import JungleChessEnv


def encode_move(move):
    from_col, from_row, to_col, to_row = move
    from_index = from_col * ROWS + from_row
    to_index = to_col * ROWS + to_row
    return from_index * (COLS * ROWS) + to_index


def make_mcts_opponent(time_limit):
    def choose_action(game):
        move = ai_best_move(game, time_limit=time_limit, verbose=False)
        if move is None:
            raise RuntimeError("MCTS returned no move in a non-terminal position")
        return encode_move(move)

    return choose_action


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/jungle_ppo")
    parser.add_argument("--opponent", choices=("random", "mcts"), default="random")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--mcts-time-limit", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.episodes < 1 or args.mcts_time_limit <= 0:
        parser.error("episodes and mcts-time-limit must be positive")
    return args


def main():
    args = parse_args()
    opponent_policy = (
        make_mcts_opponent(args.mcts_time_limit)
        if args.opponent == "mcts"
        else None
    )
    env = JungleChessEnv(opponent_policy=opponent_policy)
    model = MaskablePPO.load(args.model, device=args.device)
    wins = losses = draws = 0

    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            info = reset_info
            while not terminated and not truncated:
                action, _state = model.predict(
                    observation,
                    action_masks=env.action_masks(),
                    deterministic=True,
                )
                observation, _reward, terminated, truncated, info = env.step(
                    int(action)
                )

            winner = info.get("winner")
            if winner == info["agent_player"]:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1
    finally:
        env.close()

    print(
        f"Opponent: {args.opponent}; games: {args.episodes}; "
        f"wins: {wins}; losses: {losses}; draws: {draws}; "
        f"win rate: {wins / args.episodes:.1%}"
    )


if __name__ == "__main__":
    main()