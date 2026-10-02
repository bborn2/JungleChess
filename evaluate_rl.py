"""Evaluate a trained policy against random play or the existing MCTS AI."""
import argparse

from jungle_chess import COLS, ROWS, ai_best_move
from jungle_rl_env import JungleChessEnv, validate_model_encoding


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


def evaluate_games(model, env, episodes: int, seed: int):
    validate_model_encoding(model)
    results = {
        player: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_steps": 0,
            "pieces_captured": 0,
            "pieces_lost": 0,
        }
        for player in (1, 2)
    }
    for episode in range(episodes):
        player = 1 + episode % 2
        observation, _reset_info = env.reset(
            seed=seed + episode, options={"player": player}
        )
        terminated = truncated = False
        steps = 0
        initial_pieces = len(env.game.pieces)

        while not terminated and not truncated:
            action, _state = model.predict(
                observation,
                action_masks=env.action_masks(),
                deterministic=True,
            )
            observation, _reward, terminated, truncated, info = env.step(int(action))
            steps += 1

        winner = info.get("winner")
        outcome = "wins" if winner == player else "draws" if winner is None else "losses"
        results[player]["games"] += 1
        results[player][outcome] += 1
        results[player]["total_steps"] += steps

        # Count remaining pieces to calculate captures
        final_pieces = len(env.game.pieces)
        total_captured = initial_pieces - final_pieces
        opponent = 2 if player == 1 else 1

        # Count each side's remaining pieces
        player_pieces = sum(1 for p in env.game.pieces if p.player == player)
        opponent_pieces = sum(1 for p in env.game.pieces if p.player == opponent)

        # Pieces lost = initial 8 - remaining
        results[player]["pieces_lost"] += 8 - player_pieces
        results[opponent]["pieces_lost"] += 8 - opponent_pieces

        # Pieces captured by this player = opponent's losses in this game
        results[player]["pieces_captured"] += 8 - opponent_pieces
        results[opponent]["pieces_captured"] += 8 - player_pieces

    return results


def main():
    from sb3_contrib import MaskablePPO

    args = parse_args()
    opponent_policy = (
        make_mcts_opponent(args.mcts_time_limit)
        if args.opponent == "mcts"
        else None
    )
    env = JungleChessEnv(opponent_policy=opponent_policy)
    model = MaskablePPO.load(args.model, device=args.device)

    try:
        results = evaluate_games(model, env, args.episodes, args.seed)
    finally:
        env.close()

    wins = sum(result["wins"] for result in results.values())
    losses = sum(result["losses"] for result in results.values())
    draws = sum(result["draws"] for result in results.values())
    total_steps = sum(result["total_steps"] for result in results.values())
    total_captured = sum(result["pieces_captured"] for result in results.values())

    print(
        f"Opponent: {args.opponent}; games: {args.episodes}; "
        f"wins: {wins}; losses: {losses}; draws: {draws}; "
        f"win rate: {wins / args.episodes:.1%}"
    )
    print(
        f"Avg steps per game: {total_steps / args.episodes:.1f}; "
        f"total pieces captured: {total_captured}"
    )
    print()

    for player, result in results.items():
        rate = f"{result['wins'] / result['games']:.1%}" if result["games"] else "N/A"
        avg_steps = result["total_steps"] / result["games"] if result["games"] else 0
        capture_ratio = (
            f"{result['pieces_captured'] / result['pieces_lost']:.2f}"
            if result["pieces_lost"] > 0
            else "N/A"
        )
        print(
            f"{'Red' if player == 1 else 'Blue'}: games: {result['games']}; "
            f"wins: {result['wins']}; losses: {result['losses']}; "
            f"draws: {result['draws']}; win rate: {rate}"
        )
        print(
            f"  Avg steps: {avg_steps:.1f}; "
            f"captured: {result['pieces_captured']}; lost: {result['pieces_lost']}; "
            f"capture ratio: {capture_ratio}"
        )


if __name__ == "__main__":
    main()