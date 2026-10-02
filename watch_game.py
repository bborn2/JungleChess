"""观看 PPO 模型的完整对局过程，显示每一步的棋盘状态。"""
import argparse
import time

from sb3_contrib import MaskablePPO

from jungle_rl_env import JungleChessEnv, validate_model_encoding


def watch_game(model_path: str, opponent: str = "random", pause: float = 1.0):
    """观看一局完整对局。

    Args:
        model_path: 模型路径
        opponent: 对手类型 ("random" 或 "self")
        pause: 每步暂停秒数
    """
    # 加载模型
    print(f"加载模型: {model_path}")
    model = MaskablePPO.load(model_path)
    validate_model_encoding(model)

    # 设置对手
    if opponent == "self":
        from jungle_rl_env import encode_action, predict_rl_move

        def self_opponent(game):
            """使用同一个模型作为对手"""
            player = game.turn
            move = predict_rl_move(game, player, model)
            return encode_action(move, player=1)

        opponent_policy = self_opponent
        print("对手: 同模型自我对弈")
    else:
        opponent_policy = None
        print("对手: 随机策略")

    # 创建环境
    env = JungleChessEnv(opponent_policy=opponent_policy, max_episode_steps=300)

    # 开始对局
    observation, info = env.reset(seed=42, options={"player": 1})
    agent_player = info["agent_player"]
    player_name = "红方" if agent_player == 1 else "蓝方"

    print(f"\n{'='*60}")
    print(f"学习方执: {player_name}")
    print(f"{'='*60}\n")

    # 显示初始棋盘
    print("初始局面:")
    env.game.display()
    time.sleep(pause)

    step_count = 0
    terminated = truncated = False

    while not terminated and not truncated:
        step_count += 1

        # 模型选择动作
        action, _state = model.predict(
            observation,
            action_masks=env.action_masks(),
            deterministic=True,
        )

        print(f"\n{'='*60}")
        print(f"第 {step_count} 步 - {player_name}行动")
        print(f"{'='*60}")

        # 执行动作
        observation, reward, terminated, truncated, info = env.step(int(action))

        # 显示棋盘
        env.game.display()

        if terminated:
            winner = info.get("winner")
            if winner == agent_player:
                print(f"\n🎉 {player_name}（学习方）获胜！")
            elif winner is None:
                print(f"\n平局")
            else:
                opponent_name = "蓝方" if agent_player == 1 else "红方"
                print(f"\n{opponent_name}（对手）获胜")
            print(f"奖励: {reward:+.1f}")
        elif truncated:
            print(f"\n⏱️ 达到回合上限，游戏截断")

        time.sleep(pause)

    env.close()

    print(f"\n{'='*60}")
    print(f"对局结束，共 {step_count} 个环境步")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="模型路径（如 models/selfplay/best/best_model）"
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "self"],
        default="random",
        help="对手类型: random=随机策略, self=同模型自我对弈"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="每步暂停秒数（默认 1.0，设为 0 则不暂停）"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="观看几局（默认 1）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for game_num in range(args.games):
        if args.games > 1:
            print(f"\n\n{'#'*60}")
            print(f"第 {game_num + 1}/{args.games} 局")
            print(f"{'#'*60}\n")

        watch_game(args.model, args.opponent, args.pause)

        if game_num < args.games - 1:
            input("\n按回车继续下一局...")


if __name__ == "__main__":
    main()
