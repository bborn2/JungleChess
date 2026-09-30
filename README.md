# Jungle Chess

一个使用 Python 实现的命令行斗兽棋。项目包含棋规、存档、人类对战和一个基于蒙特卡洛树搜索（MCTS）的 AI。

## 运行

需要 Python 3.11 或更高版本。用 `uv` 安装依赖并启动：

```bash
uv sync
uv run python jungle_chess.py
```

选择双人对战或人机对战，按提示输入棋子坐标和目标序号；输入 `q` 可保存当前棋局并退出。

## AI 当前实现

AI 使用 MCTS，而不是 minimax 或强化学习模型。搜索过程反复执行选择、扩展、模拟和回传，并用访问次数最多的根节点子项选择着法。模拟阶段使用简单的吃子/入窝启发式，同时加入随机走法；棋盘搜索使用紧凑表示以加快模拟。

目前没有训练过程、神经网络或从对局中更新的策略。MCTS 只在当前局面进行搜索，搜索结果不会作为已学习的模型保存。

## 强化学习环境

`jungle_rl_env.py` 提供 Gymnasium 环境 `JungleChessEnv`，学习方每次行动后由环境自动执行一次对手回合。默认对手随机选择合法着法，也可以通过 `opponent_policy` 传入自定义策略；策略接收 `Game`，返回编码后的整数动作。

观察空间是 `(21, 9, 7)`：双方各 8 个棋子等级平面，加上己方巢穴、对方巢穴、河流和双方陷阱平面。动作空间是 `Discrete(3969)`，动作编号为 `起点索引 * 63 + 终点索引`，其中格子索引是 `列 * 9 + 行`。`action_masks()` 返回当前合法动作掩码，可供 `sb3-contrib` 的 `MaskablePPO` 使用。

胜负奖励为 `+1/-1`；到达回合上限时以截断结束，奖励为 `0`。非法动作会判负。棋规差分测试和环境测试可运行：

```bash
uv run python -m unittest -v test_jungle_chess
```

## 训练和评估

使用 `MaskablePPO` 对随机策略训练，训练过程中会定期评估并保存最佳模型；结束时另存最终模型：

```bash
uv run python train_rl.py --timesteps 500000 --device cpu
```

训练步数、回合上限、评估频率、批次大小和模型路径都可通过 `--help` 查看和调整。先用短训练确认流程：

```bash
uv run python train_rl.py --timesteps 1024 --n-steps 128 --batch-size 64 --eval-freq 512 --eval-episodes 2 --output models/smoke
```

对随机对手或 MCTS 统计胜负：

```bash
uv run python evaluate_rl.py --model models/jungle_ppo --opponent random --episodes 20
uv run python evaluate_rl.py --model models/jungle_ppo --opponent mcts --episodes 10 --mcts-time-limit 0.25
```

训练默认是对随机策略的单边训练，不是自我对弈。MCTS 和模型快照适合作为后续逐步增强的评估/训练对手；结果应以固定测试局数的胜率衡量，而不只看训练奖励。
