# Jungle Chess

一个使用 Python 实现的命令行斗兽棋。项目包含棋规、存档、人类对战和一个基于蒙特卡洛树搜索（MCTS）的 AI。

## 运行

需要 Python 3.11 或更高版本。用 `uv` 安装依赖并启动：

```bash
uv sync
uv run python jungle_chess.py
```

选择双人对战或人机对战，按提示输入棋子坐标和目标序号；输入 `q` 可保存当前棋局并退出。菜单中的 `[4] 人 vs PPO 模型` 默认加载仓库附带的新检查点 `models/canonical_100k/best/best_model`，也可以输入其他兼容的模型路径；该模式支持选择红方或蓝方，无需先训练。

在本机 WSL 上已有独立的 CPU 环境，可避免使用项目目录中未完成安装的 `.venv`：

```bash
source /home/kun/.venvs/junglechess-rl/bin/activate
python jungle_chess.py
```

选择 `4`、选择红/蓝方，再在模型路径处回车即可。该虚拟环境路径仅适用于本机；其他机器使用上面的 `uv sync` 安装。

## AI 当前实现

菜单中的 `[2]` 和 `[3]` 使用 MCTS，`[4]` 使用训练后的 PPO 模型。MCTS 搜索过程反复执行选择、扩展、模拟和回传，并用访问次数最多的根节点子项选择着法。模拟阶段使用简单的吃子/入窝启发式，同时加入随机走法；棋盘搜索使用紧凑表示以加快模拟。

MCTS 只在当前局面进行搜索，搜索结果不会作为已学习的模型保存。PPO 通过独立训练脚本更新神经网络；人机对战仅加载模型推理，不在线训练。

## 强化学习环境

`jungle_rl_env.py` 提供 Gymnasium 环境 `JungleChessEnv`，学习方每次行动后由环境自动执行一次对手回合。默认对手随机选择合法着法，也可以通过 `opponent_policy` 传入自定义策略；策略接收 `Game`，返回原始棋盘坐标编码的整数动作（可调用 `encode_action(move)`），该接口与学习方的规范视角动作不同。

观察空间是 `(21, 9, 7)`：双方各 8 个棋子等级平面，加上己方巢穴、对方巢穴、河流和双方陷阱平面。动作空间是 `Discrete(3969)`，动作编号为 `起点索引 * 63 + 终点索引`，其中格子索引是 `列 * 9 + 行`。`action_masks()` 返回当前合法动作掩码，可供 `sb3-contrib` 的 `MaskablePPO` 使用。

学习方的观察、动作编号及掩码统一采用玩家视角：蓝方棋盘旋转 180 度，红方不变。`encode_action(move, player)` 将实际走法编码，`decode_action(action, player)` 将动作还原为原始棋盘坐标；环境和 CLI 使用同一转换规则。

胜负奖励为 `+1/-1`；到达回合上限时以截断结束，奖励为 `0`。非法动作会判负。棋规差分测试和环境测试可运行：

```bash
uv run python -m unittest -v test_jungle_chess
```

## 训练和评估

仓库附带 `models/canonical_100k/` 使用修复后编码从零训练的模型，可直接用于评估和人机对战。`train_rl.py` 每次从零开始训练，并非继续训练已有权重；重复使用相同输出目录会覆盖其中的模型和评估日志，新的实验请另选目录。

### 基础训练（对随机对手）

使用 `MaskablePPO` 对随机策略训练，训练过程中会定期评估并保存最佳模型；结束时另存最终模型：

```bash
uv run python train_rl.py --timesteps 500000 --device cpu
```

训练步数、回合上限、评估频率、批次大小和模型路径都可通过 `--help` 查看和调整。先用短训练确认流程：

```bash
uv run python train_rl.py --timesteps 1024 --n-steps 128 --batch-size 64 --eval-freq 512 --eval-episodes 2 --output models/smoke
```

### 自我对弈训练

使用 `train_selfplay.py` 实现自我对弈循环，训练过程中定期将当前策略保存为快照，并用作对手：

```bash
uv run python train_selfplay.py --timesteps 200000 --selfplay-update-freq 25000 --output models/selfplay/jungle_ppo --device cpu
```

可选参数：
- `--initial-opponent models/canonical_100k/best/best_model`：使用已训练模型作为初始对手
- `--selfplay-update-freq 25000`：每 25k 步更新对手策略（默认值）

自我对弈比对随机对手训练更具挑战性，适合在基础训练后进一步提升策略强度。

### 评估

独立评估脚本从红方开始交替执方，偶数局数保证双方各占一半，奇数局数红方多一盘。输出包括总成绩、红蓝方各自的局数、胜负平、胜率、平均回合数、吃子数和吃子比率。训练过程中的评估回调仍使用环境随机选边，不替代这项分色评估。

```bash
uv run python evaluate_rl.py --model models/canonical_100k/jungle_ppo --opponent random --episodes 50 --seed 5000
uv run python evaluate_rl.py --model models/canonical_100k/best/best_model --opponent random --episodes 50 --seed 5000
uv run python evaluate_rl.py --model models/canonical_100k/best/best_model --opponent mcts --episodes 20 --seed 6000 --mcts-time-limit 0.1
```

评估指标包括：
- **胜率**：总胜局 / 总局数，分总体和红蓝各方
- **平均回合数**：每局的环境步数（含学习方和对手各自的行动）
- **吃子统计**：己方吃掉对手棋子数和己方损失棋子数
- **吃子比率**：己方吃子数 / 己方损失数，大于 1 表示交换优势

## 100k 训练记录

2026-09-30 使用修复后的 `player-relative-v1` 编码从头训练 100k 步，耗时约 9 分钟。PPO 按 1,024 步 rollout 采样，实际完成 100,352 步。每 10k 步的 10 盘随机对手评估平均奖励依次为 `0, 1, 1, 1, 1, 1, 1, 1, 1, 1`；训练回调在 20k 步首次达到 `+1.0` 并保存最佳检查点。

- `models/canonical_100k/jungle_ppo.zip`：最终权重，100,352 步。
- `models/canonical_100k/best/best_model.zip`：训练期间最佳检查点，20,000 步。
- `models/canonical_100k/evaluations/evaluations.npz`：训练期各检查点评估记录。

以下为独立确定性评估，使用固定种子、红蓝各半；MCTS 每步搜索 0.1 秒：

| 模型 | 对手 | 总胜/负/平 | 总胜率 | 执红胜/负/平 | 执蓝胜/负/平 |
| --- | --- | --- | --- | --- | --- |
| 最终权重 | 随机，50 盘 | 50/0/0 | 100% | 25/0/0 | 25/0/0 |
| 最佳检查点 | 随机，50 盘 | 50/0/0 | 100% | 25/0/0 | 25/0/0 |
| 最终权重 | MCTS，20 盘 | 19/1/0 | 95% | 9/1/0 | 10/0/0 |
| 最佳检查点 | MCTS，20 盘 | 19/1/0 | 95% | 9/1/0 | 10/0/0 |

这些结果表明模型已学会击败本项目的随机策略和当前有限时预算的 MCTS，但 MCTS 使用启发式 rollout，0.1 秒预算不代表强引擎；每项评估也只有 20 或 50 盘，不能据此保证对更强或不同策略的泛化胜率。
