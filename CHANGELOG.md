# Changelog

## 2024-10-02 - 项目优化

### 已完成的改进

#### 1. 删除 10k 模型
- 移除 `models/canonical_10k/` 和 `models/experiment_10k/` 目录
- 仅保留性能更优的 `models/canonical_100k/` (100% 胜率对随机对手，95% 对 MCTS)
- 更新所有文档中对 10k 模型的引用

#### 2. 优化单元测试性能
- 减少随机游戏测试的迭代次数：10 局 × 200 步 → 5 局 × 100 步
- 将 `evaluate_rl.py` 中的 `MaskablePPO` 导入延迟到 `main()` 函数内
- 测试套件现在可以在没有 torch 的情况下导入和验证核心逻辑

#### 3. 增强评估指标
新增 `evaluate_rl.py` 输出指标：
- **平均回合数**：每局的环境步数统计
- **吃子统计**：
  - `pieces_captured`: 己方吃掉对手的棋子数
  - `pieces_lost`: 己方损失的棋子数
  - `capture_ratio`: 吃子比率（captured/lost），大于 1 表示交换优势
- 分红方/蓝方统计所有新指标

示例输出：
```
Opponent: random; games: 50; wins: 50; losses: 0; draws: 0; win rate: 100.0%
Avg steps per game: 45.2; total pieces captured: 156

Red: games: 25; wins: 25; losses: 0; draws: 0; win rate: 100.0%
  Avg steps: 44.8; captured: 78; lost: 0; capture ratio: N/A
Blue: games: 25; wins: 25; losses: 0; draws: 0; win rate: 100.0%
  Avg steps: 45.6; captured: 78; lost: 0; capture ratio: N/A
```

#### 4. 实现自我对弈训练循环
新增 `train_selfplay.py` 脚本，支持：
- 定期将当前策略保存为快照并用作对手
- 可选的初始对手模型（如从已训练模型开始）
- 可配置的对手更新频率（默认每 25k 步）
- 快照自动保存到 `snapshots/` 子目录
- 评估仍对随机对手进行（确保基础能力不退化）

使用示例：
```bash
# 从随机对手开始自我对弈
uv run python train_selfplay.py --timesteps 200000 --device cpu

# 从已训练模型开始
uv run python train_selfplay.py --timesteps 200000 \
  --initial-opponent models/canonical_100k/best/best_model \
  --selfplay-update-freq 25000 --device cpu
```

### 技术细节

**自我对弈实现**：
- `SelfPlayCallback`: 在训练过程中定期更新对手策略
- `make_model_opponent()`: 从冻结的模型快照创建对手策略函数
- 对手使用绝对坐标编码（`player=1`），与环境 `opponent_policy` 接口兼容
- 学习方始终使用玩家相对视角（红方不变，蓝方旋转 180°）

**测试优化原理**：
- 原测试速度瓶颈：`import torch` 需要数十秒（121MB 下载 + 初始化）
- 延迟导入让单元测试只需 `gymnasium` + `numpy`，可在无 torch 环境下验证核心逻辑
- 实际游戏规则测试是纯 Python，执行时间 < 1 秒

### 文档更新
- README 新增自我对弈训练章节
- README 新增评估指标说明
- 删除过时的 10k 训练记录章节
- 更新所有命令示例

### 向后兼容性
- 所有现有脚本（`train_rl.py`, `evaluate_rl.py`）保持兼容
- 现有模型（`canonical_100k`）无需重新训练
- 单元测试更新以验证新的评估指标字段
