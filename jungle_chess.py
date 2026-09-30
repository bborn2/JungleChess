#!/usr/bin/env python3
"""斗兽棋 (Jungle Chess) CLI"""
import math
import json
import os
import time
import random

SAVE_FILE = os.path.join(os.path.dirname(__file__), "savegame.json")

COLS, ROWS = 7, 9

DEN    = {1: (3, 8), 2: (3, 0)}
TRAPS  = {1: frozenset([(2,8),(4,8),(3,7)]), 2: frozenset([(2,0),(4,0),(3,1)])}
RIVER  = frozenset([(c,r) for c in (1,2) for r in (3,4,5)] +
                   [(c,r) for c in (4,5) for r in (3,4,5)])

RANK = {'象':8,'狮':7,'虎':6,'豹':5,'狗':4,'狼':3,'猫':2,'鼠':1}

# col, row, player, name
INIT = [
    (0,6,1,'象'),(6,6,1,'鼠'),(2,6,1,'狼'),(4,6,1,'豹'),
    (1,7,1,'猫'),(5,7,1,'狗'),
    (0,8,1,'虎'),(6,8,1,'狮'),
    (6,2,2,'象'),(0,2,2,'鼠'),(4,2,2,'狼'),(2,2,2,'豹'),
    (5,1,2,'猫'),(1,1,2,'狗'),
    (6,0,2,'虎'),(0,0,2,'狮'),
]


class Piece:
    def __init__(self, col, row, player, name):
        self.col, self.row, self.player, self.name = col, row, player, name
        self.rank = RANK[name]

    def pos(self): return (self.col, self.row)
    def __repr__(self): return f"{self.name}{'红' if self.player==1 else '蓝'}"


class Game:
    def __init__(self):
        self.pieces = [Piece(*args) for args in INIT]
        self.turn = 1
        self.winner = None

    def piece_at(self, col, row):
        for p in self.pieces:
            if p.col == col and p.row == row:
                return p
        return None

    def eff_rank(self, piece):
        """Effective rank (0 if in enemy trap)."""
        enemy = 2 if piece.player == 1 else 1
        return 0 if piece.pos() in TRAPS[enemy] else piece.rank

    def can_capture(self, atk, dfn):
        # Piece in river cannot capture land piece
        if atk.pos() in RIVER and dfn.pos() not in RIVER:
            return False
        # Rat-Elephant special rule
        if atk.name == '鼠' and dfn.name == '象':
            return True
        if atk.name == '象' and dfn.name == '鼠':
            return False
        return self.eff_rank(atk) >= self.eff_rank(dfn)

    def _jump(self, piece, dc, dr):
        """Lion/Tiger river jump. Returns (nc, nr, captured|None) or None."""
        nc, nr = piece.col + dc, piece.row + dr
        while (nc, nr) in RIVER:
            if self.piece_at(nc, nr):   # rat blocking
                return None
            nc += dc; nr += dr
        if not (0 <= nc < COLS and 0 <= nr < ROWS):
            return None
        if (nc, nr) == DEN[piece.player]:
            return None
        target = self.piece_at(nc, nr)
        if target is None:
            return (nc, nr, None)
        if target.player != piece.player and self.can_capture(piece, target):
            return (nc, nr, target)
        return None

    def get_moves(self, piece):
        moves = []
        for dc, dr in ((0,-1),(0,1),(-1,0),(1,0)):
            nc, nr = piece.col + dc, piece.row + dr
            if not (0 <= nc < COLS and 0 <= nr < ROWS):
                continue
            if (nc, nr) == DEN[piece.player]:
                continue
            target = self.piece_at(nc, nr)

            if (nc, nr) in RIVER:
                if piece.name == '鼠':
                    if target is None:
                        moves.append((nc, nr, None))
                    elif target.player != piece.player and self.can_capture(piece, target):
                        moves.append((nc, nr, target))
                elif piece.name in ('狮', '虎'):
                    j = self._jump(piece, dc, dr)
                    if j:
                        moves.append(j)
                # other pieces cannot enter river
            else:
                if target is None:
                    moves.append((nc, nr, None))
                elif target.player != piece.player and self.can_capture(piece, target):
                    moves.append((nc, nr, target))
        return moves

    def apply_move(self, piece, nc, nr, captured):
        if captured:
            self.pieces.remove(captured)
        piece.col, piece.row = nc, nr
        enemy = 2 if piece.player == 1 else 1
        if (nc, nr) == DEN[enemy] or not any(p.player == enemy for p in self.pieces):
            self.winner = piece.player
        self.turn = enemy

    def save(self):
        data = {
            "turn": self.turn,
            "pieces": [{"col": p.col, "row": p.row, "player": p.player, "name": p.name}
                       for p in self.pieces],
        }
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls):
        with open(SAVE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        g = cls.__new__(cls)
        g.winner = None
        g.turn = data["turn"]
        g.pieces = [Piece(d["col"], d["row"], d["player"], d["name"]) for d in data["pieces"]]
        return g

    def all_moves(self, player):
        return [(p, m) for p in self.pieces if p.player == player
                for m in self.get_moves(p)]

    def clone(self):
        g = Game.__new__(Game)
        g.pieces = [Piece(p.col, p.row, p.player, p.name) for p in self.pieces]
        g.turn = self.turn
        g.winner = self.winner
        return g

    def display(self):
        R, B, C, X = '\033[91m', '\033[94m', '\033[96m', '\033[0m'

        # Each cell must be exactly 4 terminal columns wide.
        # Chinese char = 2 cols, ASCII char = 1 col.
        def cell(c, r):
            p = self.piece_at(c, r)
            if p:
                color = R if p.player == 1 else B
                label = '红' if p.player == 1 else '蓝'
                return f"{color}{p.name}{label}{X}"   # 2+2 = 4 cols
            if (c, r) == DEN[1]:    return f"{R}穴1 {X}"   # 2+1+1 = 4 cols
            if (c, r) == DEN[2]:    return f"{B}穴2 {X}"
            if (c, r) in TRAPS[1]:  return f"{R}阱1 {X}"   # 2+1+1 = 4 cols
            if (c, r) in TRAPS[2]:  return f"{B}阱2 {X}"
            if (c, r) in RIVER:     return f"{C}~~~~{X}"   # 4 cols
            return " .. "                                   # 4 cols

        sep = "   +" + "----+" * COLS
        # Column header: each slot is 4 cols + 1 separator
        header = "   |" + "|".join(f" {c}  " for c in range(COLS)) + "|"
        print()
        print(sep)
        print(header)
        print(sep)
        for r in range(ROWS):
            print(f" {r} |" + "|".join(cell(c, r) for c in range(COLS)) + "|")
            print(sep)
        print()


# ── MCTS AI ───────────────────────────────────────────────────────────────────
# This is search-based AI, not a trained reinforcement-learning policy.

# Fast board: board[col*9+row] = (player, rank) or None
# piece_locs[player] = list of board indices for that player's pieces

_RIVER_SET = frozenset(c * 9 + r for c in (1, 2, 4, 5) for r in (3, 4, 5))
_DEN_IDX = {1: 3 * 9 + 8, 2: 3 * 9 + 0}
_TRAP_IDX = {
    1: frozenset([2*9+8, 4*9+8, 3*9+7]),
    2: frozenset([2*9+0, 4*9+0, 3*9+1]),
}
_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _fast_from_game(game):
    board = [None] * 63
    locs = {1: [], 2: []}
    for p in game.pieces:
        idx = p.col * 9 + p.row
        board[idx] = (p.player, p.rank)
        locs[p.player].append(idx)
    return board, locs, game.turn


def _fast_moves(board, locs, turn):
    """Get legal moves for the current player in the compact search state.

    Keep these rules consistent with Game.get_moves(). Moves are
    (from_idx, to_idx) pairs using col * 9 + row indices.
    """
    moves = []
    enemy = 2 if turn == 1 else 1
    own_den = _DEN_IDX[turn]
    enemy_traps = _TRAP_IDX[turn]

    for idx in locs[turn]:
        col, row = divmod(idx, 9)
        rank = board[idx][1]
        attack_rank = 0 if idx in _TRAP_IDX[enemy] else rank

        for dc, dr in _DIRS:
            nc, nr = col + dc, row + dr
            if not (0 <= nc < 7 and 0 <= nr < 9):
                continue
            nidx = nc * 9 + nr
            if nidx == own_den:
                continue

            if nidx in _RIVER_SET:
                if rank == 1:  # rat
                    target = board[nidx]
                    if target is None:
                        moves.append((idx, nidx))
                    elif target[0] == enemy:
                        if target[1] == 8 or rank >= target[1]:
                            moves.append((idx, nidx))
                elif rank in (6, 7):  # tiger/lion jump
                    jc, jr = nc, nr
                    jidx = jc * 9 + jr
                    blocked = False
                    while jidx in _RIVER_SET:
                        if board[jidx] is not None:
                            blocked = True; break
                        jc += dc; jr += dr
                        if not (0 <= jc < 7 and 0 <= jr < 9):
                            blocked = True; break
                        jidx = jc * 9 + jr
                    if not blocked and jidx != own_den:
                        target = board[jidx]
                        if target is None:
                            moves.append((idx, jidx))
                        elif target[0] == enemy:
                            t_rank = 0 if jidx in enemy_traps else target[1]
                            if attack_rank >= t_rank:
                                moves.append((idx, jidx))
            else:
                target = board[nidx]
                if target is None:
                    moves.append((idx, nidx))
                elif target[0] == enemy:
                    t_rank = 0 if nidx in enemy_traps else target[1]
                    if idx in _RIVER_SET:
                        pass
                    elif rank == 1 and target[1] == 8:
                        moves.append((idx, nidx))
                    elif rank == 8 and target[1] == 1:
                        pass
                    elif attack_rank >= t_rank:
                        moves.append((idx, nidx))
    return moves


def _fast_simulate(board, locs, turn):
    """Run a heuristic rollout; return winner (1/2), or 0 for a draw.

    Rollout choices are not learned from game data.
    """
    _choice = random.choice
    for step in range(200):
        moves = _fast_moves(board, locs, turn)
        enemy = 2 if turn == 1 else 1
        if not moves:
            return enemy

        enemy_den = _DEN_IDX[enemy]

        # Prefer immediate wins and captures; skip threat analysis for speed.
        best_move = None
        best_score = -1

        for from_idx, to_idx in moves:
            if to_idx == enemy_den:
                best_move = (from_idx, to_idx)
                best_score = 9999
                break
            target = board[to_idx]
            if target is not None:
                score = 50 + target[1] * 5
                if score > best_score:
                    best_score = score
                    best_move = (from_idx, to_idx)

        # 30% of the time, pick random move for exploration
        if best_move is None or (best_score < 100 and step > 0 and random.random() < 0.3):
            best_move = _choice(moves)

        from_idx, to_idx = best_move
        captured = board[to_idx]
        board[to_idx] = board[from_idx]
        board[from_idx] = None
        # Update locs
        pl = locs[turn]
        for i in range(len(pl)):
            if pl[i] == from_idx:
                pl[i] = to_idx; break
        if captured is not None:
            el = locs[enemy]
            for i in range(len(el)):
                if el[i] == to_idx:
                    el.pop(i); break

        if to_idx == _DEN_IDX[enemy] or not locs[enemy]:
            return turn
        turn = enemy
    return 0


_C = 1.41421356


class MCTSNode:
    __slots__ = ('board', 'locs', 'turn', 'winner', 'move',
                 'parent', 'children', 'wins', 'visits', 'untried')

    def __init__(self, board, locs, turn, winner, move=None, parent=None):
        self.board = board
        self.locs = locs
        self.turn = turn
        self.winner = winner
        self.move = move  # (from_idx, to_idx) in the compact board representation
        self.parent = parent
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried = _fast_moves(board, locs, turn) if winner is None else []

    def select_child(self):
        log_parent = math.log(self.visits)
        best = None
        best_val = -1.0
        for c in self.children:
            v = c.visits
            if v == 0:
                return c
            val = c.wins / v + _C * math.sqrt(log_parent / v)
            if val > best_val:
                best_val = val
                best = c
        return best

    def expand(self):
        from_idx, to_idx = self.untried.pop()
        # Clone board and locs
        nb = self.board[:]
        nl = {1: self.locs[1][:], 2: self.locs[2][:]}
        captured = nb[to_idx]
        nb[to_idx] = nb[from_idx]
        nb[from_idx] = None
        # Update locs
        pl = nl[self.turn]
        for i in range(len(pl)):
            if pl[i] == from_idx:
                pl[i] = to_idx; break
        enemy = 2 if self.turn == 1 else 1
        winner = None
        if captured is not None:
            el = nl[enemy]
            for i in range(len(el)):
                if el[i] == to_idx:
                    el.pop(i); break
        if to_idx == _DEN_IDX[enemy] or not nl[enemy]:
            winner = self.turn
        child = MCTSNode(nb, nl, enemy, winner,
                         move=(from_idx, to_idx), parent=self)
        self.children.append(child)
        return child

    def backpropagate(self, winner):
        node = self
        while node is not None:
            node.visits += 1
            pjm = 2 if node.turn == 1 else 1  # player who just moved
            if winner == pjm:
                node.wins += 1.0
            elif winner == 0:
                node.wins += 0.5
            node = node.parent


def _mcts_search(game, time_limit=3.0, verbose=True):
    fb, fl, ft = _fast_from_game(game)
    root = MCTSNode(fb, fl, ft, game.winner)
    end_time = time.time() + time_limit
    iterations = 0

    while time.time() < end_time:
        node = root

        # Select
        while not node.untried and node.children:
            node = node.select_child()

        # Expand
        if node.untried:
            node = node.expand()

        # Simulate
        sb = node.board[:]
        sl = {1: node.locs[1][:], 2: node.locs[2][:]}
        winner = _fast_simulate(sb, sl, node.turn) if node.winner is None else node.winner

        # Backpropagate
        node.backpropagate(winner)
        iterations += 1

    if verbose:
        print(f"  MCTS: {iterations} 次模拟")
    if not root.children:
        return None
    best = max(root.children, key=lambda c: c.visits)
    return best.move


def ai_best_move(game, time_limit=3.0, verbose=True):
    move = _mcts_search(game, time_limit=time_limit, verbose=verbose)
    if move is None:
        return None
    from_idx, to_idx = move
    fc, fr = divmod(from_idx, 9)
    tc, tr = divmod(to_idx, 9)
    return (fc, fr, tc, tr)




# ── CLI helpers ────────────────────────────────────────────────────────────────

def ask_coord(prompt):
    while True:
        try:
            parts = input(prompt).strip().split()
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except (ValueError, EOFError):
            pass
        print("  格式：列 行  (例如: 3 4)")


def main():
    print("=" * 40)
    print("       斗兽棋  Jungle Chess")
    print("  红方(1) 底部出发  蓝方(2) 顶部出发")
    print("=" * 40)

    game = None
    if os.path.exists(SAVE_FILE):
        ans = input("  发现上次存档，是否加载？[y/N]: ").strip().lower()
        if ans == "y":
            game = Game.load()
            print("  已加载上次棋局。")
    if game is None:
        game = Game()

    print("  模式: [1] 双人对战  [2] 人 vs AI  [3] AI vs 人")
    print("         [4] 人 vs PPO 模型")
    mode = input("  选择模式 (默认1): ").strip() or "1"
    ai_player = None
    ppo_model = None
    ppo_predict_move = None
    if mode == "2":
        ai_player = 2
        print("  你执红方，AI 执蓝方")
    elif mode == "3":
        ai_player = 1
        print("  AI 执红方，你执蓝方")
    elif mode == "4":
        human_player = input("  选择你的方位 [1] 红方 [2] 蓝方 (默认1): ").strip() or "1"
        while human_player not in ("1", "2"):
            human_player = input("  请输入 1 或 2: ").strip()
        ai_player = 3 - int(human_player)
        model_path = (
            input("  PPO 模型路径 (默认 models/canonical_10k/best/best_model): ").strip()
            or "models/canonical_10k/best/best_model"
        )
        try:
            from sb3_contrib import MaskablePPO
            from jungle_rl_env import predict_rl_move, validate_model_encoding

            ppo_model = MaskablePPO.load(model_path, device="cpu")
            validate_model_encoding(ppo_model)
            ppo_predict_move = predict_rl_move
            print(f"  已加载 PPO 模型: {model_path}")
        except Exception as exc:
            print(f"  无法加载 PPO 模型: {exc}")
            return
        print(
            f"  你执{'红' if human_player == '1' else '蓝'}方，"
            f"PPO 执{'蓝' if human_player == '1' else '红'}方"
        )
    print("  输入格式：列 行  (例如: 3 4)，输入 q 退出")
    print("=" * 40)

    while game.winner is None:
        game.display()
        player = game.turn
        color  = '\033[91m红\033[0m' if player == 1 else '\033[94m蓝\033[0m'
        print(f">>> {color}方回合")

        all_moves = game.all_moves(player)
        if not all_moves:
            game.winner = 2 if player == 1 else 1
            print("无子可走，对方获胜！")
            break

        # ── AI turn ──
        if player == ai_player:
            print("  PPO 思考中…" if ppo_model is not None else "  AI 思考中…")
            if ppo_model is not None:
                pc, pr, nc, nr = ppo_predict_move(game, player, ppo_model)
            else:
                pc, pr, nc, nr = ai_best_move(game)
            piece    = game.piece_at(pc, pr)
            captured = game.piece_at(nc, nr)
            label = "PPO" if ppo_model is not None else "AI"
            print(f"  {label} 走: {piece} ({pc},{pr}) → ({nc},{nr})"
                  + (f"  吃{captured}" if captured else ""))
            game.apply_move(piece, nc, nr, captured)
            continue

        # ── Human turn ──
        while True:
            raw = input("  选择棋子 (列 行): ").strip()
            if raw.lower() == 'q':
                game.save()
                print("棋局已保存，退出游戏。"); return
            try:
                c, r = map(int, raw.split())
            except ValueError:
                print("  格式：列 行"); continue
            piece = game.piece_at(c, r)
            if piece is None:
                print("  该位置没有棋子"); continue
            if piece.player != player:
                print("  不是你的棋子"); continue
            moves = game.get_moves(piece)
            if not moves:
                print("  该棋子无法移动"); continue
            break

        print(f"  已选: {piece}  可移动到:")
        for i, (nc, nr, cap) in enumerate(moves):
            cap_str = f" 吃{cap}" if cap else ""
            print(f"    [{i}] ({nc},{nr}){cap_str}")

        while True:
            raw = input("  选择目标序号: ").strip()
            if raw.lower() == 'q':
                game.save()
                print("棋局已保存，退出游戏。"); return
            try:
                idx = int(raw)
                if 0 <= idx < len(moves):
                    break
            except ValueError:
                pass
            print("  无效序号")

        nc, nr, captured = moves[idx]
        game.apply_move(piece, nc, nr, captured)

    game.display()
    w = '\033[91m红\033[0m' if game.winner == 1 else '\033[94m蓝\033[0m'
    print(f"游戏结束！{w}方获胜！")
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)


if __name__ == '__main__':
    main()
