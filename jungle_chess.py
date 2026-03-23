#!/usr/bin/env python3
"""斗兽棋 (Jungle Chess) CLI"""
import math

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

    def all_moves(self, player):
        return [(p, m) for p in self.pieces if p.player == player
                for m in self.get_moves(p)]

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


# ── Minimax + Alpha-Beta ───────────────────────────────────────────────────────

_PIECE_VAL = {8: 900, 7: 500, 6: 400, 5: 300, 4: 200, 3: 150, 2: 100, 1: 80}

# Transposition table: hash -> (depth, value, flag)  flag: 0=exact 1=lower 2=upper
_TT: dict = {}
_TT_MAX = 200_000


def _board_hash(game):
    return hash(tuple(sorted((p.player, p.name, p.col, p.row) for p in game.pieces))
                + (game.turn,))


def _evaluate(game, player):
    enemy = 2 if player == 1 else 1

    # Build set of squares the enemy threatens next move
    enemy_attacks: set = set()
    for p in game.pieces:
        if p.player == enemy:
            for nc, nr, _ in game.get_moves(p):
                enemy_attacks.add((nc, nr))

    score = 0
    for p in game.pieces:
        val = _PIECE_VAL[p.rank]
        e_den = DEN[enemy if p.player == player else player]
        dist  = abs(p.col - e_den[0]) + abs(p.row - e_den[1])
        pos   = (12 - dist) * 4

        # Trap bonus: enemy piece sitting in our trap is nearly dead
        if p.player == enemy and p.pos() in TRAPS[player]:
            val = 0   # effectively captured already

        if p.player == player:
            total = val + pos
            # Penalty if this piece is under attack
            if p.pos() in enemy_attacks:
                total -= val // 2
            score += total
        else:
            score -= val + pos

    return score


def _move_priority(p, nc, nr, cap, ai_player):
    """Higher = search first (for move ordering)."""
    enemy = 2 if p.player == 1 else 1
    if (nc, nr) == DEN[enemy]:
        return 100_000                          # winning move
    if cap:
        return 1000 + cap.rank * 10 - p.rank   # MVV-LVA
    # Advance toward enemy den
    den = DEN[enemy]
    return -(abs(nc - den[0]) + abs(nr - den[1]))


def _minimax(game, depth, alpha, beta, maximizing, ai_player):
    if game.winner == ai_player:
        return 100_000 + depth, None
    if game.winner is not None:
        return -100_000 - depth, None
    if depth == 0:
        return _evaluate(game, ai_player), None

    # Transposition table lookup
    h = _board_hash(game)
    tt = _TT.get(h)
    if tt and tt[0] >= depth:
        td, tv, tf = tt
        if tf == 0:
            return tv, None
        if tf == 1 and tv > alpha:
            alpha = tv
        if tf == 2 and tv < beta:
            beta = tv
        if alpha >= beta:
            return tv, None

    player = game.turn
    moves  = game.all_moves(player)
    if not moves:
        game.winner = 2 if player == 1 else 1
        result = _minimax(game, 0, alpha, beta, maximizing, ai_player)
        game.winner = None
        return result

    # Move ordering
    moves.sort(key=lambda pm: _move_priority(pm[0], pm[1][0], pm[1][1], pm[1][2], ai_player),
               reverse=True)

    orig_alpha = alpha
    best_move  = None

    if maximizing:
        best = -math.inf
        for p, (nc, nr, cap) in moves:
            pc, pr = p.col, p.row
            if cap: game.pieces.remove(cap)
            p.col, p.row = nc, nr
            prev_winner, prev_turn = game.winner, game.turn
            e = 2 if p.player == 1 else 1
            if (nc, nr) == DEN[e] or not any(x.player == e for x in game.pieces):
                game.winner = p.player
            game.turn = e
            val, _ = _minimax(game, depth - 1, alpha, beta, False, ai_player)
            p.col, p.row = pc, pr
            if cap: game.pieces.append(cap)
            game.winner, game.turn = prev_winner, prev_turn
            if val > best:
                best, best_move = val, (pc, pr, nc, nr)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
    else:
        best = math.inf
        for p, (nc, nr, cap) in moves:
            pc, pr = p.col, p.row
            if cap: game.pieces.remove(cap)
            p.col, p.row = nc, nr
            prev_winner, prev_turn = game.winner, game.turn
            e = 2 if p.player == 1 else 1
            if (nc, nr) == DEN[e] or not any(x.player == e for x in game.pieces):
                game.winner = p.player
            game.turn = e
            val, _ = _minimax(game, depth - 1, alpha, beta, True, ai_player)
            p.col, p.row = pc, pr
            if cap: game.pieces.append(cap)
            game.winner, game.turn = prev_winner, prev_turn
            if val < best:
                best, best_move = val, (pc, pr, nc, nr)
            beta = min(beta, best)
            if beta <= alpha:
                break

    # Store in transposition table
    if len(_TT) < _TT_MAX:
        flag = 0 if orig_alpha < best < beta else (1 if best >= beta else 2)
        _TT[h] = (depth, best, flag)

    return best, best_move


def ai_best_move(game, depth=7):
    _TT.clear()
    ai_player = game.turn
    _, move = _minimax(game, depth, -math.inf, math.inf, True, ai_player)
    return move




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
    game = Game()
    print("=" * 40)
    print("       斗兽棋  Jungle Chess")
    print("  红方(1) 底部出发  蓝方(2) 顶部出发")
    print("=" * 40)
    print("  模式: [1] 双人对战  [2] 人 vs AI  [3] AI vs 人")
    mode = input("  选择模式 (默认1): ").strip() or "1"
    ai_player = None
    if mode == "2":
        ai_player = 2
        print("  你执红方，AI 执蓝方")
    elif mode == "3":
        ai_player = 1
        print("  AI 执红方，你执蓝方")
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
            print("  AI 思考中…")
            pc, pr, nc, nr = ai_best_move(game)
            piece    = game.piece_at(pc, pr)
            captured = game.piece_at(nc, nr)
            print(f"  AI 走: {piece} ({pc},{pr}) → ({nc},{nr})"
                  + (f"  吃{captured}" if captured else ""))
            game.apply_move(piece, nc, nr, captured)
            continue

        # ── Human turn ──
        while True:
            raw = input("  选择棋子 (列 行): ").strip()
            if raw.lower() == 'q':
                print("退出游戏。"); return
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
                print("退出游戏。"); return
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


if __name__ == '__main__':
    main()
