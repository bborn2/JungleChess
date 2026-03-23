#!/usr/bin/env python3
"""斗兽棋 (Jungle Chess) CLI"""

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
    print("  输入格式：列 行  (例如: 3 4)")
    print("  输入 q 退出")
    print("=" * 40)

    while game.winner is None:
        game.display()
        player = game.turn
        color = '\033[91m红\033[0m' if player == 1 else '\033[94m蓝\033[0m'
        print(f">>> {color}方回合")

        all_moves = game.all_moves(player)
        if not all_moves:
            game.winner = 2 if player == 1 else 1
            print("无子可走，对方获胜！")
            break

        # Select piece
        while True:
            raw = input("  选择棋子 (列 行): ").strip()
            if raw.lower() == 'q':
                print("退出游戏。")
                return
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
