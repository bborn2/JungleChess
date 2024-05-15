import copy

class JungleChess:
    
    def __init__(self):
        self.board = self.init_board()
        self.current_player = 1  # 1 for AI, -1 for human

        self.lastmove = [-1,-1,-1,-1]
        self.showBoard()

    def init_board(self):
        # 初始化棋盘，返回一个二维数组表示初始状态
        #  case Elephant = 9
        #  case Lion = 8
        #  case Tiger = 7
        #  case Leopard = 6
        #  case Wolf = 5
        #  case Dog = 4
        #  case Cat = 3
        #  case Rat = 2
        #  蓝军 10-99，红军 100-999，地图 0-9，0是空地，1 是 river，2 是 trap，3是上方 den，4 是下方 den

        return [[70,0,2,3,2,0,80],
                [0,30,0,2,0,40,0],
                [90,0,50,0,60,0,20],
                [0,1,1,0,1,1,0],
                [0,1,1,0,1,1,0],
                [0,1,1,0,1,1,0],
                [200,0,600,0,500,0,900],
                [0,400,0,2,0,300,0],
                [800,0,2,4,2,0,700]]

    def get_legal_moves(self, row, col)->list:
        # 获取当前玩家的所有合法动作
        ret = []
        # 上
        if row > 0:
            if self.board[row-1][col] == 0 or self.board[row-1][col] == 2 or \
                self.canAnimalEat(row, col, row-1, col) or self.canAnimalEnterDen(self.board[row][col], row-1, col):
                ret.append((row-1, col))
            elif self.canAnimalCrossRiver(row, col) and self.board[row-1][col] == 1 and \
                self.board[row-2][col] == 1 and self.board[row-3][col] == 1 and \
                    (self.board[row-4][col] == 0 or self.canAnimalEat(row, col, row-4, col)):
                ret.append((row-4, col))
            elif self.canAnimalEnterRiver(row, col) and \
                (self.board[row-1][col] == 1 or self.canAnimalEat(row, col, row-1, col)):
                ret.append((row-1, col))

        # 下
        if row < len(self.board) - 1:
            if self.board[row+1][col] == 0 or self.board[row+1][col] == 2 or \
                self.canAnimalEat(row, col, row+1, col) or self.canAnimalEnterDen(self.board[row][col], row+1, col):
                ret.append((row+1, col))
            elif self.canAnimalCrossRiver(row, col) and self.board[row+1][col] == 1 \
                and self.board[row+2][col] == 1 and self.board[row+3][col] == 1 and \
                    (self.board[row+4][col] == 0 or self.canAnimalEat(row, col, row+4, col)):
                ret.append((row+4, col))
            elif self.canAnimalEnterRiver(row, col) and \
                (self.board[row+1][col] == 1 or self.canAnimalEat(row, col, row+1, col)):
                ret.append((row+1, col))

        # 左
        if col > 0:
            if self.board[row][col-1] == 0 or self.board[row][col-1] == 2 or \
                self.canAnimalEat(row, col, row, col-1) or self.canAnimalEnterDen(self.board[row][col], row, col-1):
                ret.append((row, col-1))
            elif self.canAnimalCrossRiver(row, col) and self.board[row][col-1] == 1 \
                and self.board[row][col-2] == 1 and \
                    (self.board[row][col-3] == 0 or self.canAnimalEat(row, col, row, col-3)):
                ret.append((row, col-3))
            elif self.canAnimalEnterRiver(row, col) and \
                (self.board[row][col-1] == 1 or self.canAnimalEat(row, col, row, col-1)):
                ret.append((row, col-1))
 
        # 右
        if col < len(self.board[0]) - 1:
            if self.board[row][col+1] == 0 or self.board[row][col+1] == 2 or \
                self.canAnimalEat(row, col, row, col+1) or self.canAnimalEnterDen(self.board[row][col], row, col+1):
                ret.append((row, col+1))
            elif self.canAnimalCrossRiver(row, col) and self.board[row][col+1] == 1 and \
                self.board[row][col+2] == 1 and \
                    (self.board[row][col+3] == 0 or self.canAnimalEat(row, col, row, col+3)):
                ret.append((row, col+3))
            elif self.canAnimalEnterRiver(row, col) and \
                (self.board[row][col+1] == 1 or self.canAnimalEat(row, col, row, col+1)):
                ret.append((row, col+1))

        return ret

    def canAnimalEat(self, fromRow, fromCol, toRow, toCol) -> bool:
        
        if self.isSameSide(fromRow, fromCol, toRow, toCol):
            return False

        f = self.board[fromRow][fromCol]
        t = self.board[toRow][toCol]
        
        if f < 10 or t < 10:
            return False
        
        # print("isAnimalCanEat, from \(f), to\(t)")
        
        #如果f和 t 一个在河里，一个在外面，不行
        if self.inRiver(fromRow,fromCol) != self.inRiver(toRow, toCol):
            return False
        
        while f > 10:
            f = f/10
        
        while t > 10:
            t = t/10
         
        if self.isEnterTrap(toRow, toCol):
            return True
        elif f == 2 and t == 9:
            return True
        elif f == 9 and t == 2:
            return False
        elif f >= t:
            return True
        
        return False

    # 这两个位置是不是同一个边
    def isSameSide(self, row1, col1, row2, col2)->bool:
        
        # 如果某个位置没有子
        if self.board[row1][col1] < 10 or self.board[row2][col2] < 10:
            return False

        return   (self.board[row1][col1] > 100) == (self.board[row2][col2] > 100)
    
    # 判断这个位置是否是河
    def inRiver(self, row, col)->bool:
        return self.board[row][col] % 10 == 1
    
    # 判断是否能进河，就是判断是否是 rat
    def canAnimalEnterRiver(self, row, col) -> bool:
        
        v = self.board[row][col]
        
        while v > 10:
            v = v / 10

        if v == 2:
            return True
        
        return False
    
    #判断当前是否在 trap 里
    def isEnterTrap(self, row, col)->bool:
        return self.board[row][col] % 10 == 2
    
    # 判断是否能过河，就是判断是否是 lion 或者 tiger
    def canAnimalCrossRiver(self, row, col) -> bool:
        
        v = self.board[row][col]
        
        while v > 10:
            v = v / 10

        if v == 7 or v == 8:
            return True
            
        return False
    
    # 判断能否走进 den，只能进入对方的 den
    def canAnimalEnterDen(self, pieces, row, col)->bool:
        if pieces >= 100 and self.board[row][col] % 10 == 3:
            return True
        elif pieces >= 10 and pieces < 100 and self.board[row][col] % 10 == 4:
            return True

        return False

    def make_move(self, move)->bool:
        # 执行动作并返回新的状态

        if len(move) != 4:
            print("false 1")
            return False

        role = "me"
        if self.current_player == 1:
            role = "ai"

        fromRow = move[0]
        fromCol = move[1]
        toRow = move[2]
        toCol = move[3]

        # print("## %s : %d,%d move to %d,%d"%(role, fromRow, fromCol, toRow, toCol))

        if fromRow >= len(self.board) or toRow >= len(self.board) or \
            fromCol >= len(self.board[0]) or toCol >= len(self.board[0]):
            print("false 2")
            return False

        moves = self.get_legal_moves(fromRow, fromCol)

        if (self.current_player == 1 and self.board[fromRow][fromCol] > 10 and self.board[fromRow][fromCol] < 100) or\
        (self.current_player == -1 and self.board[fromRow][fromCol] > 100):
            if (toRow, toCol) in moves:

                self.board[toRow][toCol] = self.board[fromRow][fromCol] // 10 * 10 +  self.board[toRow][toCol] % 10
                self.board[fromRow][fromCol] = self.board[fromRow][fromCol] % 10

                self.lastmove = move

                # print("true")
                return True

        print("false", move, moves) 
        return False
    
    def showBoard(self):

        fromRow = self.lastmove[0]
        fromCol = self.lastmove[1]
        toRow = self.lastmove[2]
        toCol = self.lastmove[3]
        
        print("  \t0\t1\t2\t3\t4\t5\t6")
        print("---------------------------------------------------------------")
        for i in range(len(self.board)):
            s = str(i) + ":\t"
            for j in range(len(self.board[i])):

                if i == fromRow and j == fromCol:
                    s += "\033[43m"+str(self.board[i][j])+"\033[0m"
                elif i == toRow and j == toCol:
                    s += "\033[42m"+str(self.board[i][j])+"\033[0m"
                else:
                    s += str(self.board[i][j])
                s += "\t"

            print(s)
        print("===============================================================")


    def evaluate(self):
        # 评价给定状态
        # 返回一个整数值表示状态的好坏
        score = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] > 10 and self.board[i][j] < 100:
                    score = score + (self.board[i][j] // 10) * (self.board[i][j] // 10) + i * 5

                    if i == 8 and j == 3:
                        score += 10000

                elif self.board[i][j] > 100:
                    score = score - (self.board[i][j] // 100) * (self.board[i][j] // 100) - (9-i)*5

                    if i == 0 and j == 3:
                        score -= 10000
        

        # print("blue score = ", score)
        return score

    def isGameOver(self):
        # 判断游戏是否结束

        blue = False
        red = False

        for i in self.board:
            for j in i:
                if j > 100:
                    red = True
                elif j > 10 and j < 100:
                    blue = True

        if red == False or self.board[8][3] > 10:
            return -1000
        elif blue == False or self.board[0][3] > 10:
            return 1000
        else:
            return 0

    def minimax(self, depth, alpha, beta, maximizingPlayer):
        if depth == 0 or self.isGameOver():
            return self.evaluate()
        
        if maximizingPlayer:
            maxEval = float('-inf')

            for i in range(len(self.board)):
                for j in range(len(self.board[i])):
                    if self.board[i][j] > 10 and self.board[i][j]< 100:
                        for move in self.get_legal_moves(i, j):
                            tempboard = copy.deepcopy(self.board)
                            tempmove = [i, j]
                            tempmove.extend(move)
                            self.current_player = 1

                            b = self.make_move(tempmove)

                            if b:
                                eval = self.minimax(depth - 1, alpha, beta, False)
                                maxEval = max(maxEval, eval)
                                alpha = max(alpha, eval)

                                self.board = copy.deepcopy(tempboard)

                                if beta <= alpha:
                                    break
            return maxEval
        else:
            minEval = float('inf')
            
            for i in range(len(self.board)):
                for j in range(len(self.board[i])):
                    if self.board[i][j] > 100:
                        for move in self.get_legal_moves(i, j):
                            tempboard = copy.deepcopy(self.board)
                            tempmove = [i, j]
                            tempmove.extend(move)
                            self.current_player = -1

                            b = self.make_move(tempmove)

                            if b:
                                eval = self.minimax(depth - 1, alpha, beta, True)
                                minEval = min(minEval, eval)
                                beta = min(beta, eval)

                                self.board = copy.deepcopy(tempboard)

                                if beta <= alpha:
                                    break
            return minEval

    def getBestMove(self, depth):
        best_move = None
        best_value = float('-inf')

        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] > 10 and self.board[i][j]< 100:
                    for move in self.get_legal_moves(i, j):              
                                                
                        tempboard = copy.deepcopy(self.board)
                        tempmove = [i, j]
                        tempmove.extend(move)

                        self.current_player = 1
                        b = self.make_move(tempmove)
                        # game.showBoard()

                        move_value = self.minimax(depth - 1, float('-inf'), float('inf'), False)

                        # return
                        #需要复原
                        self.board = copy.deepcopy(tempboard)

                        if move_value > best_value:
                            best_value = move_value
                            best_move = tempmove

        return best_move
    
    def getHumanInput(self):
        while True:
            number = input("Enter from row, from col, to row, to col:").replace(" ", "")
            inputs =  [int(digit) for digit in str(number)]
            if len(inputs) == 4:
                return inputs
        
# 使用示例
game = JungleChess()
while game.isGameOver() == 0:
    move = None

    current_player = game.current_player

    if game.current_player == 1:
        print("ai thinking...")
        move = game.getBestMove(depth=5)
    else:
        move = game.getHumanInput()  # 获取人类玩家的动作
    
    game.current_player = current_player

    print("move ", move)
    b = game.make_move(move)

    if b:
        game.showBoard()
        game.evaluate()
        game.current_player *= -1  # 切换玩家

if game.isGameOver() == 1000:
    print("you win")
else:
    print("ai win")