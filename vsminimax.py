from main import JungleChess

if __name__ == "__main__":
            
    # 使用示例
    game = JungleChess()
    while game.isGameOver() == 0:
        move = None

        current_player = game.current_player

        if game.current_player == 1:
            move = game.getHumanInput()  # 获取人类玩家的动作

            
        else:
            print("minimax thinking...", game.current_player)
            _, move = game.minimax2(6, float('-inf'), float('inf'), True)
           
        
        game.current_player = current_player

        print("move ", move)
        b = game.make_move(move)

        if b:
            game.showBoard()
            print(game.evaluate())
            game.current_player *= -1  # 切换玩家

    if game.isGameOver() == -1:
        print("minimax win")
    else:
        print("you win")