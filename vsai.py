import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from main import JungleChess
from dqn import JungleChessEnv
from dqn import DQN



def load_model(model_path, state_dim, action_dim):
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def get_best_move(game, model):
    state = np.array(game.board)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).view(-1)
    q_values = model(state_tensor).detach().numpy().flatten()

    legal_moves = game.get_all_legal_moves(1)
    print(legal_moves    )
    legal_move_actions = [env.tuple_to_action(move) for move in legal_moves]
    print(legal_move_actions    )
    legal_q_values = [q_values[action] for action in legal_move_actions]

    best_action_index = np.argmax(legal_q_values)
    best_action = legal_move_actions[best_action_index]
    best_move = env.action_to_tuple(best_action)
    return best_move

if __name__ == "__main__":
    env = JungleChessEnv()
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    model_path = "model_200.pth"  # Replace with your saved model path
    model = load_model(model_path, state_dim, action_dim)

    game = JungleChess()
    while game.isGameOver() == 0:
        move = None

        current_player = game.current_player

        if game.current_player == 1:
            print("AI thinking...")
            move = get_best_move(game, model)
        else:
            move = game.getHumanInput()  # 获取人类玩家的动作
        
        game.current_player = current_player

        print("Move: ", move)

        if game.make_move(move):
            game.showBoard()
            game.evaluate()
            game.current_player *= -1  # 切换玩家

    if game.isGameOver() == -1:
        print("You win")
    else:
        print("AI wins")
