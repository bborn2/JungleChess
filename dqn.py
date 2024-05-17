import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from main import JungleChess

class JungleChessEnv(gym.Env):
    def __init__(self):
        super(JungleChessEnv, self).__init__()
        self.game = JungleChess()
        self.action_space = spaces.Discrete(9 * 7 * 9 * 7)  # from_row, from_col, to_row, to_col
        self.observation_space = spaces.Box(low=0, high=1000, shape=(9, 7), dtype=int)

    def reset(self):
        self.game = JungleChess()
        return self.game.board

    def step(self, action):
        move = self.action_to_tuple(action)
        reward = -10
        done = False
        self.game.current_player = 1

        print("step, ", move)

        if self.game.make_move(move):
            reward = self.game.evaluate()
            if self.game.isGameOver() != 0:
                done = True
                reward += self.game.isGameOver()
            else:
                # Opponent move using minimax2
                self.game.current_player = -1
                _, opponent_move = self.game.minimax2(6, float('-inf'), float('inf'), True)
                self.game.make_move(opponent_move)
                reward -= self.game.evaluate()
                if self.game.isGameOver() != 0:
                    done = True
                    reward -= self.game.isGameOver()

            
        else:
            reward = -10  # Illegal move penalty

        return np.array(self.game.board), reward, done, {}

    def action_to_tuple(self, action):
        from_row = action // (9 * 7 * 9 * 7)
        from_col = (action % (9 * 7 * 9 * 7)) // (9 * 7)
        to_row = (action % (9 * 7)) // 7
        to_col = action % 7
        return [from_row, from_col, to_row, to_col]

    def render(self, mode='human'):
        self.game.showBoard()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon, game):
        legal_moves = game.get_all_legal_moves(1)  # AI's legal moves
        if random.random() < epsilon:
            # Randomly select a legal move
            action = random.choice(legal_moves)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor).detach().numpy().flatten()

            # Filter Q-values to only consider legal moves
            legal_move_actions = [self.tuple_to_action(move) for move in legal_moves]
            legal_q_values = [q_values[action] for action in legal_move_actions]

            # Select the action with the highest Q-value among legal moves
            best_action_index = np.argmax(legal_q_values)
            action = legal_move_actions[best_action_index]

        return self.tuple_to_action(action)  # Return action as tuple

    def tuple_to_action(self, move):
        from_row, from_col, to_row, to_col = move
        return (from_row * 9 * 7 * 9 * 7) + (from_col * 9 * 7) + (to_row * 7) + to_col

    def train(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + gamma * next_q_value * (1 - dones)

        loss = self.loss_fn(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_dqn(env, agent, episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    epsilon = epsilon_start
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state, epsilon, env.game)
            next_state, reward, done, _ = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            agent.train(batch_size, gamma)

        agent.update_target()
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

if __name__ == "__main__":
    env = JungleChessEnv()
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    train_dqn(env, agent, episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
