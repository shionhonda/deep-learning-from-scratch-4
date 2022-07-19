if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from common.types_gym import State, Action, Reward


class QNet(nn.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(in_size=4, out_size=self.action_size)
        self.qnet_target = QNet(in_size=4, out_size=self.action_size)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

    def sync_target(self) -> None:
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state: State) -> Action:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                qs = self.qnet.forward(state)
            return qs.argmax().item()

    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.LongTensor(done)

        qs = self.qnet.forward(state)
        q = qs[np.arange(self.batch_size), action]

        with torch.no_grad():
            next_qs = self.qnet_target.forward(next_state)

        next_q, _ = torch.max(next_qs, dim=1)
        target = reward + (1 - done) * self.gamma * next_q
        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    episodes = 300
    sync_interval = 20
    env = gym.make("CartPole-v1", new_step_api=True)
    agent = DQNAgent()
    reward_history = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_target()

        reward_history.append(total_reward)

    env.close()

    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    # Greedy
    agent.epsilon = 0
    env = gym.make("CartPole-v1", new_step_api=True, render_mode="human")
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    print(f"Total Reward: {total_reward}")

    env.close()
