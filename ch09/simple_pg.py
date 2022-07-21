if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.types_gym import State, Action, Reward

Prob = torch.Tensor


class Policy(nn.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_size, 128)
        self.l2 = nn.Linear(128, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    def __init__(self) -> None:
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory: list[tuple[Reward, torch.Tensor]] = []
        self.pi = Policy(in_size=4, out_size=self.action_size)
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state: State) -> tuple[Action, Prob]:
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.pi.forward(state).squeeze(0)

        action: Action = np.random.choice(self.action_size, p=probs.detach().numpy())
        return action, probs[action]

    def add(self, reward: Reward, prob: Prob) -> None:
        data = (reward, prob)
        self.memory.append(data)

    def update(self) -> None:
        G, loss = 0, 0
        for reward, _ in reversed(self.memory):
            G = self.gamma * G + reward
        for _, prob in self.memory:
            loss += -G * torch.log(prob)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


if __name__ == "__main__":
    import gym
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    episodes = 3000
    env = gym.make("CartPole-v1", new_step_api=True)
    agent = Agent()
    reward_history = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.add(reward, prob)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        agent.update()
        reward_history.append(total_reward)

    env.close()

    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
