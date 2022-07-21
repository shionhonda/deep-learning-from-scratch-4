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


class PolicyNet(nn.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_size, 128)
        self.l2 = nn.Linear(128, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):
    def __init__(self, in_size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_size, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self) -> None:
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet(in_size=4, out_size=self.action_size)
        self.v = ValueNet(in_size=4)
        self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = torch.optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state: State) -> tuple[Action, Prob]:
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.pi.forward(state).squeeze(0)
        action: Action = np.random.choice(self.action_size, p=probs.detach().numpy())
        return action, probs[action]

    def update(
        self,
        state: State,
        action_prob: Prob,
        reward: Reward,
        next_state: State,
        done: bool,
    ) -> None:
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        with torch.no_grad():
            target = reward + self.gamma * self.v.forward(next_state) * (1 - int(done))
        v = self.v.forward(state)
        loss_v = F.mse_loss(v, target)

        delta = (target - v).detach()
        loss_pi = -delta * torch.log(action_prob)

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()


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
            done = terminated or truncated
            agent.update(state, prob, reward, next_state, done)
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

    env.close()

    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    # Greedy
    env = gym.make("CartPole-v1", new_step_api=True, render_mode="human")
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    print(f"Total Reward: {total_reward}")

    env.close()
