if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.types import Policy, State, Value, Action, Reward
from common.gridworld import GridWorld
from tqdm import tqdm

HEIGHT, WIDTH = 4, 3
ACTION_SIZE = 4


def one_hot(state: State) -> torch.Tensor:
    vec = torch.zeros(HEIGHT * WIDTH, dtype=torch.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec.unsqueeze(0)


class QNet(nn.Module):
    def __init__(self, d_in: int = 12, d_hidden: int = 100) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, ACTION_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet(d_in=HEIGHT * WIDTH)
        self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state: State) -> Action:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet.forward(state)
            return qs.argmax().item()

    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        done: bool,
    ) -> float:
        if done:
            next_q = torch.zeros(1)  # [0.]
        else:
            with torch.no_grad():
                next_qs = self.qnet.forward(next_state)
            _, next_q = next_qs.max(dim=1)

        target = self.gamma * next_q + reward
        qs = self.qnet.forward(state)
        q = qs[:, action]
        loss = F.mse_loss(target, q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = GridWorld()
    agent = QLearningAgent()

    episodes = 200
    loss_history = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        state = one_hot(state)
        total_loss, cnt = 0.0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)

            # NOTE: Added to avoid None exception
            if reward is None:
                continue

            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state

        average_loss = total_loss / cnt
        loss_history.append(average_loss)

    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.show()

    # visualize
    Q = {}
    for state in env.states():
        for action in env.action_space:
            q = agent.qnet(one_hot(state))[:, action]
            Q[state, action] = float(q.data)
    env.render_q(Q)
