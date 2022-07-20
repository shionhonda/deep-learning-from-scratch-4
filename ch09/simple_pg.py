import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.types_gym import State, Action, Reward


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

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state: State) -> tuple[Action, float]:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.pi.forward(state).unsqueeze(0)
        action: Action = np.random.choice(self.action_size, p=probs.numpy())
        return action, probs[action]
