if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from common.types import Policy, State, Value, Action, Reward
from common.gridworld import GridWorld


class RandomAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {i: 1 / self.action_size for i in range(self.action_size)}
        self.pi: Policy = defaultdict(lambda: random_actions)
        self.V: Value = defaultdict(lambda: 0)
        self.cnts: dict[State, int] = defaultdict(lambda: 0)
        self.memory: list[tuple[State, Action, Reward]] = []

    def get_action(self, state: State) -> Action:
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state: State, action: Action, reward: Reward) -> None:
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def eval(self) -> None:
        G = 0
        for data in reversed(self.memory):
            state, _, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]


if __name__ == "__main__":
    env = GridWorld()
    agent = RandomAgent()

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            # NOTE: Added to avoid None exception
            if reward is None:
                continue
            agent.add(state, action, reward)
            if done:
                agent.eval()
                break
            state = next_state

    env.render_v(agent.V, agent.pi)
