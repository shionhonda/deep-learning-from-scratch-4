if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from common.types import Policy, State, Action, Reward, Value
from common.gridworld import GridWorld


class TDAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {i: 1 / self.action_size for i in range(self.action_size)}
        self.pi: Policy = defaultdict(lambda: random_actions)
        self.V: Value = defaultdict(lambda: 0)

    def get_action(self, state: State) -> Action:
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def eval(self, state: State, reward: Reward, next_state: State, done: bool) -> None:
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V
        self.V[state] += (target - self.V[state]) * self.alpha


if __name__ == "__main__":
    env = GridWorld()
    agent = TDAgent()

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            # NOTE: Added to avoid None exception
            if reward is None:
                continue
            agent.eval(state, reward, next_state, done)
            if done:
                break
            state = next_state

    env.render_v(agent.V)
