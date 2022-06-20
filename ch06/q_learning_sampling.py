if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from common.types import State, Action, Reward, Q
from common.gridworld import GridWorld


class QLearningSamplingAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {i: 1 / self.action_size for i in range(self.action_size)}
        self.Q: Q = defaultdict(lambda: 0)

    def get_action(self, state: State) -> Action:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[(state, action)] for action in range(self.action_size)]
            return np.argmax(qs)

    def update(self, state: State, action: Action, reward: Reward, done: bool) -> None:
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningSamplingAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            # NOTE: Added to avoid None exception
            if reward is None:
                continue
            agent.update(state, action, reward, done)

            if done:
                break
            state = next_state

    env.render_q(agent.Q)
