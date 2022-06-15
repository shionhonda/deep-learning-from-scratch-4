if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from common.types import Policy, State, Action, Reward, Q
from common.gridworld import GridWorld


def greedy_probs(
    Q: Q, state: State, epsilon: float = 0, action_size: int = 4
) -> Policy:
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    base_prob = epsilon / action_size
    action_probs = {
        action: base_prob for action in range(action_size)
    }  # {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += 1 - epsilon
    return action_probs


class MCAgent:
    def __init__(self, epsilon: float = 0.1) -> None:
        self.gamma = 0.9
        self.epsilon = epsilon
        self.alpha = 0.1
        self.action_size = 4

        random_actions = {i: 1 / self.action_size for i in range(self.action_size)}
        self.pi: Policy = defaultdict(lambda: random_actions)
        self.Q: Q = defaultdict(lambda: 0)
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

    def update(self) -> None:
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


if __name__ == "__main__":
    env = GridWorld()
    agent = MCAgent(epsilon=0.1)

    episodes = 10000
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
                agent.update()
                break
            state = next_state

    env.render_q(agent.Q)
