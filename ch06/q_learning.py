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


class QLearningAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {i: 1 / self.action_size for i in range(self.action_size)}
        self.pi: Policy = defaultdict(lambda: random_actions)
        self.b: Policy = defaultdict(lambda: random_actions)
        self.Q: Q = defaultdict(lambda: 0)

    def get_action(self, state: State) -> Action:
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state: State, action: Action, reward: Reward, done: bool) -> None:
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        self.pi[state] = greedy_probs(self.Q, state, 0, self.action_size)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

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
