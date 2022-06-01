if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict

from common.gridworld import GridWorld
from common.types import Policy, State, Value


def eval_onestep(pi: Policy, V: Value, env: GridWorld, gamma: float = 0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0

        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # NOTE: Added to avoid None exception
            if r is None:
                continue
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(
    pi: Policy, V: Value, env: GridWorld, gamma: float, threshold: float = 0.001
) -> Value:
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            delta = max(delta, t)

        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9

    pi: Policy = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V: Value = defaultdict(lambda: 0)
    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)
