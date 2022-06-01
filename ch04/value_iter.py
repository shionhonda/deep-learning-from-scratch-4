if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict

from ch04.policy_iter import greedy_policy
from common.gridworld import GridWorld
from common.types import Value


def value_iter_onestep(V: Value, env: GridWorld, gamma: float) -> Value:
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values: list[int] = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # NOTE: Added to avoid None exception
            if r is None:
                continue
            value = r + gamma * V[next_state]
            action_values.append(value)

        # deterministic policy
        V[state] = max(action_values)
    return V


def value_iter(
    env: GridWorld, gamma: float, threshold: float = 0.001, is_render: bool = False
) -> Value:
    V: Value = defaultdict(lambda: 0)

    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

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
    V = value_iter(env, gamma, is_render=True)
    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
