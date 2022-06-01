from collections import defaultdict
from typing import Hashable

from policy_eval import policy_eval

from common.gridworld import GridWorld
from common.types import Policy, Value


def argmax(d: dict[Hashable, float]) -> Hashable:
    return max(d.items(), key=lambda x: x[1])[0]


def greedy_policy(V: Value, env: GridWorld, gamma: float) -> Policy:
    pi: Policy = {}
    for state in env.states():
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # NOTE: Added to avoid None exception
            if r is None:
                continue
            value = r + gamma * V[next_state]
            action_values[action] = value
        # deterministic policy
        max_action = argmax(action_values)
        action_probs = {k: 0 for k in env.actions()}
        action_probs[max_action] = 1
        pi[state] = action_probs
    return pi


def policy_iter(
    env: GridWorld, gamma: float, threshold: float = 0.001, is_render: bool = False
) -> Policy:
    pi: Policy = defaultdict(lambda: {k: 0.25 for k in env.actions()})
    V: Value = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)
        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break

        pi = new_pi

    return pi


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)
