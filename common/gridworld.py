from typing import Iterator

import numpy as np

import common.gridworld_render as render_helper
from common.types import State


class GridWorld:
    def __init__(self) -> None:
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array([[0, 0, 0, 1.0], [0, None, 0, -1.0], [0, 0, 0, 0]])
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self) -> int:
        return len(self.reward_map)

    @property
    def width(self) -> int:
        return len(self.reward_map[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.reward_map.shape

    def actions(self) -> list[int]:
        return self.action_space

    def states(self) -> Iterator[State]:
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state: State, action: int) -> State:
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])

        ny, nx = next_state
        if nx < 0 or nx >= self.width:
            return state
        if ny < 0 or ny >= self.height:
            return state

        return next_state

    def reward(self, state: State, action: int, next_state: State) -> int:
        return self.reward_map[next_state]

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_q(q, print_value)
