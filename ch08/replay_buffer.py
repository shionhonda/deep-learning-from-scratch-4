if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import deque
import random
from typing import Deque, Any
import numpy as np

from common.types_gym import State, Action, Reward, Data

BatchData = Any


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        self.buffer: Deque[Data] = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        done: bool,
    ) -> None:
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self) -> int:
        return len(self.buffer)

    def get_batch(self) -> tuple[Any, Any, Any, Any, Any]:
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.stack([x[1] for x in data])
        reward = np.stack([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.stack([x[4] for x in data])
        return state, action, reward, next_state, done


if __name__ == "__main__":
    import gym

    env = gym.make("CartPole-v1", new_step_api=True, render_mode="human")
    replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

    for episode in range(10):
        state = env.reset()
        done = False

        while not done:
            action = np.random.choice([0, 1])
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
    env.close()

    state, action, reward, next_state, done = replay_buffer.get_batch()
    print(state.shape)
    print(action.shape)
    print(reward.shape)
    print(next_state.shape)
    print(done.shape)
