import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make("CartPole-v1", new_step_api=True, render_mode="human")
    state = env.reset()
    done = False

    while not done:
        action = np.random.choice([0, 1])
        next_state, reward, done, info, _ = env.step(action)
    env.close()
