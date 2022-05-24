import numpy as np


class Bandit:
    def __init__(self, arms: int = 10) -> None:
        self.rates = np.random.rand(arms)

    def play(self, arm: int) -> int:
        rate = self.rates[arm]
        if np.random.rand() < rate:
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon: float, action_size: int = 10) -> None:
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action: int, reward: float) -> None:
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    sns.set()

    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in tqdm(range(steps)):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    print(f"{total_reward=}")

    plt.plot(total_rewards)
    plt.xlabel("Steps")
    plt.ylabel("Total reward")
    plt.show()

    plt.plot(rates)
    plt.xlabel("Steps")
    plt.ylabel("Rates")
    plt.show()
