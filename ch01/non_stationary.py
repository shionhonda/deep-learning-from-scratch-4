import numpy as np


class NonStatBandit:
    def __init__(self, arms: int = 10) -> None:
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm: int) -> int:
        rate = self.rates[arm]
        # change probability distribution of arms
        self.rates += 0.1 * np.random.randn(self.arms)
        if np.random.rand() < rate:
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, epsilon: float, alpha: float, action_size: int = 10) -> None:
        """Initialize agent.

        Args:
            epsilon (float): Probability of exploration
            alpha (float): Truncation rate. Must be in [0, 1]
            action_size (int, optional): Size of discrete actions. Defaults to 10.
        """
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.alpha = alpha

    def update(self, action: int, reward: float) -> None:
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

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

    def play(agent: AlphaAgent, bandit: NonStatBandit) -> list[float]:
        total_reward = 0
        rates = []
        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))
        return rates

    runs = 200
    steps = 1000
    epsilon = 0.1
    all_rates_0, all_rates_1 = np.zeros((runs, steps)), np.zeros((runs, steps))

    for run in tqdm(range(runs)):
        bandit = NonStatBandit()

        agent_0 = AlphaAgent(epsilon, alpha=1)
        rates_0 = play(agent_0, bandit)
        all_rates_0[run] = rates_0

        agent_1 = AlphaAgent(epsilon, alpha=0.8)
        rates_1 = play(agent_1, bandit)
        all_rates_1[run] = rates_1

    avg_rates_0 = np.mean(all_rates_0, axis=0)
    std_rates_0 = np.std(all_rates_0, axis=0)
    plt.plot(avg_rates_0, label="alpha=1")
    plt.fill_between(
        range(steps), avg_rates_0 - std_rates_0, avg_rates_0 + std_rates_0, alpha=0.2
    )

    avg_rates_1 = np.mean(all_rates_1, axis=0)
    std_rates_1 = np.std(all_rates_1, axis=0)
    plt.plot(avg_rates_1, label="alpha=0.8")
    plt.fill_between(
        range(steps), avg_rates_1 - std_rates_1, avg_rates_1 + std_rates_1, alpha=0.2
    )

    plt.xlabel("Steps")
    plt.ylabel("Rates")
    plt.legend()
    plt.show()
