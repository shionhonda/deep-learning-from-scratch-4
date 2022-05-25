if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    from bandit import Bandit, Agent

    sns.set()

    runs = 200
    steps = 1000
    epsilon = 0.1
    all_rates = np.zeros((runs, steps))

    for run in tqdm(range(runs)):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))
        all_rates[run] = rates

    avg_rates = np.mean(all_rates, axis=0)
    std_rates = np.std(all_rates, axis=0)

    plt.plot(avg_rates)
    plt.fill_between(
        range(steps), avg_rates - std_rates, avg_rates + std_rates, alpha=0.2
    )
    plt.xlabel("Steps")
    plt.ylabel("Rates")
    plt.show()
