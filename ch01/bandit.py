import numpy as np


class Bandit:
    def __init__(self, arms: int = 10):
        self.rates = np.random.rand(arms)

    def play(self, arm: int):
        rate = self.rates[arm]
        if np.random.rand() < rate:
            return 1
        else:
            return 0


if __name__ == "__main__":
    bandit = Bandit()
    for _ in range(3):
        print(bandit.play(0))
