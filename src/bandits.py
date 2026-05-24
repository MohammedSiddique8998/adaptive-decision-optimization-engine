from dataclasses import dataclass

import numpy as np


@dataclass
class BanditResult:
    reward: float
    is_optimal_action: bool


class GaussianBandit:
    """Stationary n-armed bandit with Gaussian action values and rewards."""

    def __init__(self, n_arms: int, rng: np.random.Generator, reward_std: float = 1.0):
        if n_arms < 2:
            raise ValueError("n_arms must be at least 2.")
        self.n_arms = int(n_arms)
        self.rng = rng
        self.reward_std = float(reward_std)
        self.true_values = self.rng.normal(loc=0.0, scale=1.0, size=self.n_arms)
        self.optimal_action = int(np.argmax(self.true_values))

    def pull(self, action: int) -> BanditResult:
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Action {action} is outside valid range 0-{self.n_arms - 1}.")
        reward = float(self.rng.normal(loc=self.true_values[action], scale=self.reward_std))
        return BanditResult(reward=reward, is_optimal_action=action == self.optimal_action)
