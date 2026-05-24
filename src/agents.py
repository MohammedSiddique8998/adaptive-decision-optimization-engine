from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgentConfig:
    name: str
    strategy: str
    epsilon: float = 0.0
    confidence: float = 2.0


class BanditAgent:
    def __init__(self, n_arms: int, config: AgentConfig, rng: np.random.Generator):
        self.n_arms = int(n_arms)
        self.config = config
        self.rng = rng
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.action_counts = np.zeros(self.n_arms, dtype=int)
        self.step_count = 0

    def select_action(self) -> int:
        strategy = self.config.strategy
        if strategy == "random":
            return int(self.rng.integers(self.n_arms))
        if strategy == "epsilon_greedy":
            if self.rng.random() < self.config.epsilon:
                return int(self.rng.integers(self.n_arms))
            return self._argmax_random_tie(self.estimates)
        if strategy == "ucb":
            untried = np.flatnonzero(self.action_counts == 0)
            if len(untried) > 0:
                return int(self.rng.choice(untried))
            bonus = self.config.confidence * np.sqrt(np.log(self.step_count + 1) / self.action_counts)
            return self._argmax_random_tie(self.estimates + bonus)
        raise ValueError(f"Unknown strategy: {strategy}")

    def update(self, action: int, reward: float) -> None:
        self.step_count += 1
        self.action_counts[action] += 1
        count = self.action_counts[action]
        self.estimates[action] += (reward - self.estimates[action]) / count

    def _argmax_random_tie(self, values: np.ndarray) -> int:
        best_value = np.max(values)
        best_actions = np.flatnonzero(values == best_value)
        return int(self.rng.choice(best_actions))


def default_agent_configs() -> list[AgentConfig]:
    return [
        AgentConfig(name="Random", strategy="random"),
        AgentConfig(name="Greedy", strategy="epsilon_greedy", epsilon=0.0),
        AgentConfig(name="Epsilon 0.01", strategy="epsilon_greedy", epsilon=0.01),
        AgentConfig(name="Epsilon 0.10", strategy="epsilon_greedy", epsilon=0.10),
        AgentConfig(name="UCB c=2", strategy="ucb", confidence=2.0),
    ]
