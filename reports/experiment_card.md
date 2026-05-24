# Experiment Card

## Project

Bio-Inspired Multi-Armed Bandit Reinforcement Learning

## Intended Use

Educational and portfolio demonstration of reinforcement learning fundamentals, especially exploration versus exploitation.

## Environment

Stationary Gaussian n-armed bandit:

- true action values sampled once per run from `N(0, 1)`,
- noisy rewards sampled from `N(q*(a), 1)`,
- independent runs averaged to reduce noise.

## Policies

- Random baseline
- Greedy action-value selection
- Epsilon-greedy with epsilon 0.01
- Epsilon-greedy with epsilon 0.10
- Upper Confidence Bound with confidence value 2.0

## Metrics

- Average reward over time
- Optimal-action rate over time
- Final-window average reward
- Final-window optimal-action rate

## Experiment Settings

- Seed: 42
- Arms: 5, 10, 20
- Runs: 500
- Steps: 1,000
- Final evaluation window: 100 steps

## Key Result

UCB c=2 produced the best final-window average reward in the generated experiment outputs for 5, 10 and 20 arms.

## Limitations

- Simulated stationary environment.
- No context features.
- No non-stationary drift.
- Limited hyperparameter search.
- Results can vary with seed, run count and reward assumptions.

## Future Work

- Add Thompson sampling.
- Add regret plots.
- Add non-stationary testbeds.
- Add contextual bandit setting.
- Add confidence intervals for policies.
