import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from agents import AgentConfig, BanditAgent, default_agent_configs
from bandits import GaussianBandit
from plotting import save_line_chart


@dataclass(frozen=True)
class ExperimentConfig:
    n_arms: tuple[int, ...] = (5, 10, 20)
    steps: int = 1000
    runs: int = 500
    seed: int = 42
    reward_std: float = 1.0
    output_dir: Path = Path("results")


def run_policy(
    n_arms: int,
    steps: int,
    runs: int,
    agent_config: AgentConfig,
    seed: int,
    reward_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    average_rewards = np.zeros(steps, dtype=float)
    optimal_action_rate = np.zeros(steps, dtype=float)
    master_rng = np.random.default_rng(seed)

    for _ in range(runs):
        run_seed = int(master_rng.integers(0, 2**32 - 1))
        bandit_rng = np.random.default_rng(run_seed)
        agent_rng = np.random.default_rng(run_seed + 1)
        bandit = GaussianBandit(n_arms=n_arms, rng=bandit_rng, reward_std=reward_std)
        agent = BanditAgent(n_arms=n_arms, config=agent_config, rng=agent_rng)

        for step in range(steps):
            action = agent.select_action()
            outcome = bandit.pull(action)
            agent.update(action, outcome.reward)
            average_rewards[step] += outcome.reward
            optimal_action_rate[step] += 1.0 if outcome.is_optimal_action else 0.0

    return average_rewards / runs, optimal_action_rate / runs


def run_experiment(config: ExperimentConfig) -> dict[str, dict[str, dict[str, list[float]]]]:
    results = {}
    for n_arms in config.n_arms:
        arm_key = f"{n_arms}_arms"
        results[arm_key] = {}
        for index, agent_config in enumerate(default_agent_configs()):
            rewards, optimal_rate = run_policy(
                n_arms=n_arms,
                steps=config.steps,
                runs=config.runs,
                agent_config=agent_config,
                seed=config.seed + n_arms * 100 + index,
                reward_std=config.reward_std,
            )
            results[arm_key][agent_config.name] = {
                "average_reward": rewards.tolist(),
                "optimal_action_rate": optimal_rate.tolist(),
            }
    return results


def summarise_results(results: dict, final_window: int = 100) -> list[dict[str, float | str]]:
    rows = []
    for arm_key, policy_results in results.items():
        for policy_name, metrics in policy_results.items():
            reward = np.array(metrics["average_reward"], dtype=float)
            optimal_rate = np.array(metrics["optimal_action_rate"], dtype=float)
            window = min(final_window, len(reward))
            rows.append(
                {
                    "n_arms": arm_key.replace("_arms", ""),
                    "policy": policy_name,
                    "final_window_average_reward": round(float(np.mean(reward[-window:])), 4),
                    "final_window_optimal_action_rate": round(float(np.mean(optimal_rate[-window:])), 4),
                }
            )
    return rows


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(results: dict, output_dir: Path) -> None:
    for arm_key, policy_results in results.items():
        reward_series = {
            policy_name: np.array(metrics["average_reward"], dtype=float)
            for policy_name, metrics in policy_results.items()
        }
        optimal_series = {
            policy_name: np.array(metrics["optimal_action_rate"], dtype=float)
            for policy_name, metrics in policy_results.items()
        }
        save_line_chart(
            reward_series,
            output_dir / f"{arm_key}_average_reward.svg",
            title=f"Average Reward: {arm_key.replace('_', ' ')}",
            y_label="Average reward",
        )
        save_line_chart(
            optimal_series,
            output_dir / f"{arm_key}_optimal_action_rate.svg",
            title=f"Optimal Action Rate: {arm_key.replace('_', ' ')}",
            y_label="Optimal action rate",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-armed bandit simulations.")
    parser.add_argument("--arms", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-std", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        n_arms=tuple(args.arms),
        steps=args.steps,
        runs=args.runs,
        seed=args.seed,
        reward_std=args.reward_std,
        output_dir=args.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiment(config)
    summary_rows = summarise_results(results)

    (config.output_dir / "experiment_config.json").write_text(
        json.dumps({**asdict(config), "output_dir": str(config.output_dir)}, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "bandit_results.json").write_text(json.dumps(results), encoding="utf-8")
    write_summary_csv(summary_rows, config.output_dir / "summary_metrics.csv")
    save_plots(results, config.output_dir)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
