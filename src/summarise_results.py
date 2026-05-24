import csv
from pathlib import Path


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def best_policy_by_arm(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best = {}
    for row in rows:
        n_arms = row["n_arms"]
        reward = float(row["final_window_average_reward"])
        if n_arms not in best or reward > float(best[n_arms]["final_window_average_reward"]):
            best[n_arms] = row
    return best


def main() -> None:
    rows = read_summary(Path("results") / "summary_metrics.csv")
    best = best_policy_by_arm(rows)
    for n_arms, row in best.items():
        print(
            f"{n_arms} arms: {row['policy']} achieved final-window average reward "
            f"{row['final_window_average_reward']} and optimal-action rate "
            f"{row['final_window_optimal_action_rate']}."
        )


if __name__ == "__main__":
    main()
