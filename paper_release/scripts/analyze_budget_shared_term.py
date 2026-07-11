import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "external_validation_results.json"
OUTPUT = ROOT / "results" / "budget_shared_term_audit.json"


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    results = data["results"]
    datasets = [row["dataset"] for row in results]
    mlp = np.array([row["mlp_acc"] for row in results], dtype=float)
    best_gnn = np.array([row["best_gnn_acc"] for row in results], dtype=float)
    budget = 1.0 - mlp
    advantage = best_gnn - mlp

    observed = correlation(budget, advantage)
    rng = np.random.default_rng(20260711)
    permutation_count = 10000
    null_correlations = np.empty(permutation_count, dtype=float)
    for index in range(permutation_count):
        permuted_gnn = rng.permutation(best_gnn)
        null_correlations[index] = correlation(budget, permuted_gnn - mlp)

    bootstrap_count = 10000
    bootstrap_correlations = np.empty(bootstrap_count, dtype=float)
    for index in range(bootstrap_count):
        sample = rng.integers(0, len(results), size=len(results))
        bootstrap_correlations[index] = correlation(budget[sample], advantage[sample])

    empirical_p = float(
        (np.count_nonzero(null_correlations >= observed) + 1)
        / (permutation_count + 1)
    )
    audit = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "prediction_target": "best candidate GNN accuracy minus MLP accuracy",
        "datasets": datasets,
        "n_datasets": len(results),
        "observed_pearson_r": observed,
        "observed_r_squared": observed**2,
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap_correlations, 0.025)),
            float(np.quantile(bootstrap_correlations, 0.975)),
        ],
        "shared_term_null": {
            "method": "permute best-GNN accuracies while retaining the shared -MLP term",
            "permutations": permutation_count,
            "mean_r": float(np.mean(null_correlations)),
            "median_r": float(np.median(null_correlations)),
            "95_percent_interval": [
                float(np.quantile(null_correlations, 0.025)),
                float(np.quantile(null_correlations, 0.975)),
            ],
            "one_sided_empirical_p": empirical_p,
        },
        "interpretation": (
            "The raw correlation is not treated as independent predictive evidence if "
            "it is typical under the shared-term permutation null."
        ),
        "random_seed": 20260711,
    }
    OUTPUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
