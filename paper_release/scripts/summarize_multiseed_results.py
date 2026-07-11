import argparse
import json
import math
import random
import statistics
from pathlib import Path


REQUIRED_RUN_FIELDS = {"seed", "test_accuracy"}


def normalize(data: dict) -> dict:
    if isinstance(data.get("datasets"), list):
        return data
    if isinstance(data.get("results"), list):
        return {
            "schema_version": data.get("schema_version"),
            "experiment": data.get("experiment"),
            "datasets": [
                {
                    "name": result["dataset"],
                    "models": result["per_seed_results"],
                }
                for result in data["results"]
            ],
        }
    if isinstance(data.get("datasets"), dict):
        datasets = []
        for name, result in data["datasets"].items():
            if "error" in result:
                continue
            datasets.append(
                {
                    "name": name,
                    "models": {
                        model: values["per_seed_results"]
                        for model, values in result["models"].items()
                    },
                }
            )
        return {
            "schema_version": data.get("schema_version"),
            "experiment": data.get("experiment"),
            "datasets": datasets,
        }
    raise ValueError("unsupported result layout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate multi-seed results and generate audited statistics."
    )
    parser.add_argument("input", type=Path, help="Raw multi-seed JSON file")
    parser.add_argument("--summary", type=Path, required=True, help="Summary JSON output")
    parser.add_argument("--latex", type=Path, required=True, help="LaTeX rows output")
    parser.add_argument("--min-seeds", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    values: list[float], samples: int, rng: random.Random
) -> tuple[float, float]:
    size = len(values)
    means = [
        statistics.fmean(rng.choice(values) for _ in range(size))
        for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def validate(data: dict, min_seeds: int) -> None:
    if data.get("schema_version") != "1.0":
        raise ValueError("schema_version must be '1.0'")
    if not isinstance(data.get("datasets"), list) or not data["datasets"]:
        raise ValueError("datasets must be a non-empty list")
    for dataset in data["datasets"]:
        name = dataset.get("name", "<unnamed>")
        models = dataset.get("models")
        if not isinstance(models, dict) or not models:
            raise ValueError(f"{name}: models must be a non-empty object")
        expected_seeds = None
        for model, runs in models.items():
            if len(runs) < min_seeds:
                raise ValueError(
                    f"{name}/{model}: found {len(runs)} seeds; require {min_seeds}"
                )
            seeds = []
            for run in runs:
                missing = REQUIRED_RUN_FIELDS - run.keys()
                if missing:
                    raise ValueError(f"{name}/{model}: missing fields {sorted(missing)}")
                accuracy = run["test_accuracy"]
                if not 0 <= accuracy <= 1:
                    raise ValueError(f"{name}/{model}: test_accuracy outside [0, 1]")
                seeds.append(run["seed"])
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"{name}/{model}: duplicate seeds")
            seed_set = set(seeds)
            if expected_seeds is None:
                expected_seeds = seed_set
            elif seed_set != expected_seeds:
                raise ValueError(f"{name}: models do not share identical paired seeds")


def summarize(data: dict, bootstrap_samples: int) -> dict:
    rng = random.Random(20260711)
    datasets = []
    for dataset in data["datasets"]:
        model_summaries = {}
        by_model_seed = {}
        for model, runs in dataset["models"].items():
            values = [run["test_accuracy"] for run in runs]
            interval = bootstrap_mean_ci(values, bootstrap_samples, rng)
            model_summaries[model] = {
                "n_seeds": len(values),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "bootstrap_95_ci": list(interval),
            }
            by_model_seed[model] = {run["seed"]: run["test_accuracy"] for run in runs}

        paired_differences = {}
        if "MLP" in by_model_seed:
            for model, values in by_model_seed.items():
                if model == "MLP":
                    continue
                differences = [
                    values[seed] - by_model_seed["MLP"][seed]
                    for seed in sorted(values)
                ]
                interval = bootstrap_mean_ci(differences, bootstrap_samples, rng)
                paired_differences[f"{model}-MLP"] = {
                    "mean": statistics.fmean(differences),
                    "sample_std": statistics.stdev(differences),
                    "bootstrap_95_ci": list(interval),
                }
        datasets.append(
            {
                "name": dataset["name"],
                "models": model_summaries,
                "paired_differences": paired_differences,
            }
        )
    return {
        "schema_version": "1.0",
        "source_experiment": data.get("experiment"),
        "datasets": datasets,
    }


def latex_rows(summary: dict) -> str:
    rows = ["% Generated by summarize_multiseed_results.py. Do not edit manually."]
    for dataset in summary["datasets"]:
        for model, stats in dataset["models"].items():
            mean = 100 * stats["mean"]
            std = 100 * stats["sample_std"]
            low, high = (100 * value for value in stats["bootstrap_95_ci"])
            rows.append(
                f"{dataset['name']} & {model} & {mean:.2f} $\\pm$ {std:.2f} "
                f"& [{low:.2f}, {high:.2f}] \\\\"
            )
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    data = normalize(json.loads(args.input.read_text(encoding="utf-8")))
    validate(data, args.min_seeds)
    summary = summarize(data, args.bootstrap_samples)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.latex.write_text(latex_rows(summary), encoding="utf-8")


if __name__ == "__main__":
    main()

