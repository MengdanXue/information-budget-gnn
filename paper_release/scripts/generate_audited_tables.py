import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables" / "generated"


def pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def generate_dual_heterophily_rows() -> None:
    source = RESULTS / "dual_heterophily_results.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for result in data["results"]:
        if result["heterophily_type"] not in {"Type A", "Type B"}:
            continue
        dataset = result["dataset"].capitalize()
        row = (
            f"{dataset} & {result['homophily_1hop']:.3f} & "
            f"{result['homophily_2hop']:.3f} & {result['recovery_ratio']:.2f} & "
            f"{result['heterophily_type'][-1]} & {result['best_model']} \\\\"
        )
        rows.append(row)
    header = "% Generated from code/results/dual_heterophily_results.json. Do not edit manually.\n"
    (TABLES / "dual_heterophily_rows.tex").write_text(
        header + "\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8"
    )


def generate_dual_model_rows() -> None:
    source = RESULTS / "dual_heterophily_results.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for result in data["results"]:
        if result["heterophily_type"] not in {"Type A", "Type B"}:
            continue
        dataset = result["dataset"].capitalize()
        rows.append(
            f"{dataset} & {pct(result['mlp_acc'])} & {pct(result['gcn_acc'])} & "
            f"{pct(result['sage_acc'])} & {pct(result['h2gcn_acc'])} & "
            f"{pct(result['linkx_acc'])} \\\\"
        )
    header = "% Generated from code/results/dual_heterophily_results.json. Do not edit manually.\n"
    (TABLES / "dual_model_rows.tex").write_text(
        header + "\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8"
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    generate_dual_heterophily_rows()
    generate_dual_model_rows()


if __name__ == "__main__":
    main()
