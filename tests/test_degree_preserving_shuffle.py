import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-analysis" / "Scripts" / "python.exe"
SCRIPT = ROOT / "experiments" / "degree_preserving_edge_randomization.py"
EXPERIMENTS_SECTION = ROOT / "sections" / "07_experiments_medium.tex"
LEGACY_AUDIT = ROOT / "docs" / "legacy_edge_randomization_audit.md"


def load_module():
    specification = importlib.util.spec_from_file_location("edge_randomization", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def cubic_graph_edges():
    cycle = [(node, (node + 1) % 10) for node in range(10)]
    matching = [(node, node + 5) for node in range(5)]
    return cycle + matching


class DegreePreservingShuffleTests(unittest.TestCase):
    def test_double_edge_swaps_preserve_all_declared_graph_invariants(self):
        module = load_module()
        labels = [node % 2 for node in range(10)]
        first = module.run_intervention(
            node_count=10,
            edges=cubic_graph_edges(),
            labels=labels,
            n_swaps=30,
            seed=20260808,
        )
        second = module.run_intervention(
            node_count=10,
            edges=cubic_graph_edges(),
            labels=labels,
            n_swaps=30,
            seed=20260808,
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first["original_edges"], first["randomized_edges"])
        invariants = first["audit"]["invariants"]
        self.assertTrue(invariants["node_count_identical"])
        self.assertTrue(invariants["edge_count_identical"])
        self.assertTrue(invariants["sorted_degree_sequence_identical"])
        self.assertEqual(
            invariants["degree_sequence_sha256_before"],
            invariants["degree_sequence_sha256_after"],
        )
        self.assertEqual(first["swap_status"]["successful_swaps"], 30)
        randomized = {tuple(edge) for edge in first["randomized_edges"]}
        self.assertEqual(len(randomized), 15)
        self.assertTrue(all(left < right for left, right in randomized))

    def test_audit_reports_concurrent_structural_changes(self):
        module = load_module()
        audit = module.run_intervention(
            node_count=10,
            edges=cubic_graph_edges(),
            labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            n_swaps=20,
            seed=7,
        )["audit"]
        for phase in ("before", "after"):
            self.assertEqual(
                {
                    "homophily",
                    "component_count",
                    "largest_component_fraction",
                    "degree_assortativity",
                    "average_clustering",
                },
                set(audit[phase]),
            )
        self.assertIn("homophily", audit["changes"])
        self.assertIn("component_count", audit["changes"])
        self.assertIn("degree_assortativity", audit["changes"])
        self.assertIn("average_clustering", audit["changes"])

    def test_non_simple_input_is_rejected(self):
        module = load_module()
        for edges in ([(0, 0)], [(0, 1), (1, 0)], [(0, 3)]):
            with self.subTest(edges=edges):
                with self.assertRaises(ValueError):
                    module.run_intervention(
                        node_count=3,
                        edges=edges,
                        labels=[0, 1, 0],
                        n_swaps=1,
                        seed=1,
                    )

    def test_cli_is_immutable_and_records_failure_free_provenance(self):
        payload = {
            "schema_version": "1.0",
            "node_count": 10,
            "edges": cubic_graph_edges(),
            "labels": [node % 2 for node in range(10)],
            "n_swaps": 10,
            "seed": 123,
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            input_path = directory / "input.json"
            output_path = directory / "output.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            command = [str(PYTHON), str(SCRIPT), "--input", str(input_path), "--output", str(output_path)]
            first = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            record = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(record["schema_version"], "1.0")
            self.assertEqual(record["algorithm"], "simple_undirected_double_edge_swap")
            self.assertEqual(record["status"], "success")
            before = output_path.read_bytes()
            second = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(before, output_path.read_bytes())

    def test_legacy_results_remain_explicitly_separate_from_the_verified_implementation(self):
        experiments = EXPERIMENTS_SECTION.read_text(encoding="utf-8").lower()
        audit = LEGACY_AUDIT.read_text(encoding="utf-8").lower()
        self.assertIn("discarded self-loops", experiments)
        self.assertIn("collapsed duplicate edges", experiments)
        self.assertIn("has not been used to regenerate", experiments)
        self.assertIn("does not isolate homophily", audit)
        self.assertIn("pending a clean rerun", audit)


if __name__ == "__main__":
    unittest.main()
