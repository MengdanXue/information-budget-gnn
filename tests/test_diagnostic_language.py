import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR = (ROOT / "sections" / "05_aggregation_damage_medium.tex").read_text(
    encoding="utf-8"
)
TWO_HOP = (ROOT / "sections" / "06_dual_heterophily_medium.tex").read_text(
    encoding="utf-8"
)
EXPERIMENTS = (ROOT / "sections" / "07_experiments_medium.tex").read_text(
    encoding="utf-8"
)


class DiagnosticLanguageTests(unittest.TestCase):
    def test_adr_is_an_absolute_percentage_point_difference(self):
        self.assertNotIn("Aggregation Damage Ratio", ADR)
        self.assertNotIn("destroys information", ADR)
        self.assertIn("percentage points", ADR)
        self.assertRegex(ADR, r"100\s*\\bigl?\(")

    def test_two_hop_reporting_includes_stable_quantities_and_missingness(self):
        self.assertIn(r"\Delta h", TWO_HOP)
        self.assertIn("class-prior", TWO_HOP.lower())
        self.assertIn("uncertainty", TWO_HOP.lower())
        self.assertIn("not archived", TWO_HOP.lower())
        self.assertNotRegex(TWO_HOP, r"R\s*[><]=?\s*1\.[05]")

    def test_experiments_use_a_checklist_not_a_selector_algorithm(self):
        self.assertNotIn(r"\begin{algorithm}", EXPERIMENTS)
        self.assertNotRegex(EXPERIMENTS, re.compile(r"88\.9\\%|77\.8\\%"))
        self.assertIn("reporting checklist", EXPERIMENTS.lower())


if __name__ == "__main__":
    unittest.main()
