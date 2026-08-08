import json
import unittest
from pathlib import Path

from scripts.audit_route_a_claims import resolve_active_sources, strip_latex_comment


ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_ieee_journal.tex").read_text(encoding="utf-8")
INTRODUCTION = (ROOT / "sections" / "01_intro_information_budget.tex").read_text(encoding="utf-8")
MOTIVATION = (ROOT / "sections" / "03_motivating_observations_medium.tex").read_text(encoding="utf-8")
DISCUSSION = (ROOT / "sections" / "08_discussion_medium.tex").read_text(encoding="utf-8")
CONCLUSION = (ROOT / "sections" / "09_conclusion_medium.tex").read_text(encoding="utf-8")
ADVERSARIAL_REVIEW = ROOT / "docs" / "route_a_adversarial_review.md"
CLAIM_MAP = (ROOT / "docs" / "claim_evidence_map.md").read_text(encoding="utf-8")
SUMMARY = json.loads(
    (ROOT / "results" / "discriminability" / "route_a_grid_v1" / "summary" / "summary.json").read_text(
        encoding="utf-8"
    )
)
ACTIVE_SOURCES = resolve_active_sources(ROOT, ROOT / "main_ieee_journal.tex")
ACTIVE_TEXT = "\n".join(
    strip_latex_comment(line)
    for path in ACTIVE_SOURCES
    for line in path.read_text(encoding="utf-8").splitlines()
)


class SubmissionSafetyTests(unittest.TestCase):
    def test_title_and_abstract_match_the_scoped_contribution(self):
        self.assertNotIn("SNR Dynamics", MAIN)
        self.assertIn("Scoped Discriminability", MAIN)
        self.assertIn("0.94\\%", MAIN)
        self.assertIn("2,000", MAIN)
        self.assertIn("not Bayes-error formulas", MAIN)

    def test_introduction_reports_direct_validation_as_completed(self):
        self.assertIn("median absolute relative error", INTRODUCTION)
        self.assertNotIn("Direct validation target", INTRODUCTION)
        self.assertNotIn("Existing downstream accuracy sweeps", INTRODUCTION)

    def test_motivating_language_does_not_mix_protocols_or_restore_budget_branding(self):
        self.assertNotIn("\\textbf{Budget}", MOTIVATION)
        self.assertNotIn("Chameleon & 0.24 & 49.0", MOTIVATION)
        self.assertIn("paired protocol", MOTIVATION)

    def test_discussion_and_conclusion_retain_theory_and_experiment_limits(self):
        self.assertIn("prospective diagnostic benchmark has not yet been executed", DISCUSSION)
        self.assertNotIn("binary or balanced multi-class", CONCLUSION)
        self.assertNotIn("two-hop recovery ratio", CONCLUSION)
        self.assertIn("binary fixed-degree", CONCLUSION)
        self.assertIn("2,000", CONCLUSION)
        self.assertIn("legacy edge-randomization outcomes", CONCLUSION)

    def test_adversarial_review_keeps_remaining_no_go_items_visible(self):
        review = ADVERSARIAL_REVIEW.read_text(encoding="utf-8").lower()
        self.assertIn("full journal paper", review)
        self.assertIn("no-go", review)
        self.assertIn("prospective diagnostic benchmark", review)
        self.assertIn("degree-matched rerun", review)
        self.assertIn("reverse outline", review)

    def test_claim_map_and_abstract_match_the_frozen_summary(self):
        self.assertNotIn("pending direct validation", CLAIM_MAP.lower())
        self.assertIn("exact expression matches empirical moments across a frozen parameter grid", CLAIM_MAP.lower())
        self.assertIn("| supported for the prescribed surrogate grid |", CLAIM_MAP.lower())
        self.assertEqual(SUMMARY["record_counts"]["total"], 2000)
        self.assertEqual(SUMMARY["record_counts"]["primary_relative_error_defined"], 1920)
        self.assertEqual(SUMMARY["record_counts"]["approximation_defined"], 400)
        self.assertEqual(SUMMARY["record_counts"]["error"], 0)
        self.assertAlmostEqual(SUMMARY["primary_estimand"]["median"] * 100, 0.94, places=2)
        self.assertAlmostEqual(SUMMARY["secondary_estimand"]["median_absolute_error"], 0.293, places=3)

    def test_all_active_sources_use_scoped_terms_and_keep_pending_results_pending(self):
        self.assertNotIn("Budget", ACTIVE_TEXT)
        self.assertNotIn("recovery-ratio analysis", ACTIVE_TEXT)
        self.assertNotIn("prospective diagnostic benchmark was executed", ACTIVE_TEXT)
        self.assertNotIn("verified degree-matched results", ACTIVE_TEXT)


if __name__ == "__main__":
    unittest.main()
