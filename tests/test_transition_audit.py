"""Tests for blind transition-audit sampling and scoring."""

from __future__ import annotations

import unittest

from main_experiments.export_transition_audit import stratified_sample
from main_experiments.score_transition_audit import cohen_kappa, score_rows


class TransitionAuditTest(unittest.TestCase):
    def test_stratified_sample_covers_task_type_buckets(self) -> None:
        rows = [
            {"sample_id": "a", "task": "EPM", "model_transition_type": "state_change"},
            {"sample_id": "b", "task": "EPM", "model_transition_type": "object_action"},
            {"sample_id": "c", "task": "ASI", "model_transition_type": "state_change"},
            {"sample_id": "d", "task": "ASI", "model_transition_type": "object_action"},
        ]
        selected = stratified_sample(rows, max_items=4, seed=42)
        self.assertEqual({item["sample_id"] for item in selected}, {"a", "b", "c", "d"})

    def test_score_rows_separates_assessment_from_detector_recall(self) -> None:
        predictions = [
            {
                "audit_id": "TA0001",
                "model_meaningful": True,
                "model_transition_type": "object_action",
            },
            {
                "audit_id": "TA0002",
                "model_meaningful": True,
                "model_transition_type": "state_change",
            },
        ]
        labels = [
            {
                "audit_id": "TA0001",
                "task": "EPM",
                "human_meaningful": True,
                "human_transition_type": "object_action",
                "human_temporal_alignment": True,
            },
            {
                "audit_id": "TA0002",
                "task": "EPM",
                "human_meaningful": False,
                "human_transition_type": "no_meaningful_change",
                "human_temporal_alignment": True,
            },
        ]
        report = score_rows(labels, predictions)
        self.assertEqual(report["overall"]["assessment_precision"], 0.5)
        self.assertEqual(report["overall"]["assessment_recall_on_proposed_pairs"], 1.0)
        self.assertIsNone(report["proposal_detector_recall"])

    def test_cohen_kappa_reports_perfect_agreement(self) -> None:
        labels = [
            {"audit_id": "TA0001", "human_meaningful": True},
            {"audit_id": "TA0002", "human_meaningful": False},
        ]
        self.assertEqual(cohen_kappa(labels, list(labels), "human_meaningful"), 1.0)


if __name__ == "__main__":
    unittest.main()
