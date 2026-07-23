"""Score completed blind transition labels against hidden model predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def cohen_kappa(
    first: list[dict[str, Any]], second: list[dict[str, Any]], field: str
) -> float | None:
    first_by_id = {str(item["audit_id"]): item.get(field) for item in first}
    second_by_id = {str(item["audit_id"]): item.get(field) for item in second}
    common = sorted(set(first_by_id) & set(second_by_id))
    if not common:
        return None
    categories = sorted(
        {first_by_id[item] for item in common} | {second_by_id[item] for item in common},
        key=str,
    )
    observed = sum(first_by_id[item] == second_by_id[item] for item in common) / len(common)
    expected = sum(
        (sum(first_by_id[item] == category for item in common) / len(common))
        * (sum(second_by_id[item] == category for item in common) / len(common))
        for category in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def score_rows(
    labels: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    prediction_by_id = {str(item["audit_id"]): item for item in predictions}
    missing = [item["audit_id"] for item in labels if str(item["audit_id"]) not in prediction_by_id]
    if missing:
        raise ValueError(f"Labels have no matching prediction: {missing[:5]}")
    incomplete = [
        item["audit_id"]
        for item in labels
        if not isinstance(item.get("human_meaningful"), bool)
        or not isinstance(item.get("human_temporal_alignment"), bool)
        or not item.get("human_transition_type")
    ]
    if incomplete:
        raise ValueError(f"Incomplete human labels: {incomplete[:10]}")

    def report(items: list[dict[str, Any]]) -> dict[str, Any]:
        assessed_items = [
            item for item in items
            if bool(prediction_by_id[str(item["audit_id"])].get("model_decision_available", True))
        ]
        tp = fp = tn = fn = type_correct = type_total = aligned = 0
        gate_tp = gate_fp = gate_tn = gate_fn = 0
        for label in assessed_items:
            prediction = prediction_by_id[str(label["audit_id"])]
            human = bool(label["human_meaningful"])
            model = bool(prediction["model_meaningful"])
            tp += int(model and human)
            fp += int(model and not human)
            tn += int(not model and not human)
            fn += int(not model and human)
            aligned += int(label["human_temporal_alignment"])
            human_admissible = human and str(label["human_transition_type"]) in {
                "object_action", "state_change"
            }
            model_admissible = bool(prediction.get("evidence_admissible", model))
            gate_tp += int(model_admissible and human_admissible)
            gate_fp += int(model_admissible and not human_admissible)
            gate_tn += int(not model_admissible and not human_admissible)
            gate_fn += int(not model_admissible and human_admissible)
            if model and human:
                type_total += 1
                type_correct += int(
                    str(prediction["model_transition_type"])
                    == str(label["human_transition_type"])
                )
        return {
            "count": len(items),
            "model_assessed_count": len(assessed_items),
            "annotation_control_count": len(items) - len(assessed_items),
            "meaningful_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "assessment_precision": _safe_div(tp, tp + fp),
            "assessment_recall_on_proposed_pairs": _safe_div(tp, tp + fn),
            "assessment_f1": _safe_div(2 * tp, 2 * tp + fp + fn),
            "meaningful_accuracy": _safe_div(tp + tn, len(assessed_items)),
            "transition_type_accuracy_on_joint_positives": _safe_div(type_correct, type_total),
            "temporal_alignment_rate": _safe_div(aligned, len(assessed_items)),
            "semantic_gate_confusion": {
                "tp": gate_tp, "fp": gate_fp, "tn": gate_tn, "fn": gate_fn
            },
            "semantic_gate_precision": _safe_div(gate_tp, gate_tp + gate_fp),
            "semantic_gate_recall": _safe_div(gate_tp, gate_tp + gate_fn),
            "semantic_gate_f1": _safe_div(2 * gate_tp, 2 * gate_tp + gate_fp + gate_fn),
        }

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        by_task[str(label.get("task", "unknown"))].append(label)
    return {
        "overall": report(labels),
        "by_task": {task: report(items) for task, items in sorted(by_task.items())},
        "proposal_detector_recall": None,
        "proposal_detector_recall_note": (
            "Not measurable from proposed-pair labels; use event-first full-block annotation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        action="append",
        required=True,
        help="Completed label JSONL; repeat for two independent annotators.",
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions = _load_jsonl(Path(args.predictions))
    label_sets = [(Path(path), _load_jsonl(Path(path))) for path in args.labels]
    individual = {
        path.name: score_rows(labels, predictions) for path, labels in label_sets
    }
    if len(label_sets) == 1:
        report = next(iter(individual.values()))
    else:
        first, second = label_sets[:2]
        report = {
            "individual": individual,
            "agreement_first_two_annotators": {
                "annotator_files": [first[0].name, second[0].name],
                "meaningful_cohen_kappa": cohen_kappa(
                    first[1], second[1], "human_meaningful"
                ),
                "transition_type_cohen_kappa": cohen_kappa(
                    first[1], second[1], "human_transition_type"
                ),
                "temporal_alignment_cohen_kappa": cohen_kappa(
                    first[1], second[1], "human_temporal_alignment"
                ),
            },
            "adjudication_note": (
                "Resolve disagreements in a separate adjudicated label file before final reporting."
            ),
        }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
