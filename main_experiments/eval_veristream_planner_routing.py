"""Planner-only routing and confidence diagnostics for VeriStream."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.recent_window_eval import RecentWindowQAModel
from lib.veristream_budgeted import BudgetedVeriStreamAgent, parse_evidence_plan
from main_experiments.eval_qwen3vl_ovo_budgeted import (
    _append_jsonl,
    _load_jsonl,
    _validate_or_write_manifest,
    _validate_pretrained_reference,
    _wait_for_rank_done,
)
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS


def _options_text(annotation: dict[str, Any]) -> str:
    return "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(annotation.get("options", []))
    )


def _evaluate(annotation: dict[str, Any], model: RecentWindowQAModel, threshold: float) -> dict[str, Any]:
    question = str(annotation.get("question", ""))
    raw = model.generate_from_text(BudgetedVeriStreamAgent._planner_prompt(question, _options_text(annotation)))
    plan = parse_evidence_plan(raw, question)
    raw_history = bool(plan.relations) or any(item.model_scope in {"historical", "both"} for item in plan.needs)
    effective_history = bool(plan.relations) or any(item.requires_history for item in plan.needs)
    minimum_confidence = min((item.confidence for item in plan.needs), default=0.0)
    fast_path_eligible = (
        not plan.relations
        and bool(plan.needs)
        and all(item.is_current_only and item.confidence >= threshold for item in plan.needs)
    )
    expected_history = annotation.get("task") in BACKWARD_TASKS
    raw_predicted_correct = raw_history == expected_history
    effective_predicted_correct = effective_history == expected_history
    return {
        "id": annotation.get("id"),
        "task": annotation.get("task"),
        "question": question,
        "expected_history": expected_history,
        "raw_history": raw_history,
        "effective_history": effective_history,
        "fast_path_eligible": fast_path_eligible,
        "scope_confidence": minimum_confidence,
        "raw_predicted_correct": raw_predicted_correct,
        "effective_predicted_correct": effective_predicted_correct,
        "needs": [vars(item) for item in plan.needs],
        "relations": [vars(item) for item in plan.relations],
        "planner_raw": str(raw),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in rows if not item.get("error")]
    expected_history = [item for item in valid if item["expected_history"]]
    expected_current = [item for item in valid if not item["expected_history"]]
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        members = [
            item for item in valid
            if lower <= float(item["scope_confidence"]) <= upper
            and (lower == 0.8 or float(item["scope_confidence"]) < upper)
        ]
        if not members:
            continue
        confidence = sum(float(item["scope_confidence"]) for item in members) / len(members)
        accuracy = sum(bool(item["raw_predicted_correct"]) for item in members) / len(members)
        ece += len(members) / max(len(valid), 1) * abs(confidence - accuracy)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_confidence": confidence,
                "routing_accuracy": accuracy,
            }
        )
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0, "fast_path": 0})
    for item in valid:
        task = str(item.get("task"))
        by_task[task]["total"] += 1
        by_task[task]["correct"] += int(bool(item["effective_predicted_correct"]))
        by_task[task]["fast_path"] += int(bool(item["fast_path_eligible"]))
    return {
        "samples": len(valid),
        "errors": len(rows) - len(valid),
        "raw_routing_accuracy": sum(bool(item["raw_predicted_correct"]) for item in valid) / max(len(valid), 1),
        "effective_routing_accuracy": sum(bool(item["effective_predicted_correct"]) for item in valid)
        / max(len(valid), 1),
        "raw_historical_false_negative_rate": sum(not item["raw_history"] for item in expected_history)
        / max(len(expected_history), 1),
        "effective_historical_false_negative_rate": sum(not item["effective_history"] for item in expected_history)
        / max(len(expected_history), 1),
        "raw_current_false_positive_rate": sum(item["raw_history"] for item in expected_current)
        / max(len(expected_current), 1),
        "effective_current_false_positive_rate": sum(item["effective_history"] for item in expected_current)
        / max(len(expected_current), 1),
        "fast_path_coverage": sum(bool(item["fast_path_eligible"]) for item in valid) / max(len(valid), 1),
        "ece": ece,
        "reliability_bins": bins,
        "by_task": {
            task: {
                **counts,
                "accuracy": counts["correct"] / max(counts["total"], 1),
            }
            for task, counts in sorted(by_task.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VeriStream Planner routing without video indexing")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--anno-path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--current-fast-path-confidence", type=float, default=0.7)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    args = parser.parse_args()
    _validate_pretrained_reference(args.model_path, "--model-path")
    annotations = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    rows = [
        item for item in annotations
        if item.get("task") in BACKWARD_TASKS or item.get("task") in REAL_TIME_TASKS
    ]
    random.Random(42).shuffle(rows)
    if args.max_samples_per_split is not None:
        grouped = {
            "backward": [item for item in rows if item.get("task") in BACKWARD_TASKS],
            "realtime": [item for item in rows if item.get("task") in REAL_TIME_TASKS],
        }
        rows = [
            item
            for group in grouped.values()
            for item in group[: max(1, args.max_samples_per_split)]
        ]
    accelerator = Accelerator()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_manifest(result_dir, args, accelerator.process_index)
    checkpoint = result_dir / f"rank_{accelerator.process_index}.jsonl"
    done_path = result_dir / f"rank_{accelerator.process_index}.done"
    done_path.unlink(missing_ok=True)
    completed = {
        (str(item.get("id")), str(item.get("task"))) for item in _load_jsonl(checkpoint)
    }
    local_rows = rows[accelerator.process_index :: accelerator.num_processes]
    model = RecentWindowQAModel(args.model_path, accelerator.device, args.max_new_tokens)
    iterator = tqdm(local_rows, desc=f"Planner rank{accelerator.process_index}", disable=not accelerator.is_main_process)
    for annotation in iterator:
        key = (str(annotation.get("id")), str(annotation.get("task")))
        if key in completed:
            continue
        try:
            result = _evaluate(annotation, model, args.current_fast_path_confidence)
        except Exception as exc:
            result = {
                "id": annotation.get("id"),
                "task": annotation.get("task"),
                "question": annotation.get("question"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        _append_jsonl(checkpoint, result)
        completed.add(key)
    done_path.write_text("done\n", encoding="utf-8")
    if not accelerator.is_main_process:
        return
    _wait_for_rank_done(result_dir, accelerator.num_processes)
    merged_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for rank in range(accelerator.num_processes):
        for item in _load_jsonl(result_dir / f"rank_{rank}.jsonl"):
            merged_by_key[(str(item.get("id")), str(item.get("task")))] = item
    merged = list(merged_by_key.values())
    summary = _summary(merged)
    (result_dir / "planner_routing_results.json").write_text(
        json.dumps({"config": vars(args), "summary": summary, "results": merged}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (result_dir / "planner_routing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
