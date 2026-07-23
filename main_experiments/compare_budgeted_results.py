"""Paired stratified bootstrap comparison for OVO Backward/Realtime results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.recent_window_eval import score_ovo_br


def load_split(path: str | Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get(split, [])
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a list named {split!r}")
    return rows


def paired_split_report(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
    allow_candidate_subset: bool = False,
) -> dict[str, Any]:
    baseline = {(str(item.get("id")), str(item.get("task"))): item for item in baseline_rows}
    candidate = {(str(item.get("id")), str(item.get("task"))): item for item in candidate_rows}
    common = sorted(set(baseline) & set(candidate))
    if not common:
        raise ValueError("baseline and candidate have no paired samples")
    if allow_candidate_subset:
        if len(common) != len(candidate):
            raise ValueError(
                f"candidate contains samples absent from baseline: "
                f"baseline={len(baseline)}, candidate={len(candidate)}, common={len(common)}"
            )
    elif len(common) != len(baseline) or len(common) != len(candidate):
        raise ValueError(
            f"paired comparison requires identical samples: baseline={len(baseline)}, candidate={len(candidate)}, common={len(common)}"
        )
    by_task: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for key in common:
        base_row, candidate_row = baseline[key], candidate[key]
        ground_truth = str(base_row.get("ground_truth", candidate_row.get("ground_truth", "")))
        by_task[key[1]].append(
            (
                score_ovo_br(base_row.get("response"), ground_truth),
                score_ovo_br(candidate_row.get("response"), ground_truth),
            )
        )
    base_task = {task: 100.0 * np.mean([item[0] for item in values]) for task, values in by_task.items()}
    candidate_task = {task: 100.0 * np.mean([item[1] for item in values]) for task, values in by_task.items()}
    baseline_macro = float(np.mean(list(base_task.values())))
    candidate_macro = float(np.mean(list(candidate_task.values())))
    rng = np.random.default_rng(seed)
    differences = np.empty(max(1, int(samples)), dtype=np.float64)
    arrays = {task: np.asarray(values, dtype=np.float64) for task, values in by_task.items()}
    for iteration in range(len(differences)):
        task_differences = []
        for values in arrays.values():
            indices = rng.integers(0, len(values), size=len(values))
            sampled = values[indices]
            task_differences.append(100.0 * float(np.mean(sampled[:, 1] - sampled[:, 0])))
        differences[iteration] = float(np.mean(task_differences))
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {
        "paired_samples": len(common),
        "tasks": sorted(by_task),
        "baseline_by_task": base_task,
        "candidate_by_task": candidate_task,
        "baseline_macro": baseline_macro,
        "candidate_macro": candidate_macro,
        "difference": candidate_macro - baseline_macro,
        "difference_ci95": [float(lower), float(upper)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare budgeted VeriStream with a paired OVO baseline")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-candidate-subset",
        action="store_true",
        help="Compare a smoke-test candidate subset against matching rows from a full baseline.",
    )
    args = parser.parse_args()
    report = {
        split: paired_split_report(
            load_split(args.baseline, split),
            load_split(args.candidate, split),
            args.bootstrap_samples,
            args.seed + index,
            args.allow_candidate_subset,
        )
        for index, split in enumerate(("backward", "realtime"))
    }
    report["pareto_target"] = {
        "backward_at_least_62_09": report["backward"]["candidate_macro"] >= 62.09,
        "realtime_at_least_79_65": report["realtime"]["candidate_macro"] >= 79.65,
        "backward_ci_lower_above_zero": report["backward"]["difference_ci95"][0] > 0,
        "realtime_ci_lower_above_zero": report["realtime"]["difference_ci95"][0] > 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
