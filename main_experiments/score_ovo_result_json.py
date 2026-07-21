from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.recent_window_eval import calculate_ovo_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Write machine-readable OVO score summary for an eval result JSON.")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    with open(args.result_json, encoding="utf-8") as handle:
        results = json.load(handle)

    scores = calculate_ovo_scores(
        results.get("backward", []),
        results.get("realtime", []),
        results.get("forward", []),
    )
    section_avgs: list[float] = []
    for section in ("backward", "realtime", "forward"):
        task_rows = scores.get(section, {})
        if task_rows:
            section_avgs.append(sum(float(row["accuracy"]) for row in task_rows.values()) / len(task_rows))

    summary = {
        "result_json": args.result_json,
        "config": results.get("config", {}),
        "scores": scores,
        "total_avg_pct": sum(section_avgs) / len(section_avgs) if section_avgs else None,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
