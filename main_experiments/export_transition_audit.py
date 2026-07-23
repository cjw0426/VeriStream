"""Export a blind before/after transition-assessment audit packet."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _result_rows(result_dir: Path) -> list[dict[str, Any]]:
    merged = result_dir / "budgeted_ovo_backward_realtime.json"
    if merged.exists():
        payload = json.loads(merged.read_text(encoding="utf-8"))
        return [*payload.get("backward", []), *payload.get("realtime", [])]
    return [row for path in sorted(result_dir.glob("rank_*.jsonl")) for row in _load_jsonl(path)]


def collect_predictions(result_dir: Path) -> list[dict[str, Any]]:
    rows = _result_rows(result_dir)
    task_by_video = {f"ovo-{row['id']}": str(row.get("task", "unknown")) for row in rows}
    predictions: list[dict[str, Any]] = []
    for path in sorted((result_dir / "memory").glob("ovo-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        overviews = {item["memory_id"]: item for item in payload.get("overviews", [])}
        for overview in overviews.values():
            for audit in overview.get("proposal_audits", []):
                proposal_id = str(audit.get("proposal_id", ""))
                if not proposal_id or audit.get("before_time") is None or audit.get("after_time") is None:
                    continue
                sample_id = f"{overview['memory_id']}:{proposal_id}"
                admission = str(audit.get("admission", ""))
                if admission not in {
                    "local_semantic_event",
                    "global_visual_change",
                    "no_meaningful_change",
                }:
                    continue
                meaningful = admission != "no_meaningful_change"
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "sample_source": "proposal_assessment",
                        "model_decision_available": True,
                        "video_id": str(payload["video_id"]),
                        "annotation_id": str(payload["video_id"]).removeprefix("ovo-"),
                        "task": task_by_video.get(str(payload["video_id"]), "unknown"),
                        "before_time": float(audit["before_time"]),
                        "after_time": float(audit["after_time"]),
                        "before_frame_id": str(audit.get("before_frame_id", "")),
                        "after_frame_id": str(audit.get("after_frame_id", "")),
                        "model_meaningful": meaningful,
                        "model_admission": admission,
                        "model_transition_type": str(audit.get("transition_type", "")),
                        "model_before_state": str(audit.get("before_state", "")),
                        "model_change": str(audit.get("change", "")),
                        "model_after_state": str(audit.get("after_state", "")),
                        "semantic_gate_status": str(audit.get("semantic_gate_status", "unresolved")),
                        "evidence_admissible": bool(audit.get("evidence_admissible", False)),
                        "visual_change_prior": str(audit.get("visual_change_prior", "unavailable")),
                        "span": int(audit.get("span", 0)),
                        "clip_distance": float(audit.get("clip_distance", 0.0)),
                        "normalized_salience": float(audit.get("normalized_salience", 0.0)),
                    }
                )
    return predictions


def _frame_serial(frame_id: str) -> int | None:
    match = re.search(r":f(\d+)$", str(frame_id))
    return int(match.group(1)) if match else None


def collect_low_change_controls(result_dir: Path) -> list[dict[str, Any]]:
    """Collect low-CLIP-distance adjacent pairs without assigning model decisions."""
    rows = _result_rows(result_dir)
    task_by_video = {f"ovo-{row['id']}": str(row.get("task", "unknown")) for row in rows}
    controls: list[dict[str, Any]] = []
    for path in sorted((result_dir / "memory").glob("ovo-*.json")):
        embedding_path = path.with_suffix(path.suffix + ".embeddings.npz")
        if not embedding_path.exists():
            continue
        import numpy as np

        payload = json.loads(path.read_text(encoding="utf-8"))
        arrays = np.load(embedding_path, allow_pickle=False)
        ids = [str(item) for item in arrays["frame_ids"].tolist()]
        vectors = arrays["frame_embeddings"]
        vector_by_id = {frame_id: vectors[index] for index, frame_id in enumerate(ids)}
        proposed_pairs = {
            (str(audit.get("before_frame_id", "")), str(audit.get("after_frame_id", "")))
            for overview in payload.get("overviews", [])
            for audit in overview.get("proposal_audits", [])
        }
        for overview in payload.get("overviews", []):
            frame_ids = list(overview.get("l0_frame_ids", []))
            for before_id, after_id in zip(frame_ids, frame_ids[1:]):
                if (before_id, after_id) in proposed_pairs:
                    continue
                before_serial, after_serial = _frame_serial(before_id), _frame_serial(after_id)
                left, right = vector_by_id.get(before_id), vector_by_id.get(after_id)
                if before_serial is None or after_serial is None or left is None or right is None:
                    continue
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                distance = 1.0 - float(np.dot(left, right) / denominator) if denominator > 0 else 1.0
                controls.append(
                    {
                        "sample_id": f"{overview['memory_id']}:C{before_serial:06d}",
                        "sample_source": "low_change_control",
                        "model_decision_available": False,
                        "video_id": str(payload["video_id"]),
                        "annotation_id": str(payload["video_id"]).removeprefix("ovo-"),
                        "task": task_by_video.get(str(payload["video_id"]), "unknown"),
                        "before_time": float(before_serial),
                        "after_time": float(after_serial),
                        "before_frame_id": before_id,
                        "after_frame_id": after_id,
                        "model_meaningful": None,
                        "model_admission": None,
                        "model_transition_type": None,
                        "semantic_gate_status": "not_evaluated_control",
                        "evidence_admissible": None,
                        "visual_change_prior": "unavailable",
                        "span": 1,
                        "clip_distance": distance,
                        "normalized_salience": 0.0,
                    }
                )
    return controls


def stratified_sample(
    rows: list[dict[str, Any]], max_items: int, seed: int
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["task"]), str(row["model_transition_type"]))].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < min(max_items, len(rows)):
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < max_items:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _read_frame(video_path: Path, timestamp: float) -> Image.Image:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    fps = float(reader.get_avg_fps())
    index = min(max(0, int(round(max(0.0, timestamp) * fps))), max(0, len(reader) - 1))
    return Image.fromarray(reader[index].asnumpy()).convert("RGB")


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def render_pair(
    video_path: Path,
    before_time: float,
    after_time: float,
    sample_id: str,
    destination: Path,
) -> None:
    panel_size = (448, 252)
    before = _fit_panel(_read_frame(video_path, before_time), panel_size)
    after = _fit_panel(_read_frame(video_path, after_time), panel_size)
    canvas = Image.new("RGB", (panel_size[0] * 2, panel_size[1] + 48), "white")
    canvas.paste(before, (0, 48))
    canvas.paste(after, (panel_size[0], 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"{sample_id}", fill="black")
    draw.text((8, 26), f"BEFORE  t={before_time:.2f}s", fill="black")
    draw.text((panel_size[0] + 8, 26), f"AFTER  t={after_time:.2f}s", fill="black")
    draw.line((panel_size[0], 0, panel_size[0], canvas.height), fill="#777777", width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        action="append",
        required=True,
        help="Result directory to audit; repeat the flag to combine disjoint pilot shards.",
    )
    parser.add_argument("--chunked-dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-items", type=int, default=80)
    parser.add_argument(
        "--control-items", type=int, default=0,
        help="Reserve this many slots for low-change annotation controls (not model predictions).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result_dirs = [Path(item) for item in args.result_dir]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = [row for result_dir in result_dirs for row in collect_predictions(result_dir)]
    control_count = min(max(0, int(args.control_items)), max(0, int(args.max_items) - 1))
    prediction_by_id = {str(row["sample_id"]): row for row in combined}
    predictions = stratified_sample(
        list(prediction_by_id.values()), max(1, int(args.max_items) - control_count), int(args.seed)
    )
    if control_count:
        controls = [row for result_dir in result_dirs for row in collect_low_change_controls(result_dir)]
        control_by_id = {str(row["sample_id"]): row for row in controls}
        controls = sorted(control_by_id.values(), key=lambda item: (item["clip_distance"], item["sample_id"]))
        if len(controls) < control_count:
            raise ValueError(f"Requested {control_count} controls but found only {len(controls)}")
        step = max(1, len(controls) // control_count)
        predictions.extend(controls[index * step] for index in range(control_count))
        random.Random(int(args.seed) + 1).shuffle(predictions)
    if not predictions:
        raise ValueError(f"No transition memories found under: {result_dirs}")

    blind_rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions, start=1):
        audit_id = f"TA{index:04d}"
        image_name = f"pairs/{audit_id}.jpg"
        render_pair(
            Path(args.chunked_dir) / f"{prediction['annotation_id']}.mp4",
            prediction["before_time"],
            prediction["after_time"],
            audit_id,
            output_dir / image_name,
        )
        prediction["audit_id"] = audit_id
        blind_rows.append(
            {
                "audit_id": audit_id,
                "task": prediction["task"],
                "sample_source": prediction["sample_source"],
                "pair_image": image_name,
                "before_time": prediction["before_time"],
                "after_time": prediction["after_time"],
                "human_meaningful": None,
                "human_transition_type": None,
                "human_temporal_alignment": None,
                "human_before_state": "",
                "human_change": "",
                "human_after_state": "",
                "annotator_id": "",
                "notes": "",
            }
        )

    _write_jsonl(output_dir / "blind_labels.jsonl", blind_rows)
    _write_jsonl(output_dir / "model_predictions.jsonl", predictions)
    manifest = {
        "result_dirs": [str(item) for item in result_dirs],
        "sample_count": len(predictions),
        "seed": int(args.seed),
        "sampling": (
            "round-robin over task x model_transition_type for proposals; "
            "evenly spaced low-CLIP-distance adjacent pairs for controls"
        ),
        "proposal_prediction_count": sum(item["model_decision_available"] for item in predictions),
        "control_count": sum(not item["model_decision_available"] for item in predictions),
        "task_counts": Counter(item["task"] for item in predictions),
        "model_type_counts": Counter(item["model_transition_type"] for item in predictions),
        "limitations": [
            "The blind packet measures assessment quality on proposed pairs.",
            "Low-change controls calibrate annotation and are excluded from model precision/recall.",
            "Proposal detector recall requires a separate event-first annotation over full blocks.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Blind Transition Audit\n\n"
        "Open each `pair_image` without viewing `model_predictions.jsonl`. Fill every null/empty "
        "human field in `blind_labels.jsonl`. `human_meaningful` and "
        "`human_temporal_alignment` must be JSON booleans. `human_transition_type` must be one of "
        "`object_action`, `state_change`, `scene_cut`, `camera_motion`, or "
        "`no_meaningful_change`. Use `no_meaningful_change` whenever `human_meaningful=false`.\n\n"
        "Rows with `sample_source=proposal_assessment` evaluate semantic assessment on proposed "
        "pairs. Rows with `sample_source=low_change_control` calibrate annotation and are excluded "
        "from model precision/recall because the model did not assess them. This packet does not measure "
        "proposal detector recall over all events in a video. Create separate copies for two "
        "annotators, then pass both files as repeated `--labels` arguments to "
        "`score_transition_audit.py`.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
