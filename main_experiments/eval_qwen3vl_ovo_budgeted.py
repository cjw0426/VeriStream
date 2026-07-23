"""Evaluate budgeted VeriStream on OVO Backward and Realtime splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.clip_topk_selector import CLIPTopKFrameSelector
from lib.recent_window_eval import build_ovo_prompt, calculate_ovo_scores, decode_video_to_chunks_qwen, print_ovo_results
from lib.recent_window_eval_qwen3 import RecentWindowQAModel
from lib.veristream_budgeted import (
    BGETextEncoder,
    BudgetedVeriStreamAgent,
    CoverageChangeIndex,
    CoverageChangeIndexer,
    MemoryRetriever,
    RawFrameStore,
)
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS

METHOD_VERSION = "semantic-gate-schema6"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load complete JSONL rows and discard a partial trailing write."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r+", encoding="utf-8") as handle:
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            try:
                rows.append(json.loads(raw_line))
            except json.JSONDecodeError:
                handle.seek(offset)
                handle.truncate()
                break
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _wait_for_rank_done(
    result_dir: Path,
    num_processes: int,
    timeout_seconds: float = 86400.0,
) -> None:
    """Synchronize long evaluations through rank completion files."""
    deadline = time.monotonic() + float(timeout_seconds)
    expected = [result_dir / f"rank_{rank}.done" for rank in range(num_processes)]
    while True:
        missing = [path for path in expected if not path.exists()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for rank checkpoints: "
                + ", ".join(path.name for path in missing)
            )
        time.sleep(2.0)


def _build_models(
    args: argparse.Namespace,
    device: Any,
) -> tuple[RecentWindowQAModel, RecentWindowQAModel]:
    perception = RecentWindowQAModel(
        args.perception_model_path,
        device=device,
        max_new_tokens=args.max_qa_tokens,
    )
    if args.share_model:
        return perception, perception
    reasoning = RecentWindowQAModel(
        args.reasoning_model_path or args.perception_model_path,
        device=device,
        max_new_tokens=args.max_qa_tokens,
    )
    return perception, reasoning


def _validate_pretrained_reference(value: str, label: str) -> None:
    reference = str(value).strip()
    if not reference:
        raise ValueError(f"{label} must not be empty")
    lowered = reference.lower()
    if lowered.startswith(("/path/to/", "path/to/")) or "<path" in lowered:
        raise ValueError(
            f"{label} still contains an example placeholder: {reference!r}. "
            "Use an existing local model directory or a Hugging Face repository ID."
        )
    expanded = Path(reference).expanduser()
    looks_local = expanded.is_absolute() or reference.startswith(("./", "../", "~/"))
    if looks_local and not expanded.exists():
        raise ValueError(f"{label} local path does not exist: {expanded}")


def _validate_ovo_inputs(args: argparse.Namespace) -> None:
    _validate_pretrained_reference(args.perception_model_path, "--perception-model-path")
    if args.reasoning_model_path:
        _validate_pretrained_reference(args.reasoning_model_path, "--reasoning-model-path")
    annotation = Path(args.anno_path).expanduser()
    video_directory = Path(args.chunked_dir).expanduser()
    if not annotation.is_file():
        raise ValueError(f"--anno-path is not a file: {annotation}")
    if not video_directory.is_dir():
        raise ValueError(f"--chunked-dir is not a directory: {video_directory}")
    if int(args.recent_frames) < 1:
        raise ValueError("--recent-frames must be >= 1")
    if int(args.max_qa_tokens) < 0:
        raise ValueError("--max-qa-tokens must be >= 0; use 0 for no artificial generation limit")
    if int(args.max_index_tokens) < 0:
        raise ValueError("--max-index-tokens must be >= 0; use 0 for no index safety guard")
    if args.max_samples_per_task is not None and int(args.max_samples_per_task) < 1:
        raise ValueError("--max-samples-per-task must be >= 1")
    if args.index_cache_dir and not Path(args.index_cache_dir).expanduser().is_dir():
        raise ValueError(f"--index-cache-dir is not a directory: {args.index_cache_dir}")
    if args.index_cache_dir and args.rebuild_memory:
        raise ValueError("--index-cache-dir is read-only and cannot be combined with --rebuild-memory")


def _options_text(anno: dict[str, Any]) -> str:
    return "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(anno.get("options", [])))


def _sample_rows(
    rows: list[dict[str, Any]],
    split: str,
    seed: int,
    max_per_task: int | None,
    max_per_split: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    if max_per_task is not None:
        for task in sorted({str(item["task"]) for item in rows}):
            task_rows = [item for item in rows if str(item["task"]) == task]
            random.Random(f"{seed}:{split}:{task}").shuffle(task_rows)
            selected.extend(task_rows[: int(max_per_task)])
    else:
        selected = list(rows)
    random.Random(f"{seed}:{split}:all").shuffle(selected)
    if max_per_split is not None:
        del selected[max(1, int(max_per_split)) :]
    return selected


def _parse_change_spans(value: str) -> tuple[int, ...]:
    try:
        spans = tuple(sorted({int(item.strip()) for item in str(value).split(",") if item.strip()}))
    except ValueError as exc:
        raise ValueError(f"--change-spans must be comma-separated positive integers: {value!r}") from exc
    if not spans or any(item <= 0 for item in spans):
        raise ValueError(f"--change-spans must contain positive integers: {value!r}")
    return spans


def _parse_sample_ids(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in str(value).split(",") if item.strip()))


def _validate_or_write_manifest(
    result_dir: Path,
    args: argparse.Namespace,
    process_index: int,
) -> None:
    manifest_path = result_dir / "run_config.json"
    ignored = {
        "result_dir", "output_dir", "max_samples_per_split", "max_samples", "max_videos",
        "max_questions", "max_samples_per_task", "retry_errors", "rebuild_memory",
    }
    config = {
        "method_version": METHOD_VERSION,
        "index_schema_version": CoverageChangeIndex.VERSION,
        "config": {key: value for key, value in vars(args).items() if key not in ignored},
    }
    config = json.loads(json.dumps(config, ensure_ascii=False, sort_keys=True))
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"result directory configuration differs from its run manifest: {manifest_path}. "
                "Use a new --result-dir for a different experiment."
            )
        return
    if any(result_dir.glob("rank_*.jsonl")) or (result_dir / "results_incremental.jsonl").exists():
        raise ValueError(
            f"{result_dir} contains checkpoints without a VeriStream {METHOD_VERSION} run manifest. "
            "Use a new --result-dir; do not mix answers from different method or index versions."
        )
    temporary = result_dir / f"run_config.rank{process_index}.tmp"
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


def _save_index(index: CoverageChangeIndex, path: Path) -> None:
    index.save(path)


def _memory_path(args: argparse.Namespace, video_id: str) -> tuple[Path, bool]:
    if args.index_cache_dir:
        return Path(args.index_cache_dir).expanduser() / f"{video_id}.json", True
    return Path(args.result_dir) / "memory" / f"{video_id}.json", False


def _decode_exact_current_images(
    video_path: Path,
    fps: float,
    boundary: float,
    recent_frames: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    from lib.qwen_exact_recent_decoder import fetch_recent_video_exact

    requested_end = max(0.0, float(boundary))
    decode_end = max(requested_end, 2.0 / max(float(fps), 1e-6))
    video, metadata = fetch_recent_video_exact(
        {
            "video": str(video_path),
            "fps": float(fps),
            "video_end": decode_end + 1e-4,
        },
        last_nframes=max(1, int(recent_frames)),
        return_video_metadata=True,
    )
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError(f"exact Recent decoder returned invalid tensor shape: {getattr(video, 'shape', None)}")
    raw_fps = float(metadata.get("fps", 0.0))
    raw_indices = [int(item) for item in metadata.get("frames_indices", [])]
    kept = [
        (frame, frame_index)
        for frame, frame_index in zip(video, raw_indices)
        if raw_fps <= 0 or frame_index / raw_fps <= requested_end + 1e-6
    ][-max(1, int(recent_frames)) :]
    if not kept:
        raise ValueError(f"exact Recent decoder returned no frame at causal boundary {requested_end}")
    images = [
        Image.fromarray(frame.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy(), mode="RGB")
        for frame, _ in kept
    ]
    frame_indices = [frame_index for _, frame_index in kept]
    return images, {
        "backend": str(metadata.get("video_backend", "exact_recent")),
        "frame_indices": frame_indices,
        "frame_timestamps": [item / raw_fps for item in frame_indices] if raw_fps > 0 else [],
        "full_sampled_nframes": int(metadata.get("full_sampled_nframes", len(images))),
    }


def _expected_index_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "block_seconds": float(args.history_block_seconds),
        "coverage_frames": int(args.coverage_frames_per_block),
        "change_peaks": int(args.change_peaks_per_block),
        "max_frames_per_block": int(args.max_index_frames_per_block),
        "minimum_peak_distance": float(args.minimum_peak_distance),
        "change_spans": list(args.change_spans),
        "minimum_change_distance": float(args.minimum_change_distance),
        "change_mad_scale": float(args.change_mad_scale),
        "change_normalization": "per_span_mad",
        "semantic_gate_version": 1,
        "boundary_overlap_frames": max(args.change_spans),
        "enable_proposal_repair": not args.disable_proposal_repair,
        "recent_frames": int(args.recent_frames),
        "image_embedding_model": str(args.clip_model),
        "text_embedding_model": str(args.text_embedding_model),
        "fps": float(args.fps),
        "chunk_duration": float(args.chunk_duration),
    }


def _evaluate_one(
    anno: dict[str, Any],
    perception: Any,
    reasoning: Any,
    clip: CLIPTopKFrameSelector,
    text_encoder: BGETextEncoder,
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_path = Path(args.chunked_dir) / f"{anno['id']}.mp4"
    result: dict[str, Any] = {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "ground_truth": chr(65 + int(anno["gt"])),
        "response": None,
    }
    if not video_path.exists():
        result["error"] = f"missing video: {video_path}"
        return result
    boundary = float(anno.get("realtime", 0.0))
    chunks, backend = decode_video_to_chunks_qwen(
        str(video_path), args.chunk_duration, args.fps, video_end=boundary + 1e-4
    )
    video_id = f"ovo-{anno['id']}"
    frame_store = RawFrameStore(video_id, chunks, recent_frames=args.recent_frames)
    memory_path, read_only_cache = _memory_path(args, video_id)
    expected = _expected_index_config(args)
    expected.update(
        {
            "causal_boundary": boundary,
            "history_frame_count": len(frame_store.history_records),
            "latest_frame_time": frame_store.records[-1].timestamp if frame_store.records else None,
        }
    )
    if not read_only_cache:
        expected["generation_token_limit"] = int(args.max_qa_tokens)
        expected["index_generation_token_limit"] = int(args.max_index_tokens)
        expected["assessment_prompt_version"] = 3
    loaded = False
    load_error = "cache file does not exist"
    if memory_path.exists() and not args.rebuild_memory:
        try:
            index = CoverageChangeIndex.load(memory_path, frame_store)
            loaded = index.config == expected and index.embeddings_complete()
            if not loaded:
                load_error = "index configuration mismatch or incomplete embeddings"
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            loaded = False
            load_error = f"{type(exc).__name__}: {exc}"
    if not loaded:
        if read_only_cache:
            raise RuntimeError(f"read-only index cache is unavailable or incompatible: {memory_path}: {load_error}")
        index = CoverageChangeIndexer(
            perception,
            clip,
            text_encoder,
            block_seconds=args.history_block_seconds,
            coverage_frames=args.coverage_frames_per_block,
            change_peaks=args.change_peaks_per_block,
            max_frames_per_block=args.max_index_frames_per_block,
            minimum_peak_distance=args.minimum_peak_distance,
            change_spans=args.change_spans,
            minimum_change_distance=args.minimum_change_distance,
            change_mad_scale=args.change_mad_scale,
            enable_proposal_repair=not args.disable_proposal_repair,
            max_generation_tokens=args.max_index_tokens,
        ).build(frame_store)
        index.config.update(expected)
        _save_index(index, memory_path)
        perception_index_calls = index.index_perception_calls
    else:
        perception_index_calls = 0
    retriever = MemoryRetriever(
        index,
        text_encoder,
        clip,
        candidate_metadata_tokens=args.candidate_metadata_tokens,
        redundancy_weight=args.redundancy_weight,
        minimum_semantic_similarity=args.minimum_semantic_similarity,
        minimum_visual_similarity=args.minimum_visual_similarity,
        minimum_bm25_score=args.minimum_bm25_score,
        use_bge=not args.disable_bge,
        use_bm25=not args.disable_bm25,
        use_clip_fallback=not args.disable_clip_fallback,
        use_complementary_selection=not args.disable_complementary_selection,
    )
    agent = BudgetedVeriStreamAgent(
        perception,
        reasoning,
        frame_store,
        index,
        retriever,
        clip,
        max_tool_actions=args.max_tool_actions,
        max_history_visual_frames=args.max_history_visual_frames,
        history_context_tokens=args.history_context_tokens,
        controller_mode=args.controller_mode,
        max_search_rounds=args.max_search_rounds,
        max_navigation_steps=args.max_navigation_steps,
        max_visual_actions=args.max_visual_actions,
        current_fast_path_confidence=args.current_fast_path_confidence,
        visual_verification_policy=args.visual_verification_policy,
    )
    if args.current_lane_decoder == "exact":
        current_images, current_metadata = _decode_exact_current_images(
            video_path, args.fps, boundary, args.recent_frames
        )
    else:
        current_images = frame_store.recent_images()
        current_metadata = {
            "backend": "history_tail",
            "frame_indices": [],
            "frame_timestamps": [item.timestamp for item in frame_store.current_records],
            "full_sampled_nframes": len(frame_store.records),
        }
    response, trace = agent.answer(
        str(anno["question"]),
        _options_text(anno),
        baseline_prompt=build_ovo_prompt(str(anno["task"]), anno),
        current_images=current_images,
    )
    result.update(
        {
            "response": response,
            "decode_backend": backend,
            "causal_chunk_count": len(chunks),
            "history_frame_count": len(frame_store.history_records),
            "recent_frame_count": len(frame_store.current_records),
            "current_lane_frame_count": len(current_images),
            "current_lane_decode_backend": current_metadata["backend"],
            "current_lane_frame_indices": current_metadata["frame_indices"],
            "current_lane_frame_timestamps": current_metadata["frame_timestamps"],
            "index_cache_path": str(memory_path),
            "index_cache_read_only": read_only_cache,
            "memory_overview_count": len(index.overviews),
            "memory_transition_count": len(index.transitions),
            "change_proposal_count": index.change_proposal_count,
            "initially_assessed_proposal_count": index.initially_assessed_proposal_count,
            "assessed_proposal_count": index.assessed_proposal_count,
            "accepted_proposal_count": index.accepted_proposal_count,
            "searchable_transition_count": index.searchable_transition_count,
            "repaired_proposal_count": index.repaired_proposal_count,
            "unresolved_proposal_count": index.unresolved_proposal_count,
            "proposal_repair_calls": index.proposal_repair_calls,
            "index_generated_tokens": index.index_generated_tokens,
            "index_primary_generated_tokens": index.index_primary_generated_tokens,
            "index_repair_generated_tokens": index.index_repair_generated_tokens,
            "index_eos_stop_count": index.index_eos_stop_count,
            "index_generation_context_limit_hits": index.index_generation_context_limit_hits,
            "index_json_complete_stop_count": index.index_json_complete_stop_count,
            "overview_initial_parse_success_count": sum(
                item.initial_response_parse_success for item in index.overviews.values()
            ),
            "proposal_block_count": sum(
                item.change_proposal_count > 0 for item in index.overviews.values()
            ),
            "initially_complete_proposal_block_count": sum(
                item.change_proposal_count > 0
                and item.initially_assessed_proposal_count == item.change_proposal_count
                for item in index.overviews.values()
            ),
            "perception_index_calls": perception_index_calls,
            "trace": trace.to_dict(),
        }
    )
    return result


def _merge_and_score(result_dir: Path, num_processes: int, config: dict[str, Any]) -> None:
    grouped_by_key: dict[str, dict[tuple[str, str], dict[str, Any]]] = {"backward": {}, "realtime": {}}
    for rank in range(num_processes):
        for row in _load_jsonl(result_dir / f"rank_{rank}.jsonl"):
            if row.get("task") in BACKWARD_TASKS:
                split = "backward"
            elif row.get("task") in REAL_TIME_TASKS:
                split = "realtime"
            else:
                continue
            key = (str(row.get("id")), str(row.get("task")))
            grouped_by_key[split][key] = row
    groups = {
        split: sorted(rows.values(), key=lambda item: int(item.get("id", 0)))
        for split, rows in grouped_by_key.items()
    }
    output = result_dir / "budgeted_ovo_backward_realtime.json"
    output.write_text(json.dumps({"config": config, **groups}, ensure_ascii=False, indent=2), encoding="utf-8")
    scores = calculate_ovo_scores(groups["backward"], groups["realtime"], [])
    all_rows = [*groups["backward"], *groups["realtime"]]
    proposed = sum(int(item.get("change_proposal_count", 0)) for item in all_rows)
    initially_assessed = sum(int(item.get("initially_assessed_proposal_count", 0)) for item in all_rows)
    assessed = sum(int(item.get("assessed_proposal_count", 0)) for item in all_rows)
    repaired = sum(int(item.get("repaired_proposal_count", 0)) for item in all_rows)
    initially_invalid = max(0, proposed - initially_assessed)
    trace_needs = [
        need
        for row in all_rows
        for need in row.get("trace", {}).get("needs", [])
        if isinstance(need, dict)
    ]
    scores["veristream_diagnostics"] = {
        "question_count": len(all_rows),
        "error_count": sum(bool(item.get("error")) for item in all_rows),
        "read_only_cache_hit_count": sum(bool(item.get("index_cache_read_only")) for item in all_rows),
        "index_perception_call_count": sum(int(item.get("perception_index_calls", 0)) for item in all_rows),
        "index_generated_token_count": sum(int(item.get("index_generated_tokens", 0)) for item in all_rows),
        "index_primary_generated_token_count": sum(
            int(item.get("index_primary_generated_tokens", 0)) for item in all_rows
        ),
        "index_repair_generated_token_count": sum(
            int(item.get("index_repair_generated_tokens", 0)) for item in all_rows
        ),
        "index_eos_stop_count": sum(int(item.get("index_eos_stop_count", 0)) for item in all_rows),
        "index_generation_context_limit_hit_count": sum(
            int(item.get("index_generation_context_limit_hits", 0)) for item in all_rows
        ),
        "index_json_complete_stop_count": sum(
            int(item.get("index_json_complete_stop_count", 0)) for item in all_rows
        ),
        "overview_count": sum(int(item.get("memory_overview_count", 0)) for item in all_rows),
        "overview_initial_parse_success_count": sum(
            int(item.get("overview_initial_parse_success_count", 0)) for item in all_rows
        ),
        "proposal_block_count": sum(int(item.get("proposal_block_count", 0)) for item in all_rows),
        "initially_complete_proposal_block_count": sum(
            int(item.get("initially_complete_proposal_block_count", 0)) for item in all_rows
        ),
        "exact_current_lane_count": sum(
            str(item.get("current_lane_decode_backend", "")).endswith("exact_recent") for item in all_rows
        ),
        "exact_recent4_count": sum(
            str(item.get("current_lane_decode_backend", "")).endswith("exact_recent")
            and int(item.get("current_lane_frame_count", 0)) == 4
            for item in all_rows
        ),
        "neighbor_event_need_count": sum(item.get("need_type") == "neighbor_event" for item in trace_needs),
        "l1_text_admissible_need_count": sum(
            item.get("visual_policy_reason") == "l1_text_admissible" for item in trace_needs
        ),
        "deterministic_visual_need_count": sum(
            str(item.get("visual_policy_reason", "")).startswith("visual_type:") for item in trace_needs
        ),
        "visual_policy_override_count": sum(
            item.get("model_visual_verification") is not None
            and bool(item.get("model_visual_verification")) != bool(item.get("visual_verification"))
            for item in trace_needs
        ),
        "proposal_count": proposed,
        "initially_assessed_count": initially_assessed,
        "initially_invalid_count": initially_invalid,
        "assessed_count": assessed,
        "repaired_assessment_count": repaired,
        "unresolved_count": sum(int(item.get("unresolved_proposal_count", 0)) for item in all_rows),
        "repair_call_count": sum(int(item.get("proposal_repair_calls", 0)) for item in all_rows),
        "accepted_change_count": sum(int(item.get("accepted_proposal_count", 0)) for item in all_rows),
        "searchable_transition_count": sum(int(item.get("searchable_transition_count", 0)) for item in all_rows),
        "assessment_completion_rate": assessed / proposed if proposed else 1.0,
        "initial_assessment_completion_rate": initially_assessed / proposed if proposed else 1.0,
        "overview_initial_parse_success_rate": (
            sum(int(item.get("overview_initial_parse_success_count", 0)) for item in all_rows)
            / sum(int(item.get("memory_overview_count", 0)) for item in all_rows)
            if sum(int(item.get("memory_overview_count", 0)) for item in all_rows) else 1.0
        ),
        "initially_complete_proposal_block_rate": (
            sum(int(item.get("initially_complete_proposal_block_count", 0)) for item in all_rows)
            / sum(int(item.get("proposal_block_count", 0)) for item in all_rows)
            if sum(int(item.get("proposal_block_count", 0)) for item in all_rows) else 1.0
        ),
        "repair_recovery_rate": repaired / initially_invalid if initially_invalid else None,
        "tool_budget_exhausted_count": sum(
            bool(item.get("trace", {}).get("budget_exhausted")) for item in all_rows
        ),
    }
    (result_dir / "scores_backward_realtime.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_ovo_results("VeriStream/Qwen3-VL", groups["backward"], groups["realtime"], [])
    print(json.dumps(scores["veristream_diagnostics"], ensure_ascii=False, indent=2))
    print(f"Results saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Budgeted VeriStream OVO Backward/Realtime evaluation")
    parser.add_argument("--perception-model-path", required=True)
    parser.add_argument("--reasoning-model-path", default="")
    parser.add_argument("--anno-path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked-dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument(
        "--index-cache-dir", default="",
        help="Read-only directory containing existing ovo-<id>.json index files and embedding sidecars.",
    )
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--max-qa-tokens",
        type=int,
        default=0,
        help="Per-call generation limit; 0 removes the artificial limit and uses remaining model context.",
    )
    parser.add_argument(
        "--max-index-tokens",
        type=int,
        default=2048,
        help="Safety guard for index and proposal-repair calls only; 0 disables it.",
    )
    parser.add_argument("--recent-frames", type=int, default=4)
    parser.add_argument("--history-block-seconds", type=float, default=12.0)
    parser.add_argument("--coverage-frames-per-block", type=int, default=4)
    parser.add_argument("--change-peaks-per-block", type=int, default=2)
    parser.add_argument("--max-index-frames-per-block", type=int, default=8)
    parser.add_argument("--minimum-peak-distance", type=float, default=2.0)
    parser.add_argument("--change-spans", default="1,2,4")
    parser.add_argument("--minimum-change-distance", type=float, default=0.05)
    parser.add_argument("--change-mad-scale", type=float, default=0.5)
    parser.add_argument(
        "--disable-proposal-repair", action="store_true",
        help="Do not retry missing or invalid change-proposal assessments during index construction.",
    )
    parser.add_argument(
        "--max-tool-actions", type=int, default=4,
        help="Maximum executed search/inspect/expand/compare actions per question; finish is free.",
    )
    parser.add_argument(
        "--controller-mode", choices=["llm", "deterministic"], default="llm",
        help="LLM autonomously selects retrieval tools; deterministic is the controller ablation.",
    )
    parser.add_argument("--max-search-rounds", type=int, default=2)
    parser.add_argument("--max-navigation-steps", type=int, default=2)
    parser.add_argument("--max-visual-actions", type=int, default=2)
    parser.add_argument("--current-fast-path-confidence", type=float, default=0.7)
    parser.add_argument(
        "--visual-verification-policy", choices=["deterministic", "planner"], default="deterministic"
    )
    parser.add_argument(
        "--current-lane-decoder", choices=["exact", "history_tail"], default="exact"
    )
    parser.add_argument("--max-history-visual-frames", type=int, default=8)
    parser.add_argument("--candidate-metadata-tokens", type=int, default=768)
    parser.add_argument("--history-context-tokens", type=int, default=768)
    parser.add_argument("--redundancy-weight", type=float, default=0.25)
    parser.add_argument("--minimum-semantic-similarity", type=float, default=0.25)
    parser.add_argument("--minimum-visual-similarity", type=float, default=0.20)
    parser.add_argument("--minimum-bm25-score", type=float, default=0.10)
    parser.add_argument("--disable-bge", action="store_true")
    parser.add_argument("--disable-bm25", action="store_true")
    parser.add_argument("--disable-clip-fallback", action="store_true")
    parser.add_argument("--disable-complementary-selection", action="store_true")
    parser.add_argument("--text-embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--text-embedding-device", default="auto")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument(
        "--max-samples-per-task", type=int, default=None,
        help="Take a deterministic prefix of this size independently for every task.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Optional comma-separated annotation IDs for targeted smoke tests.",
    )
    parser.add_argument("--eval-splits", nargs="+", choices=["backward", "realtime"], default=["backward", "realtime"])
    parser.add_argument("--rebuild-memory", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--share-model", action="store_true")
    args = parser.parse_args()
    try:
        args.change_spans = _parse_change_spans(args.change_spans)
        args.sample_ids = _parse_sample_ids(args.sample_ids)
        _validate_ovo_inputs(args)
    except ValueError as exc:
        parser.error(str(exc))

    accelerator = Accelerator()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_manifest(result_dir, args, accelerator.process_index)
    annotations = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    if args.sample_ids:
        requested_ids = set(args.sample_ids)
        annotations = [item for item in annotations if str(item.get("id")) in requested_ids]
        found_ids = {str(item.get("id")) for item in annotations}
        if found_ids != requested_ids:
            missing = sorted(requested_ids - found_ids)
            raise ValueError(f"--sample-ids are absent from annotations: {missing}")
    groups = {
        "backward": [item for item in annotations if item["task"] in BACKWARD_TASKS],
        "realtime": [item for item in annotations if item["task"] in REAL_TIME_TASKS],
    }
    groups = {
        split: _sample_rows(
            rows, split, args.sample_seed, args.max_samples_per_task, args.max_samples_per_split
        )
        for split, rows in groups.items()
    }
    local_groups = {
        name: rows[accelerator.process_index :: accelerator.num_processes]
        for name, rows in groups.items()
        if name in args.eval_splits
    }
    perception, reasoning = _build_models(args, accelerator.device)
    clip_device = accelerator.device if str(args.clip_device).lower() == "auto" else args.clip_device
    text_device = (
        accelerator.device
        if str(args.text_embedding_device).lower() == "auto"
        else args.text_embedding_device
    )
    clip = CLIPTopKFrameSelector(args.clip_model, clip_device, args.clip_batch_size)
    text_encoder = BGETextEncoder(args.text_embedding_model, text_device)
    checkpoint = result_dir / f"rank_{accelerator.process_index}.jsonl"
    done_path = result_dir / f"rank_{accelerator.process_index}.done"
    done_path.unlink(missing_ok=True)
    completed = {
        str(item.get("id")) + ":" + str(item.get("task"))
        for item in _load_jsonl(checkpoint)
        if not (args.retry_errors and item.get("error"))
    }
    for split in args.eval_splits:
        iterator = tqdm(
            local_groups[split],
            desc=f"Budgeted rank{accelerator.process_index} {split}",
            disable=accelerator.process_index != 0,
        )
        for anno in iterator:
            key = str(anno["id"]) + ":" + str(anno["task"])
            if key in completed:
                continue
            try:
                row = _evaluate_one(anno, perception, reasoning, clip, text_encoder, args)
            except Exception as exc:
                row = {
                    "id": anno["id"], "video": anno.get("video"), "task": anno["task"],
                    "question": anno.get("question"), "ground_truth": chr(65 + int(anno["gt"])),
                    "response": None, "error": f"{type(exc).__name__}: {exc}",
                }
            _append_jsonl(checkpoint, row)
            completed.add(key)
    done_path.write_text("done\n", encoding="utf-8")
    if not accelerator.is_main_process:
        return
    _wait_for_rank_done(result_dir, accelerator.num_processes)
    _merge_and_score(result_dir, accelerator.num_processes, {**vars(args), "num_processes": accelerator.num_processes})


if __name__ == "__main__":
    main()
