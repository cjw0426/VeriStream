"""Evaluate budgeted VeriStream on the OVO Forward split."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.clip_topk_selector import CLIPTopKFrameSelector
from lib.recent_window_eval import build_ovo_prompt, calculate_ovo_scores, decode_video_to_chunks_qwen, print_ovo_results
from lib.veristream_budgeted import (
    BGETextEncoder,
    BudgetedVeriStreamAgent,
    CoverageChangeIndex,
    CoverageChangeIndexer,
    MemoryRetriever,
    RawFrameStore,
)
from main_experiments.eval_qwen3vl_ovo_budgeted import (
    _append_jsonl,
    _build_models,
    _expected_index_config,
    _load_jsonl,
    _parse_change_spans,
    _validate_or_write_manifest,
    _validate_ovo_inputs,
    _wait_for_rank_done,
)
from ovo_constants import FORWARD_TASKS


def _evaluate_sample(
    anno: dict[str, Any],
    perception: Any,
    reasoning: Any,
    clip: CLIPTopKFrameSelector,
    text_encoder: BGETextEncoder,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = copy.deepcopy(anno)
    for item_index, test_info in enumerate(result.get("test_info", [])):
        video_path = Path(args.chunked_dir) / f"{anno['id']}_{item_index}.mp4"
        test_info["response"] = None
        if not video_path.exists():
            test_info["error"] = f"missing video: {video_path}"
            continue
        chunks, backend = decode_video_to_chunks_qwen(str(video_path), args.chunk_duration, args.fps)
        video_id = f"ovo-{anno['id']}-forward-{item_index}"
        frame_store = RawFrameStore(video_id, chunks, args.recent_frames)
        expected_config = _expected_index_config(args)
        expected_config.update(
            {
                "causal_boundary": frame_store.records[-1].timestamp if frame_store.records else None,
                "history_frame_count": len(frame_store.history_records),
                "latest_frame_time": frame_store.records[-1].timestamp if frame_store.records else None,
            }
        )
        memory_path = Path(args.result_dir) / "memory" / f"{video_id}.json"
        loaded = False
        if memory_path.exists() and not args.rebuild_memory:
            try:
                index = CoverageChangeIndex.load(memory_path, frame_store)
                loaded = index.config == expected_config and index.embeddings_complete()
            except (ValueError, OSError, json.JSONDecodeError):
                loaded = False
        if not loaded:
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
            ).build(frame_store)
            index.config.update(expected_config)
            index.save(memory_path)
            index_calls = index.index_perception_calls
        else:
            index_calls = 0
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
        )
        prompt = build_ovo_prompt(anno["task"], anno, index=item_index)
        response, trace = agent.answer(prompt, baseline_prompt=prompt)
        test_info.update(
            {
                "response": response,
                "decode_backend": backend,
                "history_frame_count": len(frame_store.history_records),
                "recent_frame_count": len(frame_store.current_records),
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
                "perception_index_calls": index_calls,
                "trace": trace.to_dict(),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Budgeted VeriStream OVO Forward evaluation")
    parser.add_argument("--perception-model-path", required=True)
    parser.add_argument("--reasoning-model-path", default="")
    parser.add_argument("--anno-path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked-dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--max-qa-tokens", type=int, default=0,
        help="Per-call generation limit; 0 removes the artificial limit and uses remaining model context.",
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
    parser.add_argument("--controller-mode", choices=["llm", "deterministic"], default="llm")
    parser.add_argument("--max-search-rounds", type=int, default=2)
    parser.add_argument("--max-navigation-steps", type=int, default=2)
    parser.add_argument("--max-visual-actions", type=int, default=2)
    parser.add_argument("--current-fast-path-confidence", type=float, default=0.7)
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rebuild-memory", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--share-model", action="store_true")
    args = parser.parse_args()
    try:
        args.change_spans = _parse_change_spans(args.change_spans)
        _validate_ovo_inputs(args)
    except ValueError as exc:
        parser.error(str(exc))

    accelerator = Accelerator()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_manifest(result_dir, args, accelerator.process_index)
    annotations = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    rows = [item for item in annotations if item.get("task") in FORWARD_TASKS]
    random.seed(42)
    random.shuffle(rows)
    if args.max_samples is not None:
        rows = rows[: max(1, args.max_samples)]
    local_rows = rows[accelerator.process_index :: accelerator.num_processes]
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
        if not (args.retry_errors and any(test.get("error") for test in item.get("test_info", [])))
    }
    iterator = tqdm(local_rows, desc=f"Budgeted rank{accelerator.process_index} forward", disable=accelerator.process_index != 0)
    for anno in iterator:
        key = str(anno["id"]) + ":" + str(anno["task"])
        if key in completed:
            continue
        try:
            row = _evaluate_sample(anno, perception, reasoning, clip, text_encoder, args)
        except Exception as exc:
            row = copy.deepcopy(anno)
            for test_info in row.get("test_info", []):
                test_info["response"] = None
                test_info["error"] = f"{type(exc).__name__}: {exc}"
        _append_jsonl(checkpoint, row)
        completed.add(key)
    done_path.write_text("done\n", encoding="utf-8")
    if not accelerator.is_main_process:
        return
    _wait_for_rank_done(result_dir, accelerator.num_processes)
    merged_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for rank in range(accelerator.num_processes):
        for row in _load_jsonl(result_dir / f"rank_{rank}.jsonl"):
            merged_by_key[(str(row.get("id")), str(row.get("task")))] = row
    merged = list(merged_by_key.values())
    merged.sort(key=lambda item: int(item.get("id", 0)))
    output = result_dir / "budgeted_ovo_forward.json"
    output.write_text(json.dumps({"config": vars(args), "forward": merged}, ensure_ascii=False, indent=2), encoding="utf-8")
    scores = calculate_ovo_scores([], [], merged)
    (result_dir / "scores_forward.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print_ovo_results("VeriStream/Qwen3-VL", [], [], merged)
    print(f"Forward results saved to {output}")


if __name__ == "__main__":
    main()
