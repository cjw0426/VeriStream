"""Causal incremental StreamingBench evaluation for budgeted VeriStream."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.clip_topk_selector import CLIPTopKFrameSelector
from lib.recent_window_eval import RecentWindowQAModel, decode_video_to_chunks_qwen, extract_mcq_answer, save_json
from lib.veristream_budgeted import (
    BGETextEncoder,
    BudgetedVeriStreamAgent,
    CoverageChangeIndex,
    CoverageChangeIndexer,
    MemoryRetriever,
    RawFrameStore,
)
from main_experiments.eval_qwen3vl_ovo_budgeted import _validate_or_write_manifest


RECENT_PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with some frames from the video and a multiple-choice question. "
    "Your task is to analyze the video and provide the best answer.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Only give the best option's letter (A, B, C, or D) directly."
)


def timestamp_to_seconds(timestamp: str) -> int:
    parts = str(timestamp).split(":")
    return sum(int(value) * 60**index for index, value in enumerate(reversed(parts)))


def resolve_video_path(video_path: str, video_dir: str) -> str:
    if os.path.isabs(video_path):
        return video_path
    return os.path.join(video_dir, os.path.basename(video_path))


def format_options(options: list[str]) -> str:
    return "\n".join(
        text if (text := str(option).strip()).startswith(("A.", "B.", "C.", "D.")) else f"{chr(65 + index)}. {text}"
        for index, option in enumerate(options)
    )


def parse_change_spans(value: str) -> tuple[int, ...]:
    spans = tuple(sorted({int(item.strip()) for item in str(value).split(",") if item.strip()}))
    if not spans or any(item <= 0 for item in spans):
        raise ValueError("--change-spans must contain comma-separated positive integers")
    return spans


def make_key(video_name: str, question: dict[str, Any]) -> str:
    return f"{video_name}:{question.get('time_stamp', '')}:{question.get('question', '')}"


def load_jsonl(path: Path, retry_errors: bool = False) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows_by_key: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            rows_by_key[str(row.get("_key"))] = row
    rows = list(rows_by_key.values())
    completed = {
        key for key, row in rows_by_key.items()
        if not (retry_errors and row.get("error"))
    }
    return rows, completed


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    totals = {"index_perception_calls": 0, "online_perception_calls": 0, "controller_calls": 0, "history_visual_frames": 0}
    for record in records:
        task = str(record.get("task_type", "unknown"))
        by_type[task]["total"] += 1
        by_type[task]["correct"] += int(bool(record.get("correct")))
        totals["index_perception_calls"] += int(record.get("index_perception_calls", 0))
        trace = record.get("trace", {})
        totals["online_perception_calls"] += int(trace.get("perception_calls", 0))
        totals["controller_calls"] += int(trace.get("controller_calls", 0))
        totals["history_visual_frames"] += int(trace.get("history_visual_frames", 0))
    correct = sum(item["correct"] for item in by_type.values())
    return {
        "overall": {"total": len(records), "correct": correct, "accuracy": 100.0 * correct / len(records) if records else 0.0},
        "by_task_type": [
            {"task_type": key, **value, "accuracy": 100.0 * value["correct"] / value["total"]}
            for key, value in sorted(by_type.items())
        ],
        "cost": totals,
    }


def run(args: argparse.Namespace) -> None:
    entries = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    videos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, str] = {}
    for entry in entries:
        raw_path = str(entry["video_path"])
        categories[raw_path] = str(entry.get("video_categories", ""))
        videos[raw_path].extend(entry.get("questions", []))
    selected = list(videos.items())[: args.max_videos] if args.max_videos else list(videos.items())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_manifest(output_dir, args, 0)
    qa = RecentWindowQAModel(args.qa_model, args.qa_device, args.max_qa_tokens)
    clip = CLIPTopKFrameSelector(args.clip_model, args.clip_device, args.clip_batch_size)
    text_encoder = BGETextEncoder(args.text_embedding_model, args.text_embedding_device)
    indexer = CoverageChangeIndexer(
        qa,
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
    )
    checkpoint = output_dir / "results_incremental.jsonl"
    records, completed = load_jsonl(checkpoint, retry_errors=args.retry_errors)
    record_positions = {str(row.get("_key")): index for index, row in enumerate(records)}
    processed = 0
    with checkpoint.open("a", encoding="utf-8") as checkpoint_file:
        for raw_path, questions in selected:
            video_path = resolve_video_path(raw_path, args.video_dir)
            if not os.path.exists(video_path):
                continue
            chunks, backend = decode_video_to_chunks_qwen(video_path, args.chunk_duration, args.fps)
            questions = sorted(questions, key=lambda item: timestamp_to_seconds(item["time_stamp"]))
            video_id = Path(video_path).stem
            previous_index: CoverageChangeIndex | None = None
            index_calls_total = 0
            for question in questions:
                boundary = float(timestamp_to_seconds(question["time_stamp"]))
                available = [item for item in chunks if float(item.end_time) <= boundary + 1e-4]
                frame_store = RawFrameStore(video_id, available, args.recent_frames)
                index = indexer.build(frame_store, reuse_index=previous_index)
                index.config.update(
                    {
                        "fps": float(args.fps),
                        "chunk_duration": float(args.chunk_duration),
                        "causal_boundary": boundary,
                        "history_frame_count": len(frame_store.history_records),
                        "latest_frame_time": frame_store.records[-1].timestamp if frame_store.records else None,
                    }
                )
                previous_index = index
                index_calls_total += index.index_perception_calls
                key = make_key(Path(video_path).name, question)
                if key in completed:
                    continue
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
                    qa,
                    qa,
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
                try:
                    question_text = str(question.get("question", ""))
                    options_text = format_options(question.get("options", []))
                    baseline_prompt = RECENT_PROMPT_TEMPLATE.format(question=question_text, options=options_text)
                    response, trace = agent.answer(
                        question_text,
                        options_text,
                        baseline_prompt=baseline_prompt,
                    )
                    expected = extract_mcq_answer(str(question.get("answer", ""))) or str(question.get("answer", "")).strip().upper()
                    predicted = extract_mcq_answer(response)
                    row = {
                        "_key": key,
                        "video": Path(video_path).name,
                        "video_category": categories.get(raw_path, ""),
                        "time_stamp": question["time_stamp"],
                        "task_type": question.get("task_type", ""),
                        "question": question.get("question", ""),
                        "answer_gt": expected,
                        "response": response,
                        "correct": bool(predicted and predicted == expected),
                        "decode_backend": backend,
                        "causal_chunk_count": len(available),
                        "history_frame_count": len(frame_store.history_records),
                        "recent_frame_count": len(frame_store.current_records),
                        "overview_count": len(index.overviews),
                        "transition_count": len(index.transitions),
                        "change_proposal_count": index.change_proposal_count,
                        "initially_assessed_proposal_count": index.initially_assessed_proposal_count,
                        "assessed_proposal_count": index.assessed_proposal_count,
                        "accepted_proposal_count": index.accepted_proposal_count,
                        "searchable_transition_count": index.searchable_transition_count,
                        "repaired_proposal_count": index.repaired_proposal_count,
                        "unresolved_proposal_count": index.unresolved_proposal_count,
                        "proposal_repair_calls": index.proposal_repair_calls,
                        "index_perception_calls": index.index_perception_calls,
                        "cumulative_index_perception_calls": index_calls_total,
                        "trace": trace.to_dict(),
                    }
                except Exception as exc:
                    row = {
                        "_key": key, "video": Path(video_path).name, "time_stamp": question["time_stamp"],
                        "task_type": question.get("task_type", ""), "question": question.get("question", ""),
                        "answer_gt": str(question.get("answer", "")), "response": None, "correct": False,
                        "error": f"{type(exc).__name__}: {exc}", "index_perception_calls": index.index_perception_calls,
                    }
                if key in record_positions:
                    records[record_positions[key]] = row
                else:
                    record_positions[key] = len(records)
                    records.append(row)
                completed.add(key)
                checkpoint_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                checkpoint_file.flush()
                processed += 1
                if args.max_questions and processed >= args.max_questions:
                    break
            if previous_index is not None:
                previous_index.save(output_dir / "memory" / f"{video_id}.json")
            if args.max_questions and processed >= args.max_questions:
                break
    summary = summarize(records)
    payload = {"config": vars(args), "summary": summary, "results": records}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json(output_dir / f"budgeted_results_{timestamp}.json", payload)
    save_json(output_dir / "scores_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Budgeted VeriStream StreamingBench evaluation")
    parser.add_argument("--anno-path", default="data/streamingbench/questions_real.json")
    parser.add_argument("--video-dir", default="data/streamingbench/videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qa-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--qa-device", default="auto")
    parser.add_argument(
        "--max-qa-tokens", type=int, default=0,
        help="Per-call generation limit; 0 removes the artificial limit and uses remaining model context.",
    )
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
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
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    args.change_spans = parse_change_spans(args.change_spans)
    run(args)


if __name__ == "__main__":
    main()
