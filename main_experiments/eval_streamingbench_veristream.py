"""VeriStream 在 StreamingBench 上的因果流式评测入口。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.clip_topk_selector import CLIPTopKFrameSelector
from lib.recent_window_eval import (
    RecentWindowQAModel,
    decode_video_to_chunks_qwen,
    extract_mcq_answer,
    save_json,
)
from lib.veristream import ToolTrace, VeriStreamAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with video evidence and a multiple-choice question. "
    "Return only the best option's letter (A, B, C, or D).\n\n"
    "Question: {question}\n\nOptions:\n{options}\n"
)


def timestamp_to_seconds(timestamp: str) -> int:
    parts = str(timestamp).split(":")
    return sum(int(value) * 60**index for index, value in enumerate(reversed(parts)))


def build_prompt(question: dict[str, Any]) -> str:
    options = []
    for index, option in enumerate(question.get("options", [])):
        text = str(option).strip()
        options.append(text if text.startswith(("A.", "B.", "C.", "D.")) else f"{chr(65 + index)}. {text}")
    return PROMPT_TEMPLATE.format(question=question.get("question", ""), options="\n".join(options))


def resolve_video_path(video_path: str, video_dir: str) -> str:
    if os.path.isabs(video_path):
        return video_path
    return os.path.join(video_dir, os.path.basename(video_path))


class CLIPNovelty:
    """把相邻帧的 CLIP 余弦距离暴露为 VeriStream 的新颖性函数。"""

    def __init__(self, selector: CLIPTopKFrameSelector) -> None:
        self.selector = selector
        self.previous: torch.Tensor | None = None
        self.frames_scanned = 0

    def __call__(self, _previous_frame: Any, current_frame: Any) -> float:
        embedding = self.selector.image_embeddings([current_frame])[0]
        self.frames_scanned += 1
        if self.previous is None:
            self.previous = embedding
            return 1.0
        similarity = float(torch.dot(self.previous, embedding).clamp(-1, 1).item())
        self.previous = embedding
        return 1.0 - similarity


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    total_calls = total_frames = total_clip_frames = 0
    for record in records:
        kind = str(record.get("task_type", "unknown"))
        by_type[kind]["total"] += 1
        by_type[kind]["correct"] += int(bool(record.get("correct")))
        trace = record.get("trace", {})
        total_calls += int(trace.get("qwen_calls", 0))
        total_frames += int(trace.get("qwen_frames", 0))
        total_clip_frames += int(record.get("clip_frames_scanned", 0))
    return {
        "overall": {
            "total": len(records),
            "correct": sum(row["correct"] for row in by_type.values()),
            "accuracy": 100.0 * sum(row["correct"] for row in by_type.values()) / len(records) if records else 0.0,
        },
        "by_task_type": [
            {"task_type": kind, **row, "accuracy": 100.0 * row["correct"] / row["total"]}
            for kind, row in sorted(by_type.items())
        ],
        "cost": {
            "qwen_calls": total_calls,
            "qwen_frames": total_frames,
            "clip_frames": total_clip_frames,
        },
    }


def run(args: argparse.Namespace) -> None:
    entries = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    videos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, str] = {}
    for entry in entries:
        raw_path = str(entry["video_path"])
        categories[raw_path] = str(entry.get("video_categories", ""))
        videos[raw_path].extend(entry.get("questions", []))
    selected_videos = list(videos.items())[: args.max_videos] if args.max_videos else list(videos.items())
    qa = RecentWindowQAModel(model_name=args.qa_model, device=args.qa_device, max_new_tokens=args.max_qa_tokens)
    selector = None
    if args.novelty_backend == "clip":
        selector = CLIPTopKFrameSelector(args.clip_model, device=args.clip_device, batch_size=args.clip_batch_size)

    records: list[dict[str, Any]] = []
    memory_root = Path(args.output_dir) / "memory"
    for video_number, (raw_path, questions) in enumerate(selected_videos, start=1):
        video_path = resolve_video_path(raw_path, args.video_dir)
        if not os.path.exists(video_path):
            logger.warning("缺少视频，跳过：%s", video_path)
            continue
        chunks, decode_backend = decode_video_to_chunks_qwen(video_path, args.chunk_duration, args.fps)
        questions = sorted(questions, key=lambda item: timestamp_to_seconds(item["time_stamp"]))
        novelty = CLIPNovelty(selector) if selector else None
        agent = VeriStreamAgent(
            qa=qa,
            video_id=Path(video_path).stem,
            recent_frames=args.recent_frames,
            max_cards=args.max_cards,
            anchor_interval=args.anchor_interval,
            novelty_threshold=args.novelty_threshold,
            max_actions=args.max_actions,
            novelty_fn=novelty,
        )
        ingested = 0
        logger.info("[%d/%d] 流式处理 %s，问题数=%d", video_number, len(selected_videos), Path(video_path).name, len(questions))
        for question in questions:
            boundary = float(timestamp_to_seconds(question["time_stamp"]))
            available = [chunk for chunk in chunks[ingested:] if float(chunk.end_time) <= boundary + 1e-4]
            trace = ToolTrace()
            clip_before = novelty.frames_scanned if novelty else 0
            agent.ingest(available, trace)
            ingested += len(available)
            response, answer_trace = agent.answer(build_prompt(question))
            answer_trace.actions = trace.actions + answer_trace.actions
            answer_trace.qwen_calls += trace.qwen_calls
            answer_trace.qwen_frames += trace.qwen_frames
            expected = extract_mcq_answer(str(question.get("answer", ""))) or str(question.get("answer", "")).strip().upper()
            predicted = extract_mcq_answer(response)
            records.append(
                {
                    "video": Path(video_path).name,
                    "video_category": categories.get(raw_path, ""),
                    "time_stamp": question["time_stamp"],
                    "task_type": question.get("task_type", ""),
                    "question": question.get("question", ""),
                    "answer_gt": expected,
                    "response": response,
                    "correct": bool(predicted and predicted == expected),
                    "decode_backend": decode_backend,
                    "causal_chunk_count": ingested,
                    "memory_card_count": len(agent.store.cards),
                    "clip_frames_scanned": (novelty.frames_scanned - clip_before) if novelty else 0,
                    "trace": answer_trace.to_dict(),
                }
            )
            if args.max_questions and len(records) >= args.max_questions:
                break
        agent.store.save(memory_root / f"{Path(video_path).stem}.json")
        if args.max_questions and len(records) >= args.max_questions:
            break
    summary = summarize(records)
    output = {"config": vars(args), "summary": summary, "results": records}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json(Path(args.output_dir) / f"veristream_results_{timestamp}.json", output)
    save_json(Path(args.output_dir) / "scores_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriStream 的因果流式 StreamingBench 评测")
    parser.add_argument("--anno-path", default="data/streamingbench/questions_real.json")
    parser.add_argument("--video-dir", default="data/streamingbench/videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qa-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--qa-device", default="auto")
    parser.add_argument("--max-qa-tokens", type=int, default=64)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--recent-frames", type=int, default=4)
    parser.add_argument("--max-cards", type=int, default=48)
    parser.add_argument("--anchor-interval", type=int, default=12)
    parser.add_argument("--novelty-threshold", type=float, default=0.12)
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--novelty-backend", choices=["pixel", "clip"], default="clip")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
