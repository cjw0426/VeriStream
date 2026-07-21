"""
OVO-Bench recent-window evaluation for Qwen3-VL.

Aligned with internal eval_recent_frames_ovo.py:
decode video -> chunk by time -> keep the last N chunks -> generate_from_frames
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ovo_constants import BACKWARD_TASKS, FORWARD_TASKS, REAL_TIME_TASKS
from lib.clip_topk_selector import CLIPTopKFrameSelector
from lib.recent_window_eval import load_jsonl_results
from lib.recent_window_eval_qwen3 import (
    RecentWindowQAModel,
    evaluate_ovo_backward_realtime,
    evaluate_ovo_forward,
    print_ovo_results,
)

MODEL_LABEL = "Qwen3-VL"


def make_ovo_key(item: dict[str, Any]) -> str:
    return f"{item.get('task', '')}:{item.get('id')}"


def get_checkpoint_path(result_dir: str, process_index: int, num_processes: int) -> str:
    if num_processes == 1:
        os.makedirs(result_dir, exist_ok=True)
        return os.path.join(result_dir, "results_incremental.jsonl")
    shard_dir = os.path.join(result_dir, f"rank_{process_index}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, "results_incremental.jsonl")


def get_done_path(result_dir: str, process_index: int, num_processes: int) -> str:
    if num_processes == 1:
        os.makedirs(result_dir, exist_ok=True)
        return os.path.join(result_dir, "done")
    shard_dir = os.path.join(result_dir, f"rank_{process_index}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, "done")


def get_error_path(result_dir: str, process_index: int, num_processes: int) -> str:
    if num_processes == 1:
        os.makedirs(result_dir, exist_ok=True)
        return os.path.join(result_dir, "error.json")
    shard_dir = os.path.join(result_dir, f"rank_{process_index}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, "error.json")


def strip_internal_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "_key"}


def load_checkpoint_state(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    records, done_keys = load_jsonl_results(path)
    backward_results: list[dict[str, Any]] = []
    realtime_results: list[dict[str, Any]] = []
    forward_results: list[dict[str, Any]] = []

    for raw in records:
        item = strip_internal_fields(raw)
        key = raw.get("_key")
        if not isinstance(key, str) or not key:
            key = make_ovo_key(item)
        done_keys.add(key)
        task = item.get("task")
        if task in BACKWARD_TASKS:
            backward_results.append(item)
        elif task in REAL_TIME_TASKS:
            realtime_results.append(item)
        elif task in FORWARD_TASKS:
            forward_results.append(item)

    return backward_results, realtime_results, forward_results, done_keys


def append_checkpoint_row(handle: Any, item: dict[str, Any]) -> None:
    record = dict(item)
    record["_key"] = make_ovo_key(item)
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def merge_shard_results(result_dir: str, num_processes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_paths = (
        [os.path.join(result_dir, "results_incremental.jsonl")]
        if num_processes == 1
        else [os.path.join(result_dir, f"rank_{rank}", "results_incremental.jsonl") for rank in range(num_processes)]
    )

    backward_results: list[dict[str, Any]] = []
    realtime_results: list[dict[str, Any]] = []
    forward_results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for path in checkpoint_paths:
        records, _ = load_jsonl_results(path)
        for raw in records:
            item = strip_internal_fields(raw)
            key = raw.get("_key")
            if not isinstance(key, str) or not key:
                key = make_ovo_key(item)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            task = item.get("task")
            if task in BACKWARD_TASKS:
                backward_results.append(item)
            elif task in REAL_TIME_TASKS:
                realtime_results.append(item)
            elif task in FORWARD_TASKS:
                forward_results.append(item)

    return backward_results, realtime_results, forward_results


def write_done_marker(path: str) -> None:
    with open(path, "w") as handle:
        handle.write(datetime.now().isoformat() + "\n")


def write_error_marker(path: str, error: Exception) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def wait_for_done_markers(result_dir: str, num_processes: int) -> None:
    if num_processes <= 1:
        return

    timeout_seconds = float(os.environ.get("FILE_SYNC_TIMEOUT_SECONDS", "43200"))
    poll_interval = float(os.environ.get("FILE_SYNC_POLL_SECONDS", "10"))
    done_paths = [os.path.join(result_dir, f"rank_{rank}", "done") for rank in range(num_processes)]
    error_paths = [os.path.join(result_dir, f"rank_{rank}", "error.json") for rank in range(num_processes)]
    deadline = time.time() + timeout_seconds

    while True:
        failures = [path for path in error_paths if os.path.exists(path)]
        if failures:
            failure_payloads: list[str] = []
            for path in failures:
                try:
                    with open(path, encoding="utf-8") as handle:
                        failure_payloads.append(f"{path}: {handle.read().strip()}")
                except OSError:
                    failure_payloads.append(path)
            raise RuntimeError("Detected rank failure markers:\n" + "\n".join(failure_payloads))
        missing = [path for path in done_paths if not os.path.exists(path)]
        if not missing:
            return
        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for rank completion markers: {missing}")
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="OVO-Bench recent-window evaluation for Qwen3-VL")
    parser.add_argument("--model_path", required=True, help="Example: Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result_dir", default="results/ovo_bench_recent_window_qwen3vl")
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument(
        "--frame_selection",
        choices=[
            "recent",
            "uniform",
            "clip_topk",
            "recent_uniform",
            "recent_clip_topk",
            "recent_memory_uniform",
            "recent_memory_clip_topk",
            "recent_state_memory_uniform_v4",
            "recent_state_memory_clip_topk_v4",
            "recent_state_memory_stratified_v4",
            "recent_vst_memory_uniform",
            "recent_vst_memory_clip_topk",
            "recent_vst_memory_uniform_v2",
            "recent_vst_memory_clip_topk_v2",
        ],
        default="recent",
    )
    parser.add_argument("--supplemental_frames", type=int, default=0)
    parser.add_argument("--memory_num_items", type=int, default=3)
    parser.add_argument("--memory_group_size", type=int, default=4)
    parser.add_argument("--memory_clip_size", type=int, default=4)
    parser.add_argument("--memory_max_clips", type=int, default=4)
    parser.add_argument(
        "--memory_max_tokens",
        type=int,
        default=None,
        help="Generation cap for each VST-style memory update. Defaults to --max_qa_tokens.",
    )
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--clip_model_path", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip_device", default="auto")
    parser.add_argument("--clip_batch_size", type=int, default=32)
    parser.add_argument("--max_qa_tokens", type=int, default=256)
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional sample cap applied independently to backward/realtime/forward after shuffle.",
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    with open(args.anno_path) as handle:
        annotations = json.load(handle)

    backward_anno = [anno for anno in annotations if anno["task"] in BACKWARD_TASKS]
    realtime_anno = [anno for anno in annotations if anno["task"] in REAL_TIME_TASKS]
    forward_anno = [anno for anno in annotations if anno["task"] in FORWARD_TASKS]

    random.seed(42)
    random.shuffle(backward_anno)
    random.shuffle(realtime_anno)
    random.shuffle(forward_anno)
    if args.max_samples_per_split is not None:
        if args.max_samples_per_split < 1:
            raise ValueError("--max_samples_per_split must be >= 1")
        backward_anno = backward_anno[: args.max_samples_per_split]
        realtime_anno = realtime_anno[: args.max_samples_per_split]
        forward_anno = forward_anno[: args.max_samples_per_split]

    vst_v1_memory_selections = {"recent_vst_memory_uniform", "recent_vst_memory_clip_topk"}
    vst_v2_memory_selections = {"recent_vst_memory_uniform_v2", "recent_vst_memory_clip_topk_v2"}
    vst_memory_selections = vst_v1_memory_selections | vst_v2_memory_selections
    state_memory_selections = {
        "recent_state_memory_uniform_v4",
        "recent_state_memory_clip_topk_v4",
        "recent_state_memory_stratified_v4",
    }
    text_memory_selections = {"recent_memory_uniform", "recent_memory_clip_topk"} | state_memory_selections
    clip_selections = {
        "clip_topk",
        "recent_clip_topk",
        "recent_memory_clip_topk",
        "recent_state_memory_clip_topk_v4",
        "recent_vst_memory_clip_topk",
        "recent_vst_memory_clip_topk_v2",
    }
    memory_max_tokens = args.memory_max_tokens if args.memory_max_tokens is not None else args.max_qa_tokens

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"OVO-Bench Recent-Window Evaluation ({MODEL_LABEL})")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"Backward: {len(backward_anno)}, Realtime: {len(realtime_anno)}, Forward: {len(forward_anno)}")
    accelerator.print(f"Processes: {accelerator.num_processes}")
    accelerator.print(
        f"Window: frame_selection={args.frame_selection}, recent_frames_only={args.recent_frames_only}, "
        f"chunk_duration={args.chunk_duration}, fps={args.fps}"
    )
    if args.frame_selection in {"recent_uniform", "recent_clip_topk", *text_memory_selections}:
        accelerator.print(f"Supplemental history frames: {args.supplemental_frames}")
    if args.frame_selection in clip_selections:
        accelerator.print(
            f"CLIP: model={args.clip_model_path}, device={args.clip_device}, batch_size={args.clip_batch_size}"
        )
    if args.frame_selection in text_memory_selections:
        accelerator.print(
            f"Text memory: type={'state_v4' if args.frame_selection in state_memory_selections else 'action'}, "
            f"max_items={args.memory_num_items}, group_size={args.memory_group_size}"
        )
    if args.frame_selection in vst_memory_selections:
        accelerator.print(
            f"VST memory: clip_size={args.memory_clip_size}, max_clips={args.memory_max_clips}, "
            f"max_tokens={memory_max_tokens}, version={'v2' if args.frame_selection in vst_v2_memory_selections else 'v1'}"
        )
    if args.max_samples_per_split is not None:
        accelerator.print(f"Sample cap per split: {args.max_samples_per_split}")
    accelerator.print(f"{'=' * 60}\n")

    evaluator = RecentWindowQAModel(
        model_name=args.model_path,
        device=accelerator.device,
        max_new_tokens=args.max_qa_tokens,
    )
    clip_selector = None
    if args.frame_selection in clip_selections:
        clip_selector = CLIPTopKFrameSelector(
            model_name=args.clip_model_path,
            device=args.clip_device if args.clip_device != "auto" else accelerator.device,
            batch_size=args.clip_batch_size,
        )
    with accelerator.split_between_processes(backward_anno) as local_backward:
        local_backward = list(local_backward)
    with accelerator.split_between_processes(realtime_anno) as local_realtime:
        local_realtime = list(local_realtime)
    with accelerator.split_between_processes(forward_anno) as local_forward:
        local_forward = list(local_forward)

    checkpoint_path = get_checkpoint_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    done_path = get_done_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    error_path = get_error_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    if os.path.exists(done_path):
        os.remove(done_path)
    if os.path.exists(error_path):
        os.remove(error_path)
    backward_results, realtime_results, forward_results, done_keys = load_checkpoint_state(checkpoint_path)

    try:
        with open(checkpoint_path, "a") as checkpoint_file:
            for anno in tqdm(local_backward, desc=f"[GPU{accelerator.process_index}] Backward", disable=not accelerator.is_local_main_process):
                key = make_ovo_key(anno)
                if key in done_keys:
                    continue
                result = evaluate_ovo_backward_realtime(
                    anno=anno,
                    chunked_dir=args.chunked_dir,
                    qa=evaluator,
                    chunk_duration=args.chunk_duration,
                    fps=args.fps,
                    recent_frames_only=args.recent_frames_only,
                    frame_selection=args.frame_selection,
                    supplemental_frames=args.supplemental_frames,
                    clip_selector=clip_selector,
                    memory_num_items=args.memory_num_items,
                    memory_group_size=args.memory_group_size,
                    memory_clip_size=args.memory_clip_size,
                    memory_max_clips=args.memory_max_clips,
                    memory_max_tokens=memory_max_tokens,
                )
                backward_results.append(result)
                done_keys.add(key)
                append_checkpoint_row(checkpoint_file, result)

            for anno in tqdm(local_realtime, desc=f"[GPU{accelerator.process_index}] Realtime", disable=not accelerator.is_local_main_process):
                key = make_ovo_key(anno)
                if key in done_keys:
                    continue
                result = evaluate_ovo_backward_realtime(
                    anno=anno,
                    chunked_dir=args.chunked_dir,
                    qa=evaluator,
                    chunk_duration=args.chunk_duration,
                    fps=args.fps,
                    recent_frames_only=args.recent_frames_only,
                    frame_selection=args.frame_selection,
                    supplemental_frames=args.supplemental_frames,
                    clip_selector=clip_selector,
                    memory_num_items=args.memory_num_items,
                    memory_group_size=args.memory_group_size,
                    memory_clip_size=args.memory_clip_size,
                    memory_max_clips=args.memory_max_clips,
                    memory_max_tokens=memory_max_tokens,
                )
                realtime_results.append(result)
                done_keys.add(key)
                append_checkpoint_row(checkpoint_file, result)

            for anno in tqdm(local_forward, desc=f"[GPU{accelerator.process_index}] Forward", disable=not accelerator.is_local_main_process):
                key = make_ovo_key(anno)
                if key in done_keys:
                    continue
                result = evaluate_ovo_forward(
                    anno=anno,
                    chunked_dir=args.chunked_dir,
                    qa=evaluator,
                    chunk_duration=args.chunk_duration,
                    fps=args.fps,
                    recent_frames_only=args.recent_frames_only,
                    frame_selection=args.frame_selection,
                    supplemental_frames=args.supplemental_frames,
                    clip_selector=clip_selector,
                    memory_num_items=args.memory_num_items,
                    memory_group_size=args.memory_group_size,
                    memory_clip_size=args.memory_clip_size,
                    memory_max_clips=args.memory_max_clips,
                    memory_max_tokens=memory_max_tokens,
                )
                forward_results.append(result)
                done_keys.add(key)
                append_checkpoint_row(checkpoint_file, result)
    except Exception as exc:
        write_error_marker(error_path, exc)
        raise

    write_done_marker(done_path)

    if accelerator.is_main_process:
        wait_for_done_markers(args.result_dir, accelerator.num_processes)
        all_backward, all_realtime, all_forward = merge_shard_results(args.result_dir, accelerator.num_processes)
        print_ovo_results(MODEL_LABEL, all_backward, all_realtime, all_forward)
        os.makedirs(args.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.result_dir, f"qwen3vl_results_{timestamp}.json")
        with open(output_path, "w") as handle:
            json.dump(
                {
                    "config": {
                        "model_path": args.model_path,
                        "frame_selection": args.frame_selection,
                        "recent_frames_only": args.recent_frames_only,
                        "supplemental_frames": args.supplemental_frames,
                        "memory_num_items": args.memory_num_items,
                        "memory_group_size": args.memory_group_size,
                        "memory_clip_size": args.memory_clip_size,
                        "memory_max_clips": args.memory_max_clips,
                        "memory_max_tokens": memory_max_tokens,
                        "chunk_duration": args.chunk_duration,
                        "fps": args.fps,
                        "clip_model_path": args.clip_model_path if args.frame_selection in clip_selections else None,
                        "clip_device": args.clip_device if args.frame_selection in clip_selections else None,
                        "clip_batch_size": args.clip_batch_size if args.frame_selection in clip_selections else None,
                        "max_samples_per_split": args.max_samples_per_split,
                    },
                    "backward": all_backward,
                    "realtime": all_realtime,
                    "forward": all_forward,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
