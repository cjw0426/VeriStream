"""VeriStream 在 OVO-Bench 上的单进程因果评测入口。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.recent_window_eval import build_ovo_prompt, calculate_ovo_scores, decode_video_to_chunks_qwen, print_ovo_results, save_json
from lib.recent_window_eval_qwen3 import RecentWindowQAModel
from lib.veristream import ToolTrace, VeriStreamAgent
from ovo_constants import BACKWARD_TASKS, FORWARD_TASKS, REAL_TIME_TASKS


def make_agent(qa: RecentWindowQAModel, video_id: str, args: argparse.Namespace) -> VeriStreamAgent:
    return VeriStreamAgent(
        qa=qa,
        video_id=video_id,
        recent_frames=args.recent_frames,
        max_cards=args.max_cards,
        anchor_interval=args.anchor_interval,
        novelty_threshold=args.novelty_threshold,
        max_actions=args.max_actions,
    )


def trace_metadata(trace: ToolTrace) -> dict[str, Any]:
    return {"trace": trace.to_dict(), "qwen_calls": trace.qwen_calls, "qwen_frames": trace.qwen_frames}


def evaluate_br_or_rt(anno: dict[str, Any], qa: RecentWindowQAModel, args: argparse.Namespace) -> dict[str, Any]:
    """OVO 的回溯/实时任务只向 Agent 暴露问题时刻之前的视频。"""

    video_path = os.path.join(args.chunked_dir, f"{anno['id']}.mp4")
    result: dict[str, Any] = {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "ground_truth": chr(65 + anno["gt"]),
        "response": None,
    }
    if not os.path.exists(video_path):
        result["error"] = f"缺少视频：{video_path}"
        return result
    boundary = float(anno.get("realtime", 0.0))
    chunks, backend = decode_video_to_chunks_qwen(
        video_path, args.chunk_duration, args.fps, video_end=boundary + 1e-4
    )
    agent = make_agent(qa, f"ovo-{anno['id']}", args)
    ingest_trace = ToolTrace()
    agent.ingest(chunks, ingest_trace)
    response, trace = agent.answer(build_ovo_prompt(anno["task"], anno))
    trace.actions = ingest_trace.actions + trace.actions
    trace.qwen_calls += ingest_trace.qwen_calls
    trace.qwen_frames += ingest_trace.qwen_frames
    result.update(
        {
            "response": response,
            "decode_backend": backend,
            "causal_chunk_count": len(chunks),
            "memory_card_count": len(agent.store.cards),
            **trace_metadata(trace),
        }
    )
    return result


def evaluate_forward(anno: dict[str, Any], qa: RecentWindowQAModel, args: argparse.Namespace) -> dict[str, Any]:
    """OVO 前瞻任务的每个官方切片独立建立因果证据记忆。"""

    result = copy.deepcopy(anno)
    for index, test_info in enumerate(result["test_info"]):
        video_path = os.path.join(args.chunked_dir, f"{anno['id']}_{index}.mp4")
        if not os.path.exists(video_path):
            test_info["response"] = None
            test_info["error"] = f"缺少视频：{video_path}"
            continue
        chunks, backend = decode_video_to_chunks_qwen(video_path, args.chunk_duration, args.fps)
        agent = make_agent(qa, f"ovo-{anno['id']}-{index}", args)
        ingest_trace = ToolTrace()
        agent.ingest(chunks, ingest_trace)
        response, trace = agent.answer(build_ovo_prompt(anno["task"], anno, index=index))
        trace.actions = ingest_trace.actions + trace.actions
        trace.qwen_calls += ingest_trace.qwen_calls
        trace.qwen_frames += ingest_trace.qwen_frames
        test_info.update(
            {
                "response": response,
                "decode_backend": backend,
                "causal_chunk_count": len(chunks),
                "memory_card_count": len(agent.store.cards),
                **trace_metadata(trace),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriStream 的 OVO-Bench 八卡因果评测")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--anno-path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked-dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--recent-frames", type=int, default=4)
    parser.add_argument("--max-cards", type=int, default=48)
    parser.add_argument("--anchor-interval", type=int, default=12)
    parser.add_argument("--novelty-threshold", type=float, default=0.12)
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    args = parser.parse_args()

    accelerator = Accelerator()

    annotations = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    groups = {
        "backward": [item for item in annotations if item["task"] in BACKWARD_TASKS],
        "realtime": [item for item in annotations if item["task"] in REAL_TIME_TASKS],
        "forward": [item for item in annotations if item["task"] in FORWARD_TASKS],
    }
    random.seed(42)
    for rows in groups.values():
        random.shuffle(rows)
        if args.max_samples_per_split:
            del rows[args.max_samples_per_split :]

    local_groups: dict[str, list[dict[str, Any]]] = {}
    for name, rows in groups.items():
        # 每个 rank 只拿到自己的样本，避免八张卡重复计算同一视频。
        with accelerator.split_between_processes(rows, apply_padding=False) as local_rows:
            local_groups[name] = list(local_rows)

    qa = RecentWindowQAModel(
        args.model_path,
        device=accelerator.device,
        max_new_tokens=args.max_qa_tokens,
    )
    local_results: dict[str, list[dict[str, Any]]] = {name: [] for name in groups}
    for name, rows in local_groups.items():
        for anno in tqdm(rows, desc=f"VeriStream rank{accelerator.process_index} {name}", disable=accelerator.process_index != 0):
            try:
                local_results[name].append(
                    evaluate_forward(anno, qa, args) if name == "forward" else evaluate_br_or_rt(anno, qa, args)
                )
            except Exception as exc:  # 保存单样本错误，保证长评测可继续复现。
                local_results[name].append(
                    {"id": anno["id"], "task": anno["task"], "response": None, "error": str(exc)}
                )

    os.makedirs(args.result_dir, exist_ok=True)
    shard_path = Path(args.result_dir) / f"veristream_rank_{accelerator.process_index}.json"
    save_json(shard_path, local_results)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        results: dict[str, list[dict[str, Any]]] = {name: [] for name in groups}
        for rank in range(accelerator.num_processes):
            rank_path = Path(args.result_dir) / f"veristream_rank_{rank}.json"
            rank_payload = json.loads(rank_path.read_text(encoding="utf-8"))
            for name in results:
                results[name].extend(rank_payload.get(name, []))
        for name in results:
            results[name].sort(key=lambda item: int(item.get("id", 0)))

        payload = {
            "config": {**vars(args), "num_processes": accelerator.num_processes},
            **results,
        }
        output_path = Path(args.result_dir) / f"veristream_ovo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(output_path, payload)
        print_ovo_results("VeriStream/Qwen3-VL", results["backward"], results["realtime"], results["forward"])
        scores = calculate_ovo_scores(results["backward"], results["realtime"], results["forward"])
        save_json(Path(args.result_dir) / "scores_report.json", scores)
        print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
