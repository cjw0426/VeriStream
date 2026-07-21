"""使用 Hybrid-4 VeriStream 单独评测 OVO-Bench Forward split。"""

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

from lib.recent_window_eval import calculate_ovo_scores, decode_video_to_chunks_qwen, print_ovo_results
from lib.veristream_dual_role import ChunkRepository, CoarseEvidenceIndexer, DualRoleVeriStreamAgent, VideoEvidenceIndex
from main_experiments.eval_qwen3vl_ovo_dual_role import (
    _append_jsonl,
    _build_models,
    _load_jsonl,
    _save_index,
    _wait_for_rank_done,
)
from ovo_constants import FORWARD_TASKS


def _evaluate_forward_sample(
    anno: dict[str, Any],
    perception: Any,
    reasoning: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """按 Forward 标注中的每个因果时间点独立建立 memory 并回答。"""

    result = copy.deepcopy(anno)
    for index, test_info in enumerate(result.get("test_info", [])):
        video_path = Path(args.chunked_dir) / f"{anno['id']}_{index}.mp4"
        test_info["response"] = None
        if not video_path.exists():
            test_info["error"] = f"missing video: {video_path}"
            continue

        chunks, backend = decode_video_to_chunks_qwen(
            str(video_path),
            args.chunk_duration,
            args.fps,
        )
        video_id = f"ovo-{anno['id']}-forward-{index}"
        repository = ChunkRepository(video_id, chunks, str(video_path))
        index_path = Path(args.result_dir) / "memory" / f"{video_id}.json"
        if index_path.exists() and not args.rebuild_memory:
            memory_index = VideoEvidenceIndex.load(index_path)
            perception_calls = 0
        else:
            memory_index = VideoEvidenceIndex(video_id, args.max_observations, args.max_events)
            perception_calls = CoarseEvidenceIndexer(
                perception,
                coarse_stride=args.coarse_stride,
                max_frames_per_call=args.max_frames_per_observation,
            ).build(repository, memory_index)
            _save_index(memory_index, index_path)

        agent = DualRoleVeriStreamAgent(
            perception=perception,
            reasoning=reasoning,
            repository=repository,
            index=memory_index,
            max_actions=args.max_actions,
            max_working_memories=args.max_working_memories,
            final_recent_frames=args.final_recent_frames,
        )
        from lib.recent_window_eval import build_ovo_prompt

        response, trace = agent.answer(build_ovo_prompt(anno["task"], anno, index=index))
        _save_index(memory_index, index_path)
        test_info.update(
            {
                "response": response,
                "decode_backend": backend,
                "memory_observation_count": len(memory_index.observations),
                "memory_event_count": len(memory_index.events),
                "perception_index_calls": perception_calls,
                "trace": trace.to_dict(),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriStream Hybrid-4 OVO Forward 评测")
    parser.add_argument("--perception-model-path", required=True)
    parser.add_argument("--reasoning-model-path", default="")
    parser.add_argument("--anno-path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked-dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--coarse-stride", type=int, default=12)
    parser.add_argument("--max-frames-per-observation", type=int, default=8)
    parser.add_argument("--max-observations", type=int, default=256)
    parser.add_argument("--max-events", type=int, default=96)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--max-working-memories", type=int, default=4)
    parser.add_argument("--final-recent-frames", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rebuild-memory", action="store_true")
    parser.add_argument("--share-model", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    forward = [item for item in annotations if item.get("task") in FORWARD_TASKS]
    random.seed(42)
    random.shuffle(forward)
    if args.max_samples is not None:
        forward = forward[: max(1, args.max_samples)]
    local_rows = forward[accelerator.process_index :: accelerator.num_processes]

    perception, reasoning = _build_models(args, accelerator.device)
    checkpoint = result_dir / f"rank_{accelerator.process_index}.jsonl"
    done_path = result_dir / f"rank_{accelerator.process_index}.done"
    done_path.unlink(missing_ok=True)
    completed = {str(row.get("id")) + ":" + str(row.get("task")) for row in _load_jsonl(checkpoint)}
    iterator = tqdm(local_rows, desc=f"Hybrid4 rank{accelerator.process_index} forward", disable=accelerator.process_index != 0)
    for anno in iterator:
        key = str(anno["id"]) + ":" + str(anno["task"])
        if key in completed:
            continue
        try:
            result = _evaluate_forward_sample(anno, perception, reasoning, args)
        except Exception as exc:  # 保留单样本失败，避免阻断其它 Forward 样本。
            result = copy.deepcopy(anno)
            for item in result.get("test_info", []):
                item["response"] = None
                item["error"] = f"{type(exc).__name__}: {exc}"
        _append_jsonl(checkpoint, result)
        completed.add(key)

    done_path.write_text("done\n", encoding="utf-8")
    if not accelerator.is_main_process:
        return
    _wait_for_rank_done(result_dir, accelerator.num_processes)
    merged: list[dict[str, Any]] = []
    for rank in range(accelerator.num_processes):
        merged.extend(_load_jsonl(result_dir / f"rank_{rank}.jsonl"))
    merged.sort(key=lambda item: int(item.get("id", 0)))
    scores = calculate_ovo_scores([], [], merged)
    output = result_dir / "dual_role_ovo_forward.json"
    output.write_text(json.dumps({"config": vars(args), "forward": merged}, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "scores_forward.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print_ovo_results("VeriStream Hybrid-4/Qwen3-VL", [], [], merged)
    print(f"Forward 结果已保存：{output}")


if __name__ == "__main__":
    main()
