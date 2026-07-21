"""使用 Accelerate 运行 VeriStream 感知-推理解耦 OVO 评测。

本入口只评测 Backward 和 Realtime。每个 rank 在自己的 GPU 上加载感知角色
和推理角色，感知角色先建立视频级粗粒度证据索引，推理角色再进行检索、
局部重观察、核验和最终回答。结果按样本增量写盘，支持中断后继续运行。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.recent_window_eval import build_ovo_prompt, calculate_ovo_scores, decode_video_to_chunks_qwen, print_ovo_results
from lib.recent_window_eval_qwen3 import RecentWindowQAModel
from lib.veristream_dual_role import (
    ChunkRepository,
    CoarseEvidenceIndexer,
    DualRoleTrace,
    DualRoleVeriStreamAgent,
    VideoEvidenceIndex,
)
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取某个 rank 已经完成的增量结果。"""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    # 磁盘写满或进程中断可能留下半条 JSON；保留前面的完整样本并截断损坏尾部。
    with path.open("r+", encoding="utf-8") as handle:
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
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


def _wait_for_rank_done(result_dir: Path, num_processes: int, timeout_seconds: float = 86400.0) -> None:
    """使用文件标记同步，避免长视频任务卡在 NCCL barrier。"""

    deadline = time.monotonic() + float(timeout_seconds)
    expected = [result_dir / f"rank_{rank}.done" for rank in range(num_processes)]
    while True:
        missing = [path for path in expected if not path.exists()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            names = ", ".join(path.name for path in missing)
            raise TimeoutError(f"等待 rank checkpoint 超时，仍缺少：{names}")
        time.sleep(2.0)


def _save_index(index: VideoEvidenceIndex, path: Path) -> None:
    """索引写入临时文件后替换，避免进程中断留下半个 JSON。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    index.save(temporary)
    temporary.replace(path)


def _build_models(args: argparse.Namespace, device: Any) -> tuple[RecentWindowQAModel, RecentWindowQAModel]:
    """按显存配置创建感知和推理角色。"""

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


def _evaluate_one(
    anno: dict[str, Any],
    perception: RecentWindowQAModel,
    reasoning: RecentWindowQAModel,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """在问题时间边界内建立索引并执行双角色回答。"""

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
        result["error"] = f"缺少视频：{video_path}"
        return result

    boundary = float(anno.get("realtime", 0.0))
    chunks, backend = decode_video_to_chunks_qwen(
        str(video_path),
        args.chunk_duration,
        args.fps,
        video_end=boundary + 1e-4,
    )
    video_id = f"ovo-{anno['id']}"
    repository = ChunkRepository(video_id, chunks, str(video_path))
    index_path = Path(args.result_dir) / "memory" / f"{video_id}.json"
    if index_path.exists() and not args.rebuild_memory:
        index = VideoEvidenceIndex.load(index_path)
        perception_calls = 0
    else:
        index = VideoEvidenceIndex(video_id, args.max_observations, args.max_events)
        perception_calls = CoarseEvidenceIndexer(
            perception,
            coarse_stride=args.coarse_stride,
            max_frames_per_call=args.max_frames_per_observation,
        ).build(repository, index)
        _save_index(index, index_path)

    agent = DualRoleVeriStreamAgent(
        perception=perception,
        reasoning=reasoning,
        repository=repository,
        index=index,
        max_actions=args.max_actions,
        max_working_memories=args.max_working_memories,
        final_recent_frames=args.final_recent_frames,
    )
    response, trace = agent.answer(build_ovo_prompt(anno["task"], anno))
    _save_index(index, index_path)
    result.update(
        {
            "response": response,
            "decode_backend": backend,
            "causal_chunk_count": len(chunks),
            "memory_observation_count": len(index.observations),
            "memory_event_count": len(index.events),
            "perception_index_calls": perception_calls,
            "trace": trace.to_dict(),
        }
    )
    return result


def _merge_and_score(result_dir: Path, num_processes: int, config: dict[str, Any]) -> None:
    """合并 rank 增量文件并只计算 Backward/Realtime 分数。"""

    groups: dict[str, list[dict[str, Any]]] = {"backward": [], "realtime": []}
    for rank in range(num_processes):
        rows = _load_jsonl(result_dir / f"rank_{rank}.jsonl")
        for row in rows:
            task = row.get("task")
            if task in BACKWARD_TASKS:
                groups["backward"].append(row)
            elif task in REAL_TIME_TASKS:
                groups["realtime"].append(row)
    for rows in groups.values():
        rows.sort(key=lambda item: int(item.get("id", 0)))
    payload = {"config": config, **groups}
    output_path = result_dir / "dual_role_ovo_backward_realtime.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_ovo_results("VeriStream DualRole/Qwen3-VL", groups["backward"], groups["realtime"], [])
    scores = calculate_ovo_scores(groups["backward"], groups["realtime"], [])
    (result_dir / "scores_backward_realtime.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果已保存：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriStream 双角色 OVO Backward/Realtime 八卡评测")
    parser.add_argument("--perception-model-path", required=True)
    parser.add_argument("--reasoning-model-path", default="")
    parser.add_argument("--anno-path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked-dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--chunk-duration", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--max-qa-tokens", type=int, default=128)
    parser.add_argument("--coarse-stride", type=int, default=12)
    parser.add_argument("--max-frames-per-observation", type=int, default=4)
    parser.add_argument("--max-observations", type=int, default=256)
    parser.add_argument("--max-events", type=int, default=96)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--max-working-memories", type=int, default=4)
    parser.add_argument(
        "--final-recent-frames",
        type=int,
        default=4,
        help="最终多模态推理使用的因果最近帧数；设为0可复现text-only最终推理",
    )
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--rebuild-memory", action="store_true")
    parser.add_argument("--share-model", action="store_true", help="感知和推理共享一个模型实例，降低显存占用")
    args = parser.parse_args()

    accelerator = Accelerator()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    groups = {
        "backward": [item for item in annotations if item["task"] in BACKWARD_TASKS],
        "realtime": [item for item in annotations if item["task"] in REAL_TIME_TASKS],
    }
    random.seed(42)
    for rows in groups.values():
        random.shuffle(rows)
        if args.max_samples_per_split:
            del rows[args.max_samples_per_split :]

    # 直接 round-robin 分片，避免旧版 Accelerate 在样本数小于 GPU 数时复制末尾样本。
    local_groups: dict[str, list[dict[str, Any]]] = {
        name: list(rows[accelerator.process_index :: accelerator.num_processes])
        for name, rows in groups.items()
    }

    perception, reasoning = _build_models(args, accelerator.device)
    checkpoint_path = result_dir / f"rank_{accelerator.process_index}.jsonl"
    # 清理上一次中断运行留下的完成标记，避免 rank0 误判所有 worker 已结束。
    done_path = result_dir / f"rank_{accelerator.process_index}.done"
    done_path.unlink(missing_ok=True)
    completed = {str(row.get("id")) + ":" + str(row.get("task")) for row in _load_jsonl(checkpoint_path)}
    for name in ("backward", "realtime"):
        rows = local_groups[name]
        iterator = tqdm(rows, desc=f"DualRole rank{accelerator.process_index} {name}", disable=accelerator.process_index != 0)
        for anno in iterator:
            key = str(anno["id"]) + ":" + str(anno["task"])
            if key in completed:
                continue
            try:
                result = _evaluate_one(anno, perception, reasoning, args)
            except Exception as exc:  # 保存单样本错误，保证其它样本可以继续。
                result = {
                    "id": anno["id"],
                    "video": anno.get("video"),
                    "task": anno["task"],
                    "question": anno.get("question"),
                    "ground_truth": chr(65 + int(anno["gt"])),
                    "response": None,
                    "error": str(exc),
                }
            _append_jsonl(checkpoint_path, result)
            completed.add(key)

    done_path.write_text("done\n", encoding="utf-8")
    # 不再使用 NCCL barrier：完成样本的 rank 立即退出，rank0 通过文件标记等待。
    if not accelerator.is_main_process:
        return
    _wait_for_rank_done(result_dir, accelerator.num_processes)
    config = {**vars(args), "num_processes": accelerator.num_processes}
    _merge_and_score(result_dir, accelerator.num_processes, config)


if __name__ == "__main__":
    main()
