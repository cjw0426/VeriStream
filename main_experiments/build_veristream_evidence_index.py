"""使用感知角色为单个视频建立 VeriStream 粗粒度证据索引。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.recent_window_eval import decode_video_to_chunks_qwen
from lib.veristream_dual_role import (
    ChunkRepository,
    CoarseEvidenceIndexer,
    OpenAICompatibleVLMClient,
    VideoEvidenceIndex,
)


def build_perception_client(args: argparse.Namespace):
    """优先连接独立感知服务；未提供地址时使用本地冻结模型。"""

    if args.perception_base_url:
        return OpenAICompatibleVLMClient(
            args.perception_base_url,
            args.perception_model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
        )
    if not args.perception_model_path:
        raise ValueError("需要 --perception-base-url 或 --perception-model-path")
    from lib.recent_window_eval_qwen3 import RecentWindowQAModel

    return RecentWindowQAModel(args.perception_model_path, device=args.device, max_new_tokens=args.max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="建立 VeriStream 视频级粗粒度证据索引")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--chunk-duration", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--coarse-stride", type=int, default=12)
    parser.add_argument("--max-observations", type=int, default=256)
    parser.add_argument("--max-events", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--perception-base-url", default="")
    parser.add_argument("--perception-model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--perception-model-path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    video_id = args.video_id or video_path.stem
    client = build_perception_client(args)
    chunks, backend = decode_video_to_chunks_qwen(str(video_path), args.chunk_duration, args.fps)
    repository = ChunkRepository(video_id, chunks, str(video_path))
    index = VideoEvidenceIndex(video_id, args.max_observations, args.max_events)
    calls = CoarseEvidenceIndexer(client, args.coarse_stride).build(repository, index)
    index.save(args.output)
    print(
        json.dumps(
            {
                "video_id": video_id,
                "decode_backend": backend,
                "chunk_count": len(chunks),
                "perception_calls": calls,
                "observation_count": len(index.observations),
                "event_count": len(index.events),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
