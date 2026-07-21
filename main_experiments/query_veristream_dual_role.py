"""使用独立感知/推理服务执行 VeriStream 的问题级工具循环。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.recent_window_eval import decode_video_to_chunks_qwen
from lib.veristream_dual_role import (
    ChunkRepository,
    DualRoleVeriStreamAgent,
    OpenAICompatibleVLMClient,
    VideoEvidenceIndex,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 VeriStream 感知-推理解耦问答")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-duration", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--perception-base-url", required=True)
    parser.add_argument("--perception-model", required=True)
    parser.add_argument("--reasoning-base-url", required=True)
    parser.add_argument("--reasoning-model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-actions", type=int, default=4)
    args = parser.parse_args()

    index = VideoEvidenceIndex.load(args.index_path)
    chunks, backend = decode_video_to_chunks_qwen(args.video_path, args.chunk_duration, args.fps)
    repository = ChunkRepository(index.video_id, chunks, args.video_path)
    perception = OpenAICompatibleVLMClient(
        args.perception_base_url, args.perception_model, api_key=args.api_key, max_tokens=args.max_tokens
    )
    reasoning = OpenAICompatibleVLMClient(
        args.reasoning_base_url, args.reasoning_model, api_key=args.api_key, max_tokens=args.max_tokens
    )
    agent = DualRoleVeriStreamAgent(perception, reasoning, repository, index, args.max_actions)
    answer, trace = agent.answer(args.question)
    index.save(args.index_path)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "video_id": index.video_id,
                "question": args.question,
                "answer": answer,
                "decode_backend": backend,
                "trace": trace.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(answer)


if __name__ == "__main__":
    main()
