"""感知-推理解耦 VeriStream 的纯逻辑测试。"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from lib.veristream import EvidenceStatus
from lib.veristream_dual_role import (
    ChunkRepository,
    CoarseEvidenceIndexer,
    DualRoleVeriStreamAgent,
    RawEvidenceRef,
    VideoEvidenceIndex,
    parse_dual_role_action,
)


@dataclass
class Chunk:
    frames: list[Image.Image]
    chunk_index: int
    start_time: float
    end_time: float


def make_chunks() -> list[Chunk]:
    colors = ["white", "white", "red", "red"]
    return [Chunk([Image.new("RGB", (12, 12), color=color)], index, float(index), float(index + 1)) for index, color in enumerate(colors)]


class FakePerception:
    def __init__(self) -> None:
        self.calls = 0

    def generate_from_frames(self, _frames, prompt: str) -> str:
        self.calls += 1
        if "Check whether" in prompt:
            return "SUPPORTED\nThe red cup is visible."
        if "Re-inspect" in prompt:
            return '{"observation":"A red cup is on the table.","entities":["cup"],"actions":["place"],"confidence":0.9}'
        return '{"observation":"A red cup is on the table.","entities":["cup"],"actions":["place"],"confidence":0.8}'


class FakeReasoning:
    def __init__(self) -> None:
        self.frame_calls = 0
        self.responses = iter(
            [
                '{"action":"verify_evidence","memory_ids":["video-a:m00001"],"target":"cup"}',
                '{"action":"answer","memory_ids":[],"answer":"A"}',
                "A",
            ]
        )

    def generate_from_text(self, _prompt: str) -> str:
        return next(self.responses)

    def generate_from_frames(self, _frames, _prompt: str) -> str:
        self.frame_calls += 1
        return "A"


class DualRoleMemoryTest(unittest.TestCase):
    def test_repository_samples_full_interval_uniformly(self) -> None:
        repository = ChunkRepository("video-a", make_chunks())
        _, frames = repository.select_interval(0.0, 4.0, max_frames=2)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(frames[1].getpixel((0, 0)), (255, 0, 0))

    def test_index_round_trip_and_namespace(self) -> None:
        index = VideoEvidenceIndex("video-a", max_observations=2)
        memory = index.add_observation(RawEvidenceRef("video-a", [0], 0, 1), "A cup is on the table.", ["cup"], ["place"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            index.save(path)
            restored = VideoEvidenceIndex.load(path)
        self.assertEqual(restored.observations[memory.memory_id].observation, memory.observation)
        with self.assertRaises(ValueError):
            index.add_observation(RawEvidenceRef("video-b", [0], 0, 1), "Wrong video")

    def test_verified_memory_ranks_before_candidate(self) -> None:
        index = VideoEvidenceIndex("video-a")
        old = index.add_observation(RawEvidenceRef("video-a", [0], 0, 1), "A cup is on the table.", ["cup"])
        new = index.add_observation(RawEvidenceRef("video-a", [3], 3, 4), "A cup is on the floor.", ["cup"])
        index.mark_verified(old.memory_id)
        hits = index.search("Where is the cup?", top_k=3)
        self.assertEqual(hits[0].memory_id, old.memory_id)
        self.assertIn(new.memory_id, [item.memory_id for item in hits])

    def test_tool_parser_rejects_unlisted_action(self) -> None:
        self.assertIsNone(parse_dual_role_action('{"action":"delete_memory","memory_ids":[]}'))
        self.assertIsNone(parse_dual_role_action("not json"))


class DualRoleAgentTest(unittest.TestCase):
    def test_index_retrieve_verify_answer_loop(self) -> None:
        perception = FakePerception()
        index = VideoEvidenceIndex("video-a")
        repository = ChunkRepository("video-a", make_chunks())
        calls = CoarseEvidenceIndexer(perception, coarse_stride=2).build(repository, index)
        self.assertEqual(calls, 2)
        reasoning = FakeReasoning()
        agent = DualRoleVeriStreamAgent(perception, reasoning, repository, index, max_actions=2)
        answer, trace = agent.answer("Where is the cup?")
        self.assertEqual(answer, "A")
        self.assertTrue(trace.verified_ids)
        self.assertEqual(reasoning.frame_calls, 1)
        self.assertEqual(trace.final_vision_frames, 4)
        self.assertEqual(index.observations[trace.verified_ids[0]].status, EvidenceStatus.VERIFIED)


if __name__ == "__main__":
    unittest.main()
