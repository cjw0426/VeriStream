"""VeriStream 纯逻辑与最小工具闭环测试。"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from lib.veristream import EvidenceMemoryStore, EvidenceStatus, VeriStreamAgent, parse_tool_action


@dataclass
class Chunk:
    frames: list[Image.Image]
    chunk_index: int
    start_time: float
    end_time: float


class FakeQA:
    """避免加载真实模型，稳定测试工具动作和记忆状态转换。"""

    def __init__(self) -> None:
        self.actions = iter(
            [
                '{"action":"retrieve","card_ids":[],"target":"cup"}',
                '{"action":"verify_and_zoom","card_ids":["video-a:e00001"],"target":"cup"}',
                '{"action":"answer","card_ids":[]}',
            ]
        )

    def generate_from_text(self, _prompt: str) -> str:
        return next(self.actions)

    def generate_from_frames(self, _frames: list[Image.Image], prompt: str) -> str:
        if "Check whether" in prompt:
            return "SUPPORTED\nA cup is visible on the table."
        if "Describe only" in prompt:
            return "A cup is visible on the table."
        return "A"


def make_chunk(index: int, color: str = "white") -> Chunk:
    return Chunk(
        frames=[Image.new("RGB", (16, 16), color=color)],
        chunk_index=index,
        start_time=float(index),
        end_time=float(index + 1),
    )


class EvidenceMemoryStoreTest(unittest.TestCase):
    def test_video_namespace_and_round_trip(self) -> None:
        first = EvidenceMemoryStore("video-a", max_cards=2)
        card = first.write_candidate(0, 1, [0], "A cup is on a table.")
        second = EvidenceMemoryStore("video-b")
        second.write_candidate(0, 1, [0], "A cup is on a table.")
        self.assertEqual([item.card_id for item in first.retrieve("cup")], [card.card_id])
        self.assertTrue(all(item.video_id == "video-a" for item in first.retrieve("cup")))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            first.save(path)
            restored = EvidenceMemoryStore.load(path)
        self.assertEqual(restored.cards[card.card_id].observation, card.observation)
        self.assertEqual(restored.video_id, "video-a")

    def test_unverified_entries_are_evicted_first(self) -> None:
        store = EvidenceMemoryStore("video-a", max_cards=1)
        old = store.write_candidate(0, 1, [0], "A red car appears.")
        keep = store.write_candidate(2, 3, [2], "A blue cup appears.")
        self.assertEqual(old.status, EvidenceStatus.EVICTED)
        self.assertNotEqual(keep.status, EvidenceStatus.EVICTED)

    def test_invalid_action_is_rejected(self) -> None:
        self.assertIsNone(parse_tool_action('{"action":"delete_everything"}'))
        self.assertIsNone(parse_tool_action("not json"))


class VeriStreamAgentTest(unittest.TestCase):
    def test_candidate_requires_visual_verification(self) -> None:
        agent = VeriStreamAgent(
            qa=FakeQA(),
            video_id="video-a",
            anchor_interval=1,
            novelty_threshold=0.0,
            max_actions=3,
        )
        agent.ingest([make_chunk(0)])
        card = next(iter(agent.store.cards.values()))
        self.assertEqual(card.status, EvidenceStatus.CANDIDATE)
        answer, trace = agent.answer("Which option mentions the cup?")
        self.assertEqual(answer, "A")
        self.assertEqual(agent.store.cards[card.card_id].status, EvidenceStatus.VERIFIED)
        self.assertEqual(trace.final_card_ids, [card.card_id])

    def test_future_chunk_is_not_ingested(self) -> None:
        agent = VeriStreamAgent(qa=FakeQA(), video_id="video-a", anchor_interval=1)
        agent.ingest([make_chunk(0)])
        self.assertEqual({card.chunk_indices[0] for card in agent.store.cards.values()}, {0})


if __name__ == "__main__":
    unittest.main()
