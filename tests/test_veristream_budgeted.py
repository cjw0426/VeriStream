"""Pure-logic tests for budgeted VeriStream."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from lib.veristream_budgeted import (
    BudgetedVeriStreamAgent,
    ChangeProposal,
    CoverageChangeIndex,
    CoverageChangeIndexer,
    EvidenceNeed,
    MemoryRetriever,
    RawFrameStore,
    SearchSpec,
    apply_visual_verification_policy,
    parse_neighbor_event_question,
    parse_evidence_needs,
    parse_evidence_plan,
    parse_budgeted_tool_call,
)
from main_experiments.eval_qwen3vl_ovo_budgeted import (
    _decode_exact_current_images,
    _memory_path,
    _parse_sample_ids,
    _sample_rows,
    _validate_or_write_manifest,
    _validate_pretrained_reference,
)
from lib.recent_window_eval import (
    RecentWindowQAModel,
    _CompleteJsonStoppingCriteria,
    extract_mcq_answer,
    score_ovo_br,
)


@dataclass
class Chunk:
    frames: list[Image.Image]
    frame_timestamps: list[float]
    chunk_index: int
    start_time: float
    end_time: float


def make_chunks(count: int = 8) -> list[Chunk]:
    colors = ["white", "white", "red", "red", "blue", "blue", "green", "green"]
    return [
        Chunk(
            [Image.new("RGB", (12, 12), color=colors[index % len(colors)])],
            [float(index)],
            index,
            float(index),
            float(index + 1),
        )
        for index in range(count)
    ]


class FakeImageEncoder:
    def image_embeddings(self, frames):
        rows = []
        for frame in frames:
            pixel = torch.tensor(frame.convert("RGB").getpixel((0, 0)), dtype=torch.float32)
            rows.append(pixel / pixel.norm().clamp_min(1))
        return torch.stack(rows) if rows else torch.empty((0, 3))

    def text_embedding(self, text: str) -> torch.Tensor:
        value = str(text).lower()
        vector = torch.tensor(
            [float("red" in value or "cup" in value), float("green" in value), float("blue" in value)],
            dtype=torch.float32,
        )
        return vector / vector.norm().clamp_min(1)


class FakeTextEncoder:
    TERMS = ["cup", "table", "leave", "phone", "red", "person", "place", "walk"]

    def encode(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            vector = torch.tensor([float(term in lowered) for term in self.TERMS], dtype=torch.float32)
            rows.append(vector / vector.norm().clamp_min(1))
        return torch.stack(rows) if rows else torch.empty((0, len(self.TERMS)))


class FakePerception:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate_from_frames(self, _frames, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if "query-independent causal memory" in prompt:
            match = re.search(r"explicitly paired as follows:\n(\[[^\n]*\])", prompt)
            proposals = json.loads(match.group(1)) if match else []
            assessments = []
            for proposal in proposals:
                assessments.append(
                    {
                        "proposal_id": proposal["proposal_id"],
                        "admission": "local_semantic_event",
                        "type": "object_action",
                        "before": "The cup is held by the person.",
                        "change": "The person places the red cup on the table.",
                        "after": "The cup is on the table.",
                        "entities": ["person", "cup", "table"],
                        "actions": ["place"],
                    }
                )
            return json.dumps(
                {
                    "overview": "A person handles a red cup near a table.",
                    "entities": ["person", "cup", "table"],
                    "actions": ["place"],
                    "visible_text": [],
                    "proposal_assessments": assessments,
                }
            )
        if "inspecting a localized historical" in prompt:
            need_match = re.search(r"Evidence need ID: (N\d+)", prompt)
            need_id = need_match.group(1) if need_match else "N1"
            return json.dumps(
                {
                    "observation": "The red cup is placed on the table.",
                    "supported_need_ids": [need_id],
                    "visible_entities": ["cup", "table"],
                    "visible_actions": ["place"],
                    "visible_text": [],
                    "frame_references": [],
                    "result": "supported",
                }
            )
        if "comparing two localized" in prompt:
            return json.dumps(
                {
                    "relation": "A_before_B",
                    "observation": "The cup is placed before the person leaves.",
                    "supported_need_ids": ["N1"],
                    "frame_references": [],
                }
            )
        if "repairing missing or invalid assessments" in prompt:
            match = re.search(r"Proposals:\n(\[[^\n]*\])", prompt)
            proposals = json.loads(match.group(1)) if match else []
            return json.dumps(
                {
                    "proposal_assessments": [
                        {
                            "proposal_id": item["proposal_id"],
                            "admission": "local_semantic_event",
                            "type": "object_action",
                            "before": "The cup is held.",
                            "change": "The cup is placed.",
                            "after": "The cup is on the table.",
                            "entities": ["cup", "table"],
                            "actions": ["place"],
                        }
                        for item in proposals
                    ]
                }
            )
        return "{}"


class MissingAssessmentPerception(FakePerception):
    def generate_from_frames(self, frames, prompt: str) -> str:
        if "query-independent causal memory" in prompt:
            self.calls += 1
            self.prompts.append(prompt)
            return json.dumps(
                {
                    "overview": "A person handles a cup.",
                    "entities": ["person", "cup"],
                    "actions": [],
                    "visible_text": [],
                    "proposal_assessments": [],
                }
            )
        return super().generate_from_frames(frames, prompt)


class FailedRepairPerception(MissingAssessmentPerception):
    def generate_from_frames(self, frames, prompt: str) -> str:
        if "repairing missing or invalid assessments" in prompt:
            self.calls += 1
            self.prompts.append(prompt)
            return json.dumps({"proposal_assessments": []})
        return super().generate_from_frames(frames, prompt)


class TokenLimitPerception(FakePerception):
    def __init__(self) -> None:
        super().__init__()
        self.max_new_tokens = None
        self.observed_limits: list[int | None] = []

    def generate_from_frames(self, frames, prompt: str) -> str:
        self.observed_limits.append(self.max_new_tokens)
        return super().generate_from_frames(frames, prompt)


class FakeReasoning:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.frame_prompts: list[str] = []
        self.frame_batches: list[list[Image.Image]] = []

    def generate_from_text(self, _prompt: str) -> str:
        return next(self.responses)

    def generate_from_frames(self, frames, prompt: str) -> str:
        self.frame_batches.append(list(frames))
        self.frame_prompts.append(prompt)
        return "A"


def build_index(count: int = 8, recent: int = 4):
    perception = FakePerception()
    image_encoder = FakeImageEncoder()
    text_encoder = FakeTextEncoder()
    store = RawFrameStore("video", make_chunks(count), recent_frames=recent)
    index = CoverageChangeIndexer(
        perception,
        image_encoder,
        text_encoder,
        block_seconds=12,
        coverage_frames=4,
        change_peaks=2,
        max_frames_per_block=8,
    ).build(store)
    return perception, image_encoder, text_encoder, store, index


class BudgetedIndexTest(unittest.TestCase):
    def test_hierarchical_assessment_rejects_inconsistent_parent_and_type(self) -> None:
        indexer = CoverageChangeIndexer(FakePerception(), FakeImageEncoder(), FakeTextEncoder())
        proposal = ChangeProposal("P1", "before", "after", 1.0, 0.4)
        rows, statuses, _ = indexer._validate_proposal_assessments(
            [{
                "proposal_id": "P1",
                "admission": "global_visual_change",
                "type": "object_action",
                "before": "before",
                "change": "change",
                "after": "after",
            }],
            [proposal],
        )
        self.assertEqual(rows, {})
        self.assertEqual(statuses["P1"], "inconsistent_hierarchy")

    def test_semantic_gate_keeps_only_local_events_searchable(self) -> None:
        self.assertEqual(
            CoverageChangeIndexer._resolve_semantic_gate("local_semantic_event", "ambiguous"),
            ("admitted_local", True),
        )
        self.assertEqual(
            CoverageChangeIndexer._resolve_semantic_gate("global_visual_change", "ambiguous"),
            ("navigation_only_global", False),
        )
        self.assertEqual(
            CoverageChangeIndexer._resolve_semantic_gate("no_meaningful_change", "ambiguous"),
            ("rejected_none", False),
        )

    def test_index_prompt_has_no_positive_meaningful_default(self) -> None:
        perception, _, _, _, _ = build_index()
        prompt = perception.prompts[0]
        self.assertIn('"admission": "local_semantic_event|global_visual_change|no_meaningful_change"', prompt)
        self.assertNotIn('"meaningful": true', prompt.lower())

    def test_index_generation_guard_is_scoped_and_restored(self) -> None:
        perception = TokenLimitPerception()
        store = RawFrameStore("video", make_chunks(8), recent_frames=4)
        CoverageChangeIndexer(
            perception,
            FakeImageEncoder(),
            FakeTextEncoder(),
            max_generation_tokens=2048,
        ).build(store)
        self.assertTrue(perception.observed_limits)
        self.assertEqual(set(perception.observed_limits), {2048})
        self.assertIsNone(perception.max_new_tokens)

    def test_five_choice_ovo_answers_are_scored(self) -> None:
        self.assertEqual(extract_mcq_answer("E"), "E")
        self.assertEqual(extract_mcq_answer("5"), "E")
        self.assertEqual(score_ovo_br("E", "E"), 1)

    def test_parse_sample_ids_preserves_order_and_removes_duplicates(self) -> None:
        self.assertEqual(_parse_sample_ids("528, 491,528"), ("528", "491"))

    def test_index_prompt_expands_every_proposal_id(self) -> None:
        store = RawFrameStore("video", make_chunks(8), recent_frames=4)
        proposals = [
            ChangeProposal("P1", "video:f000000", "video:f000001", 1.0, 0.2),
            ChangeProposal("P2", "video:f000001", "video:f000002", 2.0, 0.3),
        ]
        prompt = CoverageChangeIndexer._index_prompt(
            ["video:f000000", "video:f000001", "video:f000002"], proposals, store
        )
        self.assertIn("exactly 2 proposal assessments", prompt)
        self.assertIn('"proposal_id": "P1"', prompt)
        self.assertIn('"proposal_id": "P2"', prompt)

    def test_complete_json_stopping_criterion_has_no_numeric_limit(self) -> None:
        class CharacterTokenizer:
            @staticmethod
            def decode(token_ids, **_kwargs):
                return "".join(chr(int(item)) for item in token_ids)

        criterion = _CompleteJsonStoppingCriteria(CharacterTokenizer(), prompt_length=0)
        incomplete = torch.tensor([[ord(char) for char in '{"value":1']])
        complete = torch.tensor([[ord(char) for char in '{"value":1}']])
        malformed_but_closed = torch.tensor([[ord(char) for char in '{"value":}']])
        brace_in_string = torch.tensor([[ord(char) for char in '{"value":"}"']])
        nested_incomplete = torch.tensor([[ord(char) for char in '{"value":{"nested":1}']])
        self.assertFalse(bool(criterion(incomplete, None)[0]))
        self.assertTrue(bool(criterion(complete, None)[0]))
        self.assertTrue(criterion.matched)
        self.assertTrue(bool(criterion(malformed_but_closed, None)[0]))
        self.assertFalse(bool(criterion(brace_in_string, None)[0]))
        self.assertFalse(bool(criterion(nested_incomplete, None)[0]))
        prefilled = _CompleteJsonStoppingCriteria(
            CharacterTokenizer(), prompt_length=0, initial_text="{"
        )
        generated_body = torch.tensor([[ord(char) for char in '"value":1}']])
        duplicated_open = torch.tensor([[ord(char) for char in '{"value":1}']])
        self.assertTrue(bool(prefilled(generated_body, None)[0]))
        self.assertTrue(bool(prefilled(duplicated_open, None)[0]))
        self.assertEqual(
            _CompleteJsonStoppingCriteria._join_initial_text("{", '  {"value":1}'),
            '{"value":1}',
        )
        self.assertTrue(RecentWindowQAModel._requests_json_object("Return exactly one JSON object:"))
        self.assertFalse(RecentWindowQAModel._requests_json_object("Only give the answer letter."))

    def test_generation_without_artificial_limit_uses_remaining_context(self) -> None:
        self.assertEqual(RecentWindowQAModel._resolve_generation_budget(262144, 2048, None), 260096)
        self.assertEqual(RecentWindowQAModel._resolve_generation_budget(262144, 2048, 0), 260096)
        self.assertEqual(RecentWindowQAModel._resolve_generation_budget(262144, 2048, 512), 512)
        with self.assertRaisesRegex(ValueError, "reaches"):
            RecentWindowQAModel._resolve_generation_budget(1024, 1024, None)

    def test_generation_prefix_distinguishes_input_ids_from_input_embeddings(self) -> None:
        self.assertEqual(
            RecentWindowQAModel._generation_sequence_prefix_length(
                2048, {"inputs_embeds": torch.empty((1, 2048, 4))}
            ),
            0,
        )
        self.assertEqual(
            RecentWindowQAModel._generation_sequence_prefix_length(
                2048, {"input_ids": torch.empty((1, 2048), dtype=torch.long)}
            ),
            2048,
        )

    def test_model_reference_validation_rejects_placeholder(self) -> None:
        with self.assertRaisesRegex(ValueError, "example placeholder"):
            _validate_pretrained_reference("/path/to/Qwen3-VL-8B-Instruct", "model")
        _validate_pretrained_reference("Qwen/Qwen3-VL-8B-Instruct", "model")

    def test_current_frames_are_excluded_from_history(self) -> None:
        store = RawFrameStore("video", make_chunks(8), recent_frames=4)
        self.assertEqual([item.timestamp for item in store.history_records], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual([item.timestamp for item in store.current_records], [4.0, 5.0, 6.0, 7.0])

    def test_past_tense_overrides_incorrect_current_scope(self) -> None:
        needs = parse_evidence_needs(
            json.dumps(
                {
                    "needs": [
                        {
                            "scope": "current",
                            "evidence_type": "current",
                            "query": "items currently visible in the dustbin",
                            "confidence": 0.99,
                        }
                    ]
                }
            ),
            "What did I put in the black dustbin?",
        )
        self.assertTrue(needs[0].requires_history)
        self.assertEqual(needs[0].need_type, "history_event")

    def test_string_false_does_not_enable_visual_verification(self) -> None:
        needs = parse_evidence_needs(
            json.dumps(
                {
                    "needs": [
                        {
                            "scope": "historical",
                            "evidence_type": "history_event",
                            "query": "the earlier event",
                            "visual_verification": "false",
                        }
                    ]
                }
            ),
            "What happened earlier?",
        )
        self.assertFalse(needs[0].visual_verification)

    def test_deterministic_visual_policy_is_not_planner_self_report(self) -> None:
        plan = parse_evidence_plan(
            json.dumps(
                {
                    "needs": [
                        {
                            "scope": "historical", "evidence_type": "transition",
                            "query": "person places cup", "visual_verification": True,
                        },
                        {
                            "scope": "historical", "evidence_type": "attribute",
                            "query": "cup color", "visual_verification": False,
                        },
                    ]
                }
            ),
            "What color was the cup after it was placed?",
        )
        apply_visual_verification_policy(plan)
        self.assertFalse(plan.needs[0].visual_verification)
        self.assertEqual(plan.needs[0].visual_policy_reason, "l1_text_admissible")
        self.assertTrue(plan.needs[1].visual_verification)
        self.assertEqual(plan.needs[1].visual_policy_reason, "visual_type:attribute")

    def test_asi_neighbor_question_parser_preserves_anchor_and_direction(self) -> None:
        self.assertEqual(
            parse_neighbor_event_question("What does the person do after load the wheel"),
            ("after", "load the wheel"),
        )
        self.assertEqual(
            parse_neighbor_event_question("What does the person do before arrange the seperated wire?"),
            ("before", "arrange the seperated wire"),
        )
        self.assertEqual(
            parse_neighbor_event_question(
                "Which object did the person put down before they held the phone/camera?"
            ),
            ("before", "they held the phone/camera"),
        )
        self.assertEqual(
            parse_neighbor_event_question("What did the person do to the book before opening the door?"),
            ("before", "opening the door"),
        )

    def test_run_manifest_is_json_stable_and_rejects_config_mixing(self) -> None:
        base = SimpleNamespace(
            result_dir="ignored",
            change_spans=(1, 2, 4),
            recent_frames=4,
            retry_errors=False,
            rebuild_memory=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            _validate_or_write_manifest(result_dir, base, 0)
            equivalent = SimpleNamespace(**{**vars(base), "change_spans": [1, 2, 4]})
            _validate_or_write_manifest(result_dir, equivalent, 1)
            changed = SimpleNamespace(**{**vars(base), "recent_frames": 8})
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                _validate_or_write_manifest(result_dir, changed, 0)

    def test_external_index_cache_path_is_read_only_source(self) -> None:
        args = SimpleNamespace(index_cache_dir="/old/cache", result_dir="/new/results")
        path, read_only = _memory_path(args, "ovo-483")
        self.assertEqual(path, Path("/old/cache/ovo-483.json"))
        self.assertTrue(read_only)

    def test_task_stratified_sampling_is_balanced_and_deterministic(self) -> None:
        rows = [
            {"id": f"{task}-{index}", "task": task}
            for task in ("EPM", "HLD", "ASI")
            for index in range(5)
        ]
        first = _sample_rows(rows, "backward", 42, max_per_task=2, max_per_split=None)
        second = _sample_rows(rows, "backward", 42, max_per_task=2, max_per_split=None)
        self.assertEqual(first, second)
        self.assertEqual({task: sum(item["task"] == task for item in first) for task in ("EPM", "HLD", "ASI")}, {"EPM": 2, "HLD": 2, "ASI": 2})

    def test_exact_current_decoder_filters_temporary_minimum_window(self) -> None:
        fake_video = torch.stack(
            [torch.full((3, 2, 2), float(index)) for index in range(4)]
        )
        captured: dict[str, object] = {}

        def fake_fetch(request, **_kwargs):
            captured.update(request)
            return fake_video, {
                "fps": 1.0,
                "frames_indices": [0, 1, 2, 3],
                "video_backend": "fake_exact_recent",
                "full_sampled_nframes": 4,
            }

        with patch("lib.qwen_exact_recent_decoder.fetch_recent_video_exact", fake_fetch):
            images, metadata = _decode_exact_current_images(Path("video.mp4"), 1.0, 1.0, 4)
        self.assertGreaterEqual(float(captured["video_end"]), 2.0)
        self.assertEqual(metadata["frame_indices"], [0, 1])
        self.assertEqual(len(images), 2)

    def test_one_call_builds_overview_and_transition(self) -> None:
        perception, _, _, store, index = build_index()
        self.assertEqual(perception.calls, 1)
        self.assertEqual(len(index.overviews), 1)
        self.assertEqual(len(index.transitions), 1)
        self.assertEqual(index.change_proposal_count, 1)
        self.assertEqual(index.assessed_proposal_count, 1)
        self.assertTrue(next(iter(index.overviews.values())).initial_response_parse_success)
        transition = next(iter(index.transitions.values()))
        self.assertEqual(transition.parent_overview_id, "video:o00001")
        self.assertTrue(all(store.record(item).timestamp < 4.0 for item in transition.l0_frame_ids))
        self.assertFalse(any("\u4e00" <= char <= "\u9fff" for char in "".join(perception.prompts)))

    def test_missing_proposal_assessment_is_repaired_and_audited(self) -> None:
        perception = MissingAssessmentPerception()
        image_encoder = FakeImageEncoder()
        text_encoder = FakeTextEncoder()
        store = RawFrameStore("video", make_chunks(8), recent_frames=4)
        index = CoverageChangeIndexer(
            perception, image_encoder, text_encoder, block_seconds=12,
            coverage_frames=4, change_peaks=2, max_frames_per_block=8,
        ).build(store)
        self.assertEqual(index.change_proposal_count, 1)
        self.assertEqual(index.assessed_proposal_count, 1)
        self.assertEqual(index.repaired_proposal_count, 1)
        self.assertEqual(index.proposal_repair_calls, 1)
        overview = next(iter(index.overviews.values()))
        self.assertEqual(overview.proposal_audits[0]["status"], "repaired")
        self.assertEqual(len(index.transitions), 1)

    def test_failed_repair_preserves_initial_and_final_status(self) -> None:
        perception = FailedRepairPerception()
        store = RawFrameStore("video", make_chunks(8), recent_frames=4)
        index = CoverageChangeIndexer(
            perception, FakeImageEncoder(), FakeTextEncoder(), block_seconds=12,
            coverage_frames=4, change_peaks=2, max_frames_per_block=8,
        ).build(store)
        overview = next(iter(index.overviews.values()))
        audit = overview.proposal_audits[0]
        self.assertEqual(audit["initial_status"], "missing")
        self.assertEqual(audit["status"], "repair_failed:missing")
        self.assertEqual(index.repaired_proposal_count, 0)
        self.assertEqual(index.unresolved_proposal_count, 1)

    def test_change_proposal_crosses_history_block_boundary(self) -> None:
        store = RawFrameStore("video", make_chunks(12), recent_frames=0)
        index = CoverageChangeIndexer(
            FakePerception(), FakeImageEncoder(), FakeTextEncoder(),
            block_seconds=4, coverage_frames=2, change_peaks=2,
            max_frames_per_block=6, change_spans=(1,),
        ).build(store)
        second = index.overviews["video:o00002"]
        boundary_transitions = [
            index.transitions[item]
            for item in second.transition_ids
            if index.transitions[item].start_time < second.start_time
        ]
        self.assertTrue(boundary_transitions)

    def test_per_span_mad_proposals_store_normalized_salience(self) -> None:
        perception, _, _, _, index = build_index(count=20, recent=4)
        audits = [audit for overview in index.overviews.values() for audit in overview.proposal_audits]
        self.assertTrue(audits)
        self.assertTrue(all(audit["normalized_salience"] >= 0.5 for audit in audits))
        self.assertEqual(index.config["change_normalization"], "per_span_mad")

    def test_round_trip_preserves_embeddings_and_links(self) -> None:
        _, _, _, store, index = build_index()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            index.save(path)
            restored_store = RawFrameStore("video", make_chunks(8), recent_frames=4)
            restored = CoverageChangeIndex.load(path, restored_store)
        self.assertEqual(set(restored.overviews), set(index.overviews))
        self.assertEqual(set(restored.transitions), set(index.transitions))
        self.assertEqual(set(restored_store.clip_embeddings), set(store.clip_embeddings))
        self.assertTrue(restored.text_embeddings)
        self.assertTrue(restored.embeddings_complete())
        self.assertEqual(restored.change_proposal_count, index.change_proposal_count)

    def test_incremental_build_reuses_immutable_blocks(self) -> None:
        perception = FakePerception()
        image_encoder = FakeImageEncoder()
        text_encoder = FakeTextEncoder()
        indexer = CoverageChangeIndexer(perception, image_encoder, text_encoder, block_seconds=12)
        first_store = RawFrameStore("video", make_chunks(16), recent_frames=4)
        first = indexer.build(first_store)
        self.assertEqual(first.index_perception_calls, 1)
        second_store = RawFrameStore("video", make_chunks(20), recent_frames=4)
        second = indexer.build(second_store, reuse_index=first)
        self.assertEqual(second.index_perception_calls, 1)
        self.assertEqual(len(second.overviews), 2)

    def test_parent_child_and_temporal_navigation(self) -> None:
        _, _, _, _, index = build_index(count=20, recent=4)
        transitions = sorted(index.transitions.values(), key=lambda item: item.peak_time)
        self.assertGreaterEqual(len(transitions), 2)
        parent = index.neighbors(transitions[0].memory_id, "parent")
        self.assertEqual([item.memory_id for item in parent], [transitions[0].parent_overview_id])
        following = index.neighbors(transitions[0].memory_id, "after")
        self.assertEqual([item.memory_id for item in following], [transitions[1].memory_id])


class BudgetedRetrievalTest(unittest.TestCase):
    def test_temporal_graph_parser_binds_atomic_need_ids(self) -> None:
        plan = parse_evidence_plan(
            json.dumps(
                {
                    "needs": [
                        {"need_id": "E1", "scope": "historical", "evidence_type": "history_event", "query": "place cup"},
                        {"need_id": "E2", "scope": "historical", "evidence_type": "history_event", "query": "leave room"},
                    ],
                    "relations": [
                        {"type": "before", "source_need_id": "E1", "target_need_id": "E2"}
                    ],
                }
            ),
            "Did placing the cup happen before leaving?",
        )
        self.assertEqual([item.need_id for item in plan.needs], ["N1", "N2"])
        self.assertEqual(plan.relations[0].source_need_id, "N1")
        self.assertEqual(plan.relations[0].target_need_id, "N2")

    def test_need_retrieval_returns_searchable_memory(self) -> None:
        _, image_encoder, text_encoder, _, index = build_index()
        retriever = MemoryRetriever(index, text_encoder, image_encoder)
        need = EvidenceNeed("N1", "transition", "where the red cup was placed", ["cup"], ["place"])
        hits = retriever.search([need], [SearchSpec("N1", need.search_text())])
        self.assertTrue(hits)
        self.assertEqual(hits[0].kind, "transition")
        self.assertIn("N1", hits[0].ranks)

    def test_irrelevant_query_returns_no_candidate(self) -> None:
        _, image_encoder, text_encoder, _, index = build_index()
        retriever = MemoryRetriever(index, text_encoder, image_encoder)
        need = EvidenceNeed("N1", "history_event", "airplane runway clouds", scope="historical")
        self.assertEqual(retriever.search([need], [SearchSpec("N1", need.search_text())]), [])

    def test_temporal_order_retrieval_keeps_two_localized_memories(self) -> None:
        _, image_encoder, text_encoder, _, index = build_index(count=32, recent=4)
        retriever = MemoryRetriever(index, text_encoder, image_encoder)
        need = EvidenceNeed("N1", "temporal_order", "person places cup", ["person", "cup"], ["place"])
        hits = retriever.search(
            [need],
            [SearchSpec("N1", need.search_text(), memory_types=["transition"])],
        )
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(hit.kind == "transition" for hit in hits))

    def test_parser_rejects_unlisted_tool(self) -> None:
        self.assertIsNone(parse_budgeted_tool_call('{"action":"read_file"}'))
        self.assertIsNotNone(parse_budgeted_tool_call('{"action":"finish_retrieval","selected_memory_ids":[]}'))


class BudgetedAgentTest(unittest.TestCase):
    def test_deterministic_controller_admits_l1_transition_text(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index()
        reasoning = FakeReasoning(
            [json.dumps({"needs": [{
                "scope": "historical", "evidence_type": "transition",
                "query": "red cup placed on table", "visual_verification": True,
            }]})]
        )
        agent = BudgetedVeriStreamAgent(
            perception, reasoning, store, index,
            MemoryRetriever(index, text_encoder, image_encoder), image_encoder,
            max_tool_actions=1, controller_mode="deterministic",
        )
        _, trace = agent.answer("Where was the red cup placed?", "A. table\nB. floor")
        self.assertEqual(trace.executed_tools, 1)
        self.assertLessEqual(
            len([item for item in trace.calls if item.action != "finish_retrieval"]),
            1,
        )
        self.assertFalse(trace.budget_exhausted)
        self.assertTrue(trace.selected_memory_ids)
        self.assertEqual(trace.perception_calls, 0)
        self.assertNotIn("Historical evidence:\n(none)", reasoning.frame_prompts[-1])

    def test_temporal_evidence_graph_uses_distinct_endpoint_memories(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index(count=32, recent=4)
        reasoning = FakeReasoning(
            [
                json.dumps(
                    {
                        "needs": [
                            {
                                "need_id": "E1", "scope": "historical", "evidence_type": "transition",
                                "query": "the person places the cup", "visual_verification": True,
                            },
                            {
                                "need_id": "E2", "scope": "historical", "evidence_type": "transition",
                                "query": "the person leaves", "visual_verification": True,
                            },
                        ],
                        "relations": [
                            {"type": "before", "source_need_id": "E1", "target_need_id": "E2"}
                        ],
                    }
                )
            ]
        )
        retriever = MemoryRetriever(index, text_encoder, image_encoder)
        agent = BudgetedVeriStreamAgent(
            perception, reasoning, store, index, retriever, image_encoder,
            max_visual_actions=2, max_history_visual_frames=6,
            controller_mode="deterministic",
        )
        _, trace = agent.answer("Did placing the cup happen before leaving?", "A. yes\nB. no")
        relation = trace.relation_states["R1"]
        self.assertEqual(relation["state"], "relation_verified")
        self.assertNotEqual(relation["source_memory_id"], relation["target_memory_id"])
        self.assertTrue(relation["satisfied"])
        inspected = {
            (need_id, memory_id)
            for entry in trace.working_evidence
            if entry["status"] == "inspected"
            for need_id in entry["need_ids"]
            for memory_id in entry["memory_ids"]
        }
        self.assertIn(("N1", relation["source_memory_id"]), inspected)
        self.assertIn(("N2", relation["target_memory_id"]), inspected)

    def test_llm_controller_evaluates_temporal_graph_under_shared_budgets(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index(count=32, recent=4)
        retriever = MemoryRetriever(index, text_encoder, image_encoder)
        probe_needs = [
            EvidenceNeed("N1", "transition", "red cup placed", scope="historical"),
            EvidenceNeed("N2", "transition", "person handles cup", scope="historical"),
        ]
        hits = retriever.search(
            probe_needs,
            [SearchSpec("N1", "red cup placed", ["transition"]), SearchSpec("N2", "person handles cup", ["transition"])],
        )
        first = next(item.memory_id for item in hits if "N1" in item.need_scores)
        complementary = retriever.search(
            probe_needs,
            [SearchSpec("N2", "person handles cup", ["transition"])],
            exclude_ids=[first],
            reference_ids=[first],
        )
        second = next(item.memory_id for item in complementary if "N2" in item.need_scores)
        reasoning = FakeReasoning(
            [
                json.dumps(
                    {
                        "needs": [
                            {"need_id": "E1", "scope": "historical", "evidence_type": "transition", "query": "red cup placed", "visual_verification": True},
                            {"need_id": "E2", "scope": "historical", "evidence_type": "transition", "query": "person handles cup", "visual_verification": True},
                        ],
                        "relations": [{"type": "before", "source_need_id": "E1", "target_need_id": "E2"}],
                    }
                ),
                json.dumps({"action": "search_memory", "requests": [
                    {"need_id": "N1", "query": "red cup placed", "memory_types": ["transition"]},
                    {"need_id": "N2", "query": "person handles cup", "memory_types": ["transition"]},
                ]}),
                json.dumps({"action": "search_memory", "requests": [
                    {"need_id": "N2", "query": "person handles cup", "memory_types": ["transition"]},
                ], "exclude_memory_ids": [first]}),
                json.dumps({"action": "inspect_memory", "requests": [
                    {"memory_id": first, "need_id": "N1", "mode": "temporal", "focus": "red cup placed"}
                ]}),
                json.dumps({"action": "inspect_memory", "requests": [
                    {"memory_id": second, "need_id": "N2", "mode": "temporal", "focus": "person handles cup"}
                ]}),
                json.dumps({"action": "finish_retrieval", "selected_memory_ids": [first, second]}),
            ]
        )
        agent = BudgetedVeriStreamAgent(
            perception, reasoning, store, index,
            retriever, image_encoder,
            controller_mode="llm", max_tool_actions=4, max_visual_actions=2,
        )
        _, trace = agent.answer("Did placing the cup happen before handling it?", "A. yes\nB. no")
        self.assertEqual(trace.executed_tools, 4)
        self.assertLessEqual(trace.visual_actions, 2)
        self.assertEqual(trace.relation_states["R1"]["state"], "relation_verified")

    def test_llm_controller_enforces_search_sub_budget(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index()
        repeated_search = json.dumps({
            "action": "search_memory",
            "requests": [{
                "need_id": "N1", "query": "red cup placed",
                "memory_types": ["overview", "transition"],
            }],
        })
        reasoning = FakeReasoning([
            json.dumps({"needs": [{
                "scope": "historical", "evidence_type": "transition",
                "query": "red cup placed",
            }]}),
            repeated_search,
            repeated_search,
            json.dumps({"action": "finish_retrieval", "selected_memory_ids": []}),
        ])
        agent = BudgetedVeriStreamAgent(
            perception, reasoning, store, index,
            MemoryRetriever(index, text_encoder, image_encoder), image_encoder,
            controller_mode="llm", max_tool_actions=2, max_search_rounds=1,
        )
        _, trace = agent.answer("Where was the cup placed?", "A. table\nB. floor")
        self.assertEqual(trace.executed_tools, 2)
        self.assertEqual(trace.search_rounds, 1)
        self.assertEqual(trace.retrieval_stop_reason, "controller_finish")

    def test_high_confidence_current_need_uses_exact_recent_fast_path(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index()
        reasoning = FakeReasoning(
            [
                json.dumps(
                    {
                        "needs": [
                            {
                                "scope": "current",
                                "evidence_type": "current",
                                "query": "the action visible now",
                                "temporal_relation": "current",
                                "visual_verification": False,
                                "confidence": 0.95,
                            }
                        ]
                    }
                )
            ]
        )
        retriever = MemoryRetriever(index, text_encoder, image_encoder)
        agent = BudgetedVeriStreamAgent(
            perception, reasoning, store, index, retriever, image_encoder,
            controller_mode="deterministic",
        )
        baseline_prompt = "What is happening now?\nOptions: A. place; B. leave;\nOnly give the best option's letter directly."
        answer, trace = agent.answer(
            "What is happening now?", "A. place\nB. leave", baseline_prompt=baseline_prompt
        )
        self.assertEqual(answer, "A")
        self.assertTrue(trace.fast_path)
        self.assertEqual(trace.calls, [])
        self.assertEqual(trace.need_states["N1"]["state"], "current_lane")
        self.assertTrue(trace.need_states["N1"]["satisfied"])
        self.assertEqual(reasoning.frame_prompts, [baseline_prompt])

    def test_low_confidence_current_scope_opens_both_lanes(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index()
        reasoning = FakeReasoning(
            [
                json.dumps(
                    {
                        "needs": [
                            {
                                "scope": "current",
                                "evidence_type": "current",
                                "query": "the object involved",
                                "visual_verification": False,
                                "confidence": 0.4,
                            }
                        ]
                    }
                )
            ]
        )
        retriever = MemoryRetriever(index, text_encoder, image_encoder)
        agent = BudgetedVeriStreamAgent(
            perception, reasoning, store, index, retriever, image_encoder,
            controller_mode="deterministic",
        )
        _, trace = agent.answer("What object is involved?", "A. cup\nB. phone")
        self.assertFalse(trace.fast_path)
        self.assertEqual(trace.needs[0].scope, "both")
        self.assertGreaterEqual(trace.search_rounds, 1)

    def test_search_inspect_and_dynamic_history_context(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index()
        transition_id = next(iter(index.transitions))
        reasoning = FakeReasoning(
            [
                json.dumps(
                    {
                        "needs": [
                            {
                                "need_id": "N1", "type": "transition", "query": "red cup placed on table",
                                "entities": ["cup"], "actions": ["place"], "temporal_relation": "before", "priority": 1,
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "action": "search_memory",
                        "requests": [
                            {"need_id": "N1", "query": "red cup placed", "memory_types": ["overview", "transition"], "temporal_relation": "any", "anchor_memory_id": None}
                        ],
                        "exclude_memory_ids": [],
                    }
                ),
                json.dumps(
                    {
                        "action": "inspect_memory",
                        "requests": [{"memory_id": transition_id, "need_id": "N1", "mode": "temporal", "focus": "where the cup ends up"}],
                    }
                ),
                json.dumps(
                    {
                        "action": "finish_retrieval", "satisfied_need_ids": ["N1"],
                        "selected_memory_ids": [transition_id], "reason": "The inspected evidence covers the need.",
                    }
                ),
            ]
        )
        retriever = MemoryRetriever(index, text_encoder, image_encoder)
        agent = BudgetedVeriStreamAgent(
            perception, reasoning, store, index, retriever, image_encoder,
            max_tool_actions=3, max_history_visual_frames=3, history_context_tokens=200,
            visual_verification_policy="planner",
        )
        answer, trace = agent.answer("Where was the red cup placed?", "A. table\nB. floor")
        self.assertEqual(answer, "A")
        self.assertEqual(trace.selected_memory_ids, [transition_id])
        self.assertLessEqual(trace.history_visual_frames, 3)
        self.assertTrue(any(item["status"] == "inspected" for item in trace.working_evidence))
        self.assertEqual(trace.need_states["N1"]["state"], "visually_verified")
        self.assertTrue(trace.need_states["N1"]["satisfied"])
        self.assertIn("The red cup is placed on the table", reasoning.frame_prompts[0])

    def test_asi_neighbor_policy_expands_to_successor_without_visual_call(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index(count=32, recent=4)
        reasoning = FakeReasoning(
            [
                json.dumps(
                    {
                        "needs": [
                            {
                                "scope": "historical", "evidence_type": "transition",
                                "query": "load the wheel", "visual_verification": True,
                            }
                        ]
                    }
                )
            ]
        )
        agent = BudgetedVeriStreamAgent(
            perception, reasoning, store, index,
            MemoryRetriever(index, text_encoder, image_encoder), image_encoder,
            controller_mode="deterministic",
        )
        _, trace = agent.answer(
            "What does the person do after load the wheel",
            "A. pump up the tire\nB. unload the wheel",
        )
        self.assertEqual(trace.needs[0].need_type, "neighbor_event")
        self.assertEqual(trace.needs[0].temporal_relation, "after")
        self.assertFalse(trace.needs[0].visual_verification)
        self.assertTrue(any(call.action == "expand_memory" for call in trace.calls))
        self.assertEqual(trace.perception_calls, 0)
        self.assertGreaterEqual(len(trace.selected_memory_ids), 2)

    def test_explicit_current_images_replace_frame_store_tail(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index()
        reasoning = FakeReasoning(
            [
                json.dumps(
                    {
                        "needs": [
                            {
                                "scope": "current", "evidence_type": "current",
                                "query": "visible action", "confidence": 0.99,
                            }
                        ]
                    }
                )
            ]
        )
        exact = [Image.new("RGB", (7, 7), "purple") for _ in range(3)]
        agent = BudgetedVeriStreamAgent(
            perception, reasoning, store, index,
            MemoryRetriever(index, text_encoder, image_encoder), image_encoder,
        )
        _, trace = agent.answer("What is happening now?", current_images=exact)
        self.assertEqual(trace.final_recent_frames, 3)
        self.assertEqual(len(reasoning.frame_batches[-1]), 3)
        self.assertEqual(reasoning.frame_batches[-1][0].size, (7, 7))

    def test_expand_then_compare_uses_only_discovered_memories(self) -> None:
        perception, image_encoder, text_encoder, store, index = build_index(count=20, recent=4)
        transition_ids = [item.memory_id for item in sorted(index.transitions.values(), key=lambda item: item.peak_time)]
        first, second = transition_ids[:2]
        reasoning = FakeReasoning(
            [
                json.dumps(
                    {
                        "needs": [
                            {
                                "need_id": "N1", "type": "temporal_order", "query": "cup placement order",
                                "entities": ["cup"], "actions": ["place"], "temporal_relation": "before", "priority": 1,
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "action": "search_memory",
                        "requests": [{"need_id": "N1", "query": "cup placement", "memory_types": ["transition"], "temporal_relation": "any", "anchor_memory_id": None}],
                        "exclude_memory_ids": [],
                    }
                ),
                json.dumps({"action": "expand_memory", "memory_id": first, "need_id": "N1", "relation": "after"}),
                json.dumps(
                    {
                        "action": "compare_memory", "memory_ids": [first, second], "need_id": "N1",
                        "mode": "temporal_order", "target": "Compare the two placements",
                    }
                ),
                json.dumps(
                    {
                        "action": "finish_retrieval", "satisfied_need_ids": ["N1"],
                        "selected_memory_ids": [first, second], "reason": "The temporal relation was compared.",
                    }
                ),
            ]
        )
        retriever = MemoryRetriever(index, text_encoder, image_encoder)
        agent = BudgetedVeriStreamAgent(perception, reasoning, store, index, retriever, image_encoder, max_tool_actions=4)
        answer, trace = agent.answer("Which cup placement happened first?", "A. first\nB. second")
        self.assertEqual(answer, "A")
        self.assertEqual(trace.selected_memory_ids, [first, second])
        self.assertTrue(any(item["status"] == "compared" for item in trace.working_evidence))


if __name__ == "__main__":
    unittest.main()
