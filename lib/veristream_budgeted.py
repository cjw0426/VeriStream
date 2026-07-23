"""Budgeted causal evidence retrieval for long-video question answering.

The persistent video memory has two layers: raw frame records (L0) and a
searchable coverage/change index (L1). Question-specific retrieval and tool
outputs live in a separate working table and never mutate the persistent
semantic memory.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import torch
from PIL import Image


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", str(text).lower())


def _clean_text(value: Any, limit: int = 1000) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def _clean_list(value: Any, limit: int = 16) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        cleaned = _clean_text(item, 160)
        if cleaned and cleaned.lower() not in {row.lower() for row in result}:
            result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    flat = vector.detach().float().cpu().reshape(-1)
    norm = torch.linalg.vector_norm(flat)
    return flat / norm if float(norm) > 0 else flat


def _cosine(left: torch.Tensor | None, right: torch.Tensor | None) -> float:
    if left is None or right is None or left.numel() == 0 or right.numel() == 0:
        return 0.0
    return float(torch.dot(_normalize(left), _normalize(right)).clamp(-1, 1).item())


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


class FrameModel(Protocol):
    def generate_from_frames(self, frames: list[Image.Image], prompt: str) -> str: ...


class TextModel(Protocol):
    def generate_from_text(self, prompt: str) -> str: ...


class ImageEmbeddingModel(Protocol):
    def image_embeddings(self, frames: Sequence[Image.Image]) -> torch.Tensor: ...

    def text_embedding(self, text: str) -> torch.Tensor: ...


class TextEmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class BGETextEncoder:
    """Frozen BGE encoder implemented with Transformers mean pooling."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str | torch.device = "auto") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        resolved = "cuda" if str(device) == "auto" and torch.cuda.is_available() else ("cpu" if str(device) == "auto" else device)
        self.device = torch.device(resolved)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        values = [str(text) for text in texts]
        if not values:
            return torch.empty((0, 0), dtype=torch.float32)
        inputs = self.tokenizer(values, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, dim=-1).float().cpu()


@dataclass
class RawFrameRecord:
    frame_id: str
    video_id: str
    timestamp: float
    chunk_index: int
    frame_index: int
    history_block_id: str | None = None


class RawFrameStore:
    """In-memory L0 frame store with a strict current/history split."""

    def __init__(self, video_id: str, chunks: Sequence[Any], recent_frames: int = 4) -> None:
        if not video_id:
            raise ValueError("video_id must not be empty")
        self.video_id = video_id
        self.records: list[RawFrameRecord] = []
        self.images: dict[str, Image.Image] = {}
        self.clip_embeddings: dict[str, torch.Tensor] = {}
        self.recent_frames = max(0, int(recent_frames))
        serial = 0
        for chunk in sorted(chunks, key=lambda item: int(item.chunk_index)):
            frames = list(chunk.frames)
            timestamps = list(getattr(chunk, "frame_timestamps", []) or [])
            if len(timestamps) != len(frames):
                if not frames:
                    timestamps = []
                elif len(frames) == 1:
                    timestamps = [float(chunk.start_time)]
                else:
                    width = max(0.0, float(chunk.end_time) - float(chunk.start_time))
                    timestamps = [float(chunk.start_time) + width * i / len(frames) for i in range(len(frames))]
            for local_index, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                frame_id = f"{video_id}:f{serial:06d}"
                serial += 1
                record = RawFrameRecord(
                    frame_id=frame_id,
                    video_id=video_id,
                    timestamp=float(timestamp),
                    chunk_index=int(chunk.chunk_index),
                    frame_index=int(local_index),
                )
                self.records.append(record)
                self.images[frame_id] = frame
        self.records.sort(key=lambda item: (item.timestamp, item.chunk_index, item.frame_index))
        split = max(0, len(self.records) - self.recent_frames)
        self.history_records = self.records[:split]
        self.current_records = self.records[split:]
        self._record_map = {item.frame_id: item for item in self.records}

    def record(self, frame_id: str) -> RawFrameRecord | None:
        return self._record_map.get(frame_id)

    def frames_for_ids(self, frame_ids: Sequence[str]) -> list[Image.Image]:
        return [self.images[item] for item in frame_ids if item in self.images]

    def recent_images(self) -> list[Image.Image]:
        return self.frames_for_ids([item.frame_id for item in self.current_records])

    def records_in_interval(self, start_time: float, end_time: float) -> list[RawFrameRecord]:
        return [item for item in self.history_records if start_time <= item.timestamp <= end_time]

    def set_clip_embeddings(self, frame_ids: Sequence[str], vectors: torch.Tensor) -> None:
        if len(frame_ids) != len(vectors):
            raise ValueError("frame_ids and vectors must have equal length")
        for frame_id, vector in zip(frame_ids, vectors):
            self.clip_embeddings[str(frame_id)] = _normalize(vector)


@dataclass
class OverviewMemory:
    memory_id: str
    video_id: str
    start_time: float
    end_time: float
    summary: str
    entities: list[str]
    actions: list[str]
    visible_text: list[str]
    l0_frame_ids: list[str]
    coverage_frame_ids: list[str]
    change_frame_ids: list[str]
    transition_ids: list[str] = field(default_factory=list)
    previous_overview_id: str | None = None
    next_overview_id: str | None = None
    partial: bool = False
    change_proposal_count: int = 0
    initially_assessed_proposal_count: int = 0
    assessed_proposal_count: int = 0
    accepted_proposal_count: int = 0
    searchable_transition_count: int = 0
    repaired_proposal_count: int = 0
    unresolved_proposal_count: int = 0
    proposal_repair_calls: int = 0
    proposal_audits: list[dict[str, Any]] = field(default_factory=list)
    initial_response_parse_success: bool = False

    @property
    def kind(self) -> str:
        return "overview"

    def searchable_text(self) -> str:
        return " ".join([self.summary, *self.entities, *self.actions, *self.visible_text])


@dataclass
class TransitionMemory:
    memory_id: str
    video_id: str
    parent_overview_id: str
    start_time: float
    peak_time: float
    end_time: float
    transition_type: str
    before_state: str
    change: str
    after_state: str
    entities: list[str]
    actions: list[str]
    l0_frame_ids: list[str]
    semantic_admission: str = "local_semantic_event"
    evidence_admissible: bool = True
    semantic_gate_status: str = "admitted"
    visual_change_prior: str = "ambiguous"
    visual_change_score: float = 0.0
    visual_change_fraction: float = 0.0
    visual_change_concentration: float = 0.0
    previous_transition_id: str | None = None
    next_transition_id: str | None = None

    @property
    def kind(self) -> str:
        return "transition"

    def searchable_text(self) -> str:
        return " ".join(
            [self.before_state, self.change, self.after_state, self.transition_type, *self.entities, *self.actions]
        )


MemoryNode = OverviewMemory | TransitionMemory


class CoverageChangeIndex:
    VERSION = 6

    def __init__(self, video_id: str, frame_store: RawFrameStore) -> None:
        if frame_store.video_id != video_id:
            raise ValueError("frame store and index video IDs differ")
        self.video_id = video_id
        self.frame_store = frame_store
        self.overviews: dict[str, OverviewMemory] = {}
        self.transitions: dict[str, TransitionMemory] = {}
        self.text_embeddings: dict[str, torch.Tensor] = {}
        self.visual_embeddings: dict[str, torch.Tensor] = {}
        self.config: dict[str, Any] = {}
        self.index_perception_calls = 0
        self.change_proposal_count = 0
        self.initially_assessed_proposal_count = 0
        self.assessed_proposal_count = 0
        self.accepted_proposal_count = 0
        self.searchable_transition_count = 0
        self.repaired_proposal_count = 0
        self.unresolved_proposal_count = 0
        self.proposal_repair_calls = 0
        self.index_generated_tokens = 0
        self.index_primary_generated_tokens = 0
        self.index_repair_generated_tokens = 0
        self.index_eos_stop_count = 0
        self.index_generation_context_limit_hits = 0
        self.index_json_complete_stop_count = 0

    def all_nodes(self, include_scene_changes: bool = True) -> list[MemoryNode]:
        rows: list[MemoryNode] = list(self.overviews.values())
        rows.extend(
            item
            for item in self.transitions.values()
            if item.evidence_admissible
            and item.transition_type != "no_meaningful_change"
            and (include_scene_changes or item.transition_type not in {"scene_cut", "camera_motion"})
        )
        return sorted(rows, key=lambda item: (item.start_time, item.end_time, item.memory_id))

    def get(self, memory_id: str) -> MemoryNode | None:
        return self.overviews.get(memory_id) or self.transitions.get(memory_id)

    def embeddings_complete(self) -> bool:
        history_ids = {item.frame_id for item in self.frame_store.history_records}
        searchable_ids = {item.memory_id for item in self.all_nodes(include_scene_changes=True)}
        return (
            history_ids.issubset(self.frame_store.clip_embeddings)
            and searchable_ids.issubset(self.text_embeddings)
            and searchable_ids.issubset(self.visual_embeddings)
        )

    def neighbors(self, memory_id: str, relation: str) -> list[MemoryNode]:
        node = self.get(memory_id)
        if node is None:
            return []
        ids: list[str | None] = []
        if relation in {"before", "both"}:
            ids.append(node.previous_overview_id if isinstance(node, OverviewMemory) else node.previous_transition_id)
        if relation in {"after", "both"}:
            ids.append(node.next_overview_id if isinstance(node, OverviewMemory) else node.next_transition_id)
        if relation == "parent" and isinstance(node, TransitionMemory):
            ids.append(node.parent_overview_id)
        if relation == "children" and isinstance(node, OverviewMemory):
            ids.extend(node.transition_ids)
        return [found for item in ids if item and (found := self.get(item)) is not None]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "video_id": self.video_id,
            "config": self.config,
            "index_perception_calls": self.index_perception_calls,
            "change_proposal_count": self.change_proposal_count,
            "initially_assessed_proposal_count": self.initially_assessed_proposal_count,
            "assessed_proposal_count": self.assessed_proposal_count,
            "accepted_proposal_count": self.accepted_proposal_count,
            "searchable_transition_count": self.searchable_transition_count,
            "repaired_proposal_count": self.repaired_proposal_count,
            "unresolved_proposal_count": self.unresolved_proposal_count,
            "proposal_repair_calls": self.proposal_repair_calls,
            "index_generated_tokens": self.index_generated_tokens,
            "index_primary_generated_tokens": self.index_primary_generated_tokens,
            "index_repair_generated_tokens": self.index_repair_generated_tokens,
            "index_eos_stop_count": self.index_eos_stop_count,
            "index_generation_context_limit_hits": self.index_generation_context_limit_hits,
            "index_json_complete_stop_count": self.index_json_complete_stop_count,
            "overviews": [asdict(item) for item in self.overviews.values()],
            "transitions": [asdict(item) for item in self.transitions.values()],
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        frame_ids = list(self.frame_store.clip_embeddings)
        memory_ids = sorted(set(self.text_embeddings) | set(self.visual_embeddings))
        frame_matrix = _stack_vectors([self.frame_store.clip_embeddings[item] for item in frame_ids])
        text_matrix = _stack_vectors([self.text_embeddings.get(item) for item in memory_ids])
        visual_matrix = _stack_vectors([self.visual_embeddings.get(item) for item in memory_ids])
        np.savez_compressed(
            destination.with_suffix(destination.suffix + ".embeddings.npz"),
            frame_ids=np.asarray(frame_ids),
            frame_embeddings=frame_matrix.numpy(),
            memory_ids=np.asarray(memory_ids),
            text_embeddings=text_matrix.numpy(),
            visual_embeddings=visual_matrix.numpy(),
        )

    @classmethod
    def load(cls, path: str | Path, frame_store: RawFrameStore) -> "CoverageChangeIndex":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != cls.VERSION:
            raise ValueError(f"unsupported budgeted index version: {payload.get('version')}")
        index = cls(str(payload["video_id"]), frame_store)
        index.config = dict(payload.get("config", {}))
        index.index_perception_calls = int(payload.get("index_perception_calls", 0))
        index.change_proposal_count = int(payload.get("change_proposal_count", 0))
        index.initially_assessed_proposal_count = int(payload.get("initially_assessed_proposal_count", 0))
        index.assessed_proposal_count = int(payload.get("assessed_proposal_count", 0))
        index.accepted_proposal_count = int(payload.get("accepted_proposal_count", 0))
        index.searchable_transition_count = int(payload.get("searchable_transition_count", 0))
        index.repaired_proposal_count = int(payload.get("repaired_proposal_count", 0))
        index.unresolved_proposal_count = int(payload.get("unresolved_proposal_count", 0))
        index.proposal_repair_calls = int(payload.get("proposal_repair_calls", 0))
        index.index_generated_tokens = int(payload.get("index_generated_tokens", 0))
        index.index_primary_generated_tokens = int(payload.get("index_primary_generated_tokens", 0))
        index.index_repair_generated_tokens = int(payload.get("index_repair_generated_tokens", 0))
        index.index_eos_stop_count = int(payload.get("index_eos_stop_count", 0))
        index.index_generation_context_limit_hits = int(
            payload.get("index_generation_context_limit_hits", 0)
        )
        index.index_json_complete_stop_count = int(payload.get("index_json_complete_stop_count", 0))
        index.overviews = {item["memory_id"]: OverviewMemory(**item) for item in payload.get("overviews", [])}
        index.transitions = {item["memory_id"]: TransitionMemory(**item) for item in payload.get("transitions", [])}
        embedding_path = source.with_suffix(source.suffix + ".embeddings.npz")
        if embedding_path.exists():
            arrays = np.load(embedding_path, allow_pickle=False)
            frame_ids = [str(item) for item in arrays["frame_ids"].tolist()]
            if len(frame_ids):
                frame_store.set_clip_embeddings(frame_ids, torch.from_numpy(arrays["frame_embeddings"]))
            memory_ids = [str(item) for item in arrays["memory_ids"].tolist()]
            for row, memory_id in enumerate(memory_ids):
                if arrays["text_embeddings"].shape[1] > 0:
                    index.text_embeddings[memory_id] = torch.from_numpy(arrays["text_embeddings"][row]).float()
                if arrays["visual_embeddings"].shape[1] > 0:
                    index.visual_embeddings[memory_id] = torch.from_numpy(arrays["visual_embeddings"][row]).float()
        return index


def _stack_vectors(vectors: Sequence[torch.Tensor | None]) -> torch.Tensor:
    present = [item for item in vectors if item is not None and item.numel() > 0]
    if not vectors or not present:
        return torch.empty((len(vectors), 0), dtype=torch.float32)
    width = int(present[0].numel())
    zero = torch.zeros(width, dtype=torch.float32)
    return torch.stack([_normalize(item) if item is not None and item.numel() == width else zero for item in vectors])


@dataclass
class ChangeProposal:
    proposal_id: str
    before_id: str
    after_id: str
    peak_time: float
    distance: float
    span: int = 1
    salience: float = 0.0


class CoverageChangeIndexer:
    """Build query-independent L1 overview and transition memories."""

    ALLOWED_TRANSITIONS = {"object_action", "state_change", "scene_cut", "camera_motion", "no_meaningful_change"}
    ALLOWED_ADMISSIONS = {
        "local_semantic_event",
        "global_visual_change",
        "no_meaningful_change",
    }
    LOCAL_TRANSITIONS = {"object_action", "state_change"}
    GLOBAL_TRANSITIONS = {"scene_cut", "camera_motion"}

    def __init__(
        self,
        perception: FrameModel,
        image_encoder: ImageEmbeddingModel,
        text_encoder: TextEmbeddingModel,
        block_seconds: float = 12.0,
        coverage_frames: int = 4,
        change_peaks: int = 2,
        max_frames_per_block: int = 8,
        minimum_peak_distance: float = 2.0,
        change_spans: Sequence[int] = (1, 2, 4),
        minimum_change_distance: float = 0.05,
        change_mad_scale: float = 0.5,
        enable_proposal_repair: bool = True,
        max_generation_tokens: int | None = None,
    ) -> None:
        self.perception = perception
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.block_seconds = max(1.0, float(block_seconds))
        self.coverage_frames = max(0, int(coverage_frames))
        self.change_peaks = max(0, int(change_peaks))
        self.max_frames_per_block = max(1, int(max_frames_per_block))
        self.minimum_peak_distance = max(0.0, float(minimum_peak_distance))
        self.change_spans = tuple(sorted({max(1, int(item)) for item in change_spans})) or (1,)
        self.minimum_change_distance = max(0.0, float(minimum_change_distance))
        self.change_mad_scale = max(0.0, float(change_mad_scale))
        self.enable_proposal_repair = bool(enable_proposal_repair)
        self.max_generation_tokens = (
            None
            if max_generation_tokens is None or int(max_generation_tokens) <= 0
            else int(max_generation_tokens)
        )

    def _generate(self, frames: list[Image.Image], prompt: str) -> str:
        if self.max_generation_tokens is None:
            return self.perception.generate_from_frames(frames, prompt)
        had_limit = hasattr(self.perception, "max_new_tokens")
        previous_limit = getattr(self.perception, "max_new_tokens", None)
        setattr(self.perception, "max_new_tokens", self.max_generation_tokens)
        try:
            return self.perception.generate_from_frames(frames, prompt)
        finally:
            if had_limit:
                setattr(self.perception, "max_new_tokens", previous_limit)
            else:
                delattr(self.perception, "max_new_tokens")

    def build(
        self,
        frame_store: RawFrameStore,
        reuse_index: CoverageChangeIndex | None = None,
    ) -> CoverageChangeIndex:
        index = CoverageChangeIndex(frame_store.video_id, frame_store)
        index.config = {
            "block_seconds": self.block_seconds,
            "coverage_frames": self.coverage_frames,
            "change_peaks": self.change_peaks,
            "max_frames_per_block": self.max_frames_per_block,
            "minimum_peak_distance": self.minimum_peak_distance,
            "change_spans": list(self.change_spans),
            "minimum_change_distance": self.minimum_change_distance,
            "change_mad_scale": self.change_mad_scale,
            "change_normalization": "per_span_mad",
            "semantic_gate_version": 1,
            "boundary_overlap_frames": max(self.change_spans),
            "enable_proposal_repair": self.enable_proposal_repair,
            "index_generation_token_limit": self.max_generation_tokens or 0,
            "recent_frames": frame_store.recent_frames,
            "image_embedding_model": getattr(self.image_encoder, "model_name", type(self.image_encoder).__name__),
            "text_embedding_model": getattr(self.text_encoder, "model_name", type(self.text_encoder).__name__),
        }
        records = frame_store.history_records
        if not records:
            return index
        if reuse_index is not None:
            if reuse_index.video_id != frame_store.video_id:
                raise ValueError("reused index belongs to a different video")
            reusable_ids = [item.frame_id for item in records if item.frame_id in reuse_index.frame_store.clip_embeddings]
            reusable_vectors = torch.stack([reuse_index.frame_store.clip_embeddings[item] for item in reusable_ids]) if reusable_ids else torch.empty((0, 0))
            if reusable_ids:
                frame_store.set_clip_embeddings(reusable_ids, reusable_vectors)
        missing = [item for item in records if item.frame_id not in frame_store.clip_embeddings]
        if missing:
            vectors = self.image_encoder.image_embeddings([frame_store.images[item.frame_id] for item in missing])
            frame_store.set_clip_embeddings([item.frame_id for item in missing], vectors)
        blocks = self._blocks(records)
        overview_rows: list[OverviewMemory] = []
        transition_rows: list[TransitionMemory] = []
        previous_tail: list[RawFrameRecord] = []
        for block_number, block in enumerate(blocks, start=1):
            overview_id = f"{frame_store.video_id}:o{block_number:05d}"
            for record in block:
                record.history_block_id = overview_id
            previous = reuse_index.overviews.get(overview_id) if reuse_index is not None else None
            block_ids = [item.frame_id for item in block]
            if previous is not None and previous.l0_frame_ids == block_ids:
                overview = OverviewMemory(**asdict(previous))
                reused_transitions = [
                    TransitionMemory(**asdict(item))
                    for item in reuse_index.transitions.values()
                    if item.parent_overview_id == overview_id
                ]
                overview_rows.append(overview)
                transition_rows.extend(reused_transitions)
                index.change_proposal_count += overview.change_proposal_count
                index.initially_assessed_proposal_count += overview.initially_assessed_proposal_count
                index.assessed_proposal_count += overview.assessed_proposal_count
                index.accepted_proposal_count += overview.accepted_proposal_count
                index.searchable_transition_count += overview.searchable_transition_count
                index.repaired_proposal_count += overview.repaired_proposal_count
                index.unresolved_proposal_count += overview.unresolved_proposal_count
                index.proposal_repair_calls += overview.proposal_repair_calls
                previous_tail = list(block[-max(self.change_spans) :])
                continue
            coverage = self._uniform_records(block, self.coverage_frames)
            proposal_records = [*previous_tail, *block]
            proposals = self._change_proposals(
                proposal_records,
                frame_store,
                allowed_after_ids={item.frame_id for item in block},
            )
            previous_tail = list(block[-max(self.change_spans) :])
            pack_ids: list[str] = []
            for proposal in proposals:
                context_ids = self._proposal_context_ids(proposal, proposal_records, frame_store)
                new_ids = [frame_id for frame_id in context_ids if frame_id not in pack_ids]
                if len(pack_ids) + len(new_ids) <= self.max_frames_per_block:
                    pack_ids.extend(new_ids)
                else:
                    for frame_id in (proposal.before_id, proposal.after_id):
                        if frame_id not in pack_ids and len(pack_ids) < self.max_frames_per_block:
                            pack_ids.append(frame_id)
            for record in coverage:
                if record.frame_id not in pack_ids and len(pack_ids) < self.max_frames_per_block:
                    pack_ids.append(record.frame_id)
            pack_ids = self._ordered_unique(pack_ids, frame_store)
            proposals = [
                item for item in proposals if item.before_id in pack_ids and item.after_id in pack_ids
            ]
            index.change_proposal_count += len(proposals)
            if not pack_ids:
                pack_ids = [block[len(block) // 2].frame_id]
            raw = self._generate(
                frame_store.frames_for_ids(pack_ids), self._index_prompt(pack_ids, proposals, frame_store)
            )
            index.index_perception_calls += 1
            self._record_generation(index, repair=False)
            payload = _parse_json_object(raw) or {}
            summary = _clean_text(payload.get("overview")) or _clean_text(raw)
            overview = OverviewMemory(
                memory_id=overview_id,
                video_id=frame_store.video_id,
                start_time=block[0].timestamp,
                end_time=block[-1].timestamp,
                summary=summary or "No reliable overview was produced.",
                entities=_clean_list(payload.get("entities")),
                actions=_clean_list(payload.get("actions")),
                visible_text=_clean_list(payload.get("visible_text")),
                l0_frame_ids=[item.frame_id for item in block],
                coverage_frame_ids=[item.frame_id for item in coverage],
                change_frame_ids=self._ordered_unique(
                    [frame_id for proposal in proposals for frame_id in (proposal.before_id, proposal.after_id)], frame_store
                ),
                partial=(block[-1].timestamp - block[0].timestamp + 1e-6) < self.block_seconds - 1.0,
                change_proposal_count=len(proposals),
                initial_response_parse_success=bool(payload),
            )
            overview_rows.append(overview)
            valid_rows, initial_statuses, extra_audits = self._validate_proposal_assessments(
                payload.get("proposal_assessments", payload.get("transitions")), proposals
            )
            initial_statuses = dict(initial_statuses)
            final_statuses = dict(initial_statuses)
            initial_valid_count = len(valid_rows)
            repaired_ids: set[str] = set()
            unresolved = [item for item in proposals if item.proposal_id not in valid_rows]
            if unresolved and self.enable_proposal_repair:
                repair_ids = self._ordered_unique(
                    [frame_id for item in unresolved for frame_id in (item.before_id, item.after_id)], frame_store
                )
                repair_raw = self._generate(
                    frame_store.frames_for_ids(repair_ids),
                    self._proposal_repair_prompt(repair_ids, unresolved, frame_store, initial_statuses),
                )
                index.index_perception_calls += 1
                self._record_generation(index, repair=True)
                index.proposal_repair_calls += 1
                overview.proposal_repair_calls += 1
                repair_payload = _parse_json_object(repair_raw) or {}
                repaired_rows, repair_statuses, repair_extras = self._validate_proposal_assessments(
                    repair_payload.get("proposal_assessments"), unresolved
                )
                extra_audits.extend(repair_extras)
                for proposal in unresolved:
                    proposal_id = proposal.proposal_id
                    if proposal_id in repaired_rows:
                        valid_rows[proposal_id] = repaired_rows[proposal_id]
                        repaired_ids.add(proposal_id)
                    else:
                        final_statuses[proposal_id] = (
                            f"repair_failed:{repair_statuses.get(proposal_id, 'missing')}"
                        )
            audits = []
            for proposal in proposals:
                proposal_id = proposal.proposal_id
                row = valid_rows.get(proposal_id)
                prior = self._visual_change_prior(proposal, frame_store)
                admission = str(row.get("admission")) if row else None
                gate_status, evidence_admissible = self._resolve_semantic_gate(admission, prior["category"])
                before_record = frame_store.record(proposal.before_id)
                after_record = frame_store.record(proposal.after_id)
                audits.append(
                    {
                        "proposal_id": proposal_id,
                        "initial_status": initial_statuses.get(proposal_id, "missing"),
                        "status": (
                            "repaired" if proposal_id in repaired_ids
                            else "assessed" if row is not None
                            else final_statuses.get(proposal_id, "missing")
                        ),
                        "admission": admission,
                        "meaningful": admission not in {None, "no_meaningful_change"},
                        "transition_type": row.get("type") if row else None,
                        "before_state": _clean_text(row.get("before"), 400) if row else "",
                        "change": _clean_text(row.get("change"), 400) if row else "",
                        "after_state": _clean_text(row.get("after"), 400) if row else "",
                        "semantic_gate_status": gate_status,
                        "evidence_admissible": evidence_admissible,
                        "visual_change_prior": prior["category"],
                        "visual_change_score": prior["score"],
                        "visual_change_fraction": prior["fraction"],
                        "visual_change_concentration": prior["concentration"],
                        "before_frame_id": proposal.before_id,
                        "after_frame_id": proposal.after_id,
                        "before_time": before_record.timestamp if before_record else None,
                        "after_time": after_record.timestamp if after_record else None,
                        "span": proposal.span,
                        "clip_distance": proposal.distance,
                        "normalized_salience": proposal.salience,
                    }
                )
            overview.proposal_audits = [*audits, *extra_audits]
            overview.initially_assessed_proposal_count = initial_valid_count
            overview.assessed_proposal_count = len(valid_rows)
            overview.repaired_proposal_count = len(repaired_ids)
            overview.unresolved_proposal_count = len(proposals) - len(valid_rows)
            index.initially_assessed_proposal_count += initial_valid_count
            index.assessed_proposal_count += len(valid_rows)
            index.repaired_proposal_count += len(repaired_ids)
            index.unresolved_proposal_count += overview.unresolved_proposal_count
            parsed_transitions = self._parse_transitions(
                list(valid_rows.values()), proposals, overview, frame_store
            )
            overview.accepted_proposal_count = len(parsed_transitions)
            overview.searchable_transition_count = sum(
                item.evidence_admissible
                for item in parsed_transitions
            )
            index.accepted_proposal_count += len(parsed_transitions)
            index.searchable_transition_count += overview.searchable_transition_count
            transition_rows.extend(parsed_transitions)
        self._link_nodes(overview_rows, transition_rows)
        index.overviews = {item.memory_id: item for item in overview_rows}
        index.transitions = {item.memory_id: item for item in transition_rows}
        self._embed_memories(index, reuse_index)
        return index

    def _record_generation(self, index: CoverageChangeIndex, repair: bool) -> None:
        generated = max(0, int(getattr(self.perception, "_last_generated_tokens", 0)))
        index.index_generated_tokens += generated
        if repair:
            index.index_repair_generated_tokens += generated
        else:
            index.index_primary_generated_tokens += generated
        index.index_eos_stop_count += int(
            bool(getattr(self.perception, "_last_generation_ended_with_eos", False))
        )
        index.index_generation_context_limit_hits += int(
            bool(getattr(self.perception, "_last_generation_hit_context_limit", False))
        )
        index.index_json_complete_stop_count += int(
            bool(getattr(self.perception, "_last_generation_stopped_on_complete_json", False))
        )

    def _blocks(self, records: Sequence[RawFrameRecord]) -> list[list[RawFrameRecord]]:
        blocks: list[list[RawFrameRecord]] = []
        current: list[RawFrameRecord] = []
        block_start = records[0].timestamp
        for record in records:
            if current and record.timestamp >= block_start + self.block_seconds:
                blocks.append(current)
                current = []
                block_start = record.timestamp
            current.append(record)
        if current:
            blocks.append(current)
        return blocks

    @staticmethod
    def _uniform_records(records: Sequence[RawFrameRecord], count: int) -> list[RawFrameRecord]:
        if count <= 0:
            return []
        if len(records) <= count:
            return list(records)
        if count == 1:
            return [records[len(records) // 2]]
        positions = [round(i * (len(records) - 1) / (count - 1)) for i in range(count)]
        return [records[position] for position in positions]

    def _change_proposals(
        self,
        block: Sequence[RawFrameRecord],
        store: RawFrameStore,
        allowed_after_ids: set[str] | None = None,
    ) -> list[ChangeProposal]:
        if len(block) < 2 or self.change_peaks <= 0:
            return []
        by_span: dict[int, list[tuple[int, int, float]]] = {}
        for span in self.change_spans:
            values: list[tuple[int, int, float]] = []
            for after_index in range(span, len(block)):
                if allowed_after_ids is not None and block[after_index].frame_id not in allowed_after_ids:
                    continue
                before_index = after_index - span
                distance = 1.0 - _cosine(
                    store.clip_embeddings.get(block[before_index].frame_id),
                    store.clip_embeddings.get(block[after_index].frame_id),
                )
                values.append((after_index, before_index, distance))
            if values:
                by_span[span] = values
        if not by_span:
            return []

        candidates: dict[int, list[tuple[float, float, int, int]]] = {}
        for span, values in by_span.items():
            distances = np.asarray([item[2] for item in values], dtype=np.float32)
            if len(distances) < 3:
                center = 0.0
                scale = max(self.minimum_change_distance, 1e-6)
            else:
                center = float(np.median(distances))
                mad = float(np.median(np.abs(distances - center)))
                q25, q75 = np.percentile(distances, [25, 75])
                robust_scale = max(1.4826 * mad, float(q75 - q25) / 1.349)
                scale = robust_scale if robust_scale > 1e-6 else max(float(np.std(distances)), 1e-6)
            for after_index, before_index, distance in values:
                salience = (distance - center) / scale
                candidates.setdefault(after_index, []).append(
                    (float(salience), float(distance), before_index, span)
                )

        winning = {
            after_index: max(values, key=lambda item: (item[0], item[1], -item[3]))
            for after_index, values in candidates.items()
        }
        local = [
            (salience, distance, after_index, before_index, span)
            for after_index, (salience, distance, before_index, span) in winning.items()
            if distance >= self.minimum_change_distance
            and salience >= self.change_mad_scale
            and salience >= winning.get(after_index - 1, (-math.inf, 0.0, 0, 0))[0]
            and salience >= winning.get(after_index + 1, (-math.inf, 0.0, 0, 0))[0]
        ]
        if not local:
            return []
        selected: list[tuple[float, float, int, int, int]] = []
        for salience, distance, after_index, before_index, span in sorted(
            local, key=lambda item: (-item[0], -item[1], item[2])
        ):
            timestamp = block[after_index].timestamp
            if any(
                abs(timestamp - block[other_after].timestamp) < self.minimum_peak_distance
                for _, _, other_after, _, _ in selected
            ):
                continue
            selected.append((salience, distance, after_index, before_index, span))
            if len(selected) >= self.change_peaks:
                break
        selected.sort(key=lambda item: block[item[2]].timestamp)
        return [
            ChangeProposal(
                proposal_id=f"P{proposal_index}",
                before_id=block[before_index].frame_id,
                after_id=block[after_index].frame_id,
                peak_time=block[after_index].timestamp,
                distance=float(distance),
                span=span,
                salience=float(salience),
            )
            for proposal_index, (salience, distance, after_index, before_index, span) in enumerate(selected, start=1)
        ]

    @staticmethod
    def _ordered_unique(frame_ids: Iterable[str], store: RawFrameStore) -> list[str]:
        unique = {item for item in frame_ids if store.record(item) is not None}
        return sorted(unique, key=lambda item: store.record(item).timestamp if store.record(item) else math.inf)

    @staticmethod
    def _proposal_context_ids(
        proposal: ChangeProposal,
        records: Sequence[RawFrameRecord],
        store: RawFrameStore,
    ) -> list[str]:
        """Return one causal neighbor on each side plus the assessed pair."""
        ordered = sorted(records, key=lambda item: (item.timestamp, item.frame_id))
        positions = {item.frame_id: index for index, item in enumerate(ordered)}
        before_index = positions.get(proposal.before_id)
        after_index = positions.get(proposal.after_id)
        ids: list[str] = []
        if before_index is not None and before_index > 0:
            ids.append(ordered[before_index - 1].frame_id)
        ids.extend([proposal.before_id, proposal.after_id])
        if after_index is not None and after_index + 1 < len(ordered):
            ids.append(ordered[after_index + 1].frame_id)
        return CoverageChangeIndexer._ordered_unique(ids, store)

    @staticmethod
    def _visual_change_prior(proposal: ChangeProposal, store: RawFrameStore) -> dict[str, float | str]:
        """Compute a conservative, dependency-free scene-dynamics prior.

        The prior is intentionally allowed to be ambiguous. It may veto semantic
        admission only at the two easy extremes: almost no pixel change, or a
        near-global appearance replacement. Local-event recognition remains a
        VLM responsibility.
        """
        before = store.images.get(proposal.before_id)
        after = store.images.get(proposal.after_id)
        if before is None or after is None:
            return {"category": "ambiguous", "score": 0.0, "fraction": 0.0, "concentration": 0.0}
        size = (96, 96)
        left = np.asarray(before.convert("RGB").resize(size), dtype=np.float32) / 255.0
        right = np.asarray(after.convert("RGB").resize(size), dtype=np.float32) / 255.0
        pixel_change = np.abs(left - right).mean(axis=2)
        score = float(pixel_change.mean())
        fraction = float((pixel_change >= 0.12).mean())
        cells = pixel_change.reshape(12, 8, 12, 8).mean(axis=(1, 3)).reshape(-1)
        total = float(cells.sum())
        top_quartile = max(1, len(cells) // 4)
        concentration = float(np.sort(cells)[-top_quartile:].sum() / total) if total > 1e-8 else 0.0
        if score <= 0.018 and fraction <= 0.05:
            category = "no_meaningful_change"
        elif score >= 0.18 and fraction >= 0.78 and concentration <= 0.42:
            category = "global_visual_change"
        else:
            category = "ambiguous"
        return {
            "category": category,
            "score": round(score, 6),
            "fraction": round(fraction, 6),
            "concentration": round(concentration, 6),
        }

    @staticmethod
    def _resolve_semantic_gate(admission: str | None, prior: str) -> tuple[str, bool]:
        if admission is None:
            return "unresolved", False
        if admission == "no_meaningful_change":
            return "rejected_none", False
        if admission == "global_visual_change":
            return "navigation_only_global", False
        if prior in {"global_visual_change", "no_meaningful_change"}:
            # RGB change extent has no motion compensation, so it is an audit
            # diagnostic rather than an uncalibrated hard veto.
            return f"admitted_local_prior_conflict_{prior}", True
        return "admitted_local", True

    @staticmethod
    def _index_prompt(frame_ids: Sequence[str], proposals: Sequence[ChangeProposal], store: RawFrameStore) -> str:
        frame_positions = {frame_id: index for index, frame_id in enumerate(frame_ids)}
        frame_map = [
            {
                "image_index": index,
                "frame_id": frame_id,
                "timestamp": store.record(frame_id).timestamp,
            }
            for index, frame_id in enumerate(frame_ids)
            if store.record(frame_id) is not None
        ]
        proposal_map = [
            {
                "proposal_id": item.proposal_id,
                "before_image_index": frame_positions[item.before_id],
                "after_image_index": frame_positions[item.after_id],
                "before_time": store.record(item.before_id).timestamp if store.record(item.before_id) else None,
                "after_time": store.record(item.after_id).timestamp if store.record(item.after_id) else None,
                "clip_distance": round(item.distance, 6),
                "normalized_salience": round(item.salience, 6),
                "frame_span": item.span,
                "context_before_image_index": (
                    frame_positions.get(frame_ids[max(0, frame_positions[item.before_id] - 1)])
                    if frame_positions[item.before_id] > 0 else None
                ),
                "context_after_image_index": (
                    frame_positions.get(frame_ids[frame_positions[item.after_id] + 1])
                    if frame_positions[item.after_id] + 1 < len(frame_ids) else None
                ),
            }
            for item in proposals
            if item.before_id in frame_positions and item.after_id in frame_positions
        ]
        assessment_skeleton = [
            {
                "proposal_id": item["proposal_id"],
                "admission": "local_semantic_event|global_visual_change|no_meaningful_change",
                "type": "object_action|state_change|scene_cut|camera_motion|no_meaningful_change",
                "before": "...",
                "change": "...",
                "after": "...",
                "entities": ["..."],
                "actions": ["..."],
            }
            for item in proposal_map
        ]
        output_skeleton = {
            "overview": "...",
            "entities": ["..."],
            "actions": ["..."],
            "visible_text": ["..."],
            "proposal_assessments": assessment_skeleton,
        }
        return (
            "You are building a query-independent causal memory for a long video.\n\n"
            "The images are ordered by time. The zero-based image map is:\n"
            f"{json.dumps(frame_map)}\n\n"
            "The CLIP change proposals are explicitly paired as follows:\n"
            f"{json.dumps(proposal_map)}\n\n"
            "Describe only facts that are directly visible in the provided frames. "
            "Do not answer any question. Do not infer hidden intentions, causes, identities, "
            "or events that are not visually supported.\n\n"
            "Produce a concise overview and readable visible text. You must also return exactly "
            f"{len(proposal_map)} proposal assessments, one for every supplied proposal_id, even when there is no "
            "meaningful change. Preserve every proposal_id shown in the output skeleton. First decide admission: "
            "use local_semantic_event only when an object action or object state change is directly visible; use "
            "global_visual_change for a scene cut or dominant camera/viewpoint motion; otherwise use "
            "no_meaningful_change. Then choose a consistent fine type. The valid pairs are "
            "local_semantic_event -> object_action/state_change, global_visual_change -> scene_cut/camera_motion, "
            "and no_meaningful_change -> no_meaningful_change. Context images are for disambiguation only; before, "
            "change, and after must refer to the specified pair. Use empty strings/lists for no_meaningful_change. "
            "Do not merge proposals or invent proposal IDs.\n\n"
            "Return exactly one JSON object:\n"
            f"{json.dumps(output_skeleton)}"
        )

    @staticmethod
    def _proposal_repair_prompt(
        frame_ids: Sequence[str],
        proposals: Sequence[ChangeProposal],
        store: RawFrameStore,
        initial_statuses: dict[str, str],
    ) -> str:
        frame_positions = {frame_id: index for index, frame_id in enumerate(frame_ids)}
        proposal_map = [
            {
                "proposal_id": item.proposal_id,
                "before_image_index": frame_positions[item.before_id],
                "after_image_index": frame_positions[item.after_id],
                "before_time": store.record(item.before_id).timestamp if store.record(item.before_id) else None,
                "after_time": store.record(item.after_id).timestamp if store.record(item.after_id) else None,
                "previous_error": initial_statuses.get(item.proposal_id, "missing"),
            }
            for item in proposals
        ]
        assessment_skeleton = [
            {
                "proposal_id": item["proposal_id"],
                "admission": "local_semantic_event|global_visual_change|no_meaningful_change",
                "type": "object_action|state_change|scene_cut|camera_motion|no_meaningful_change",
                "before": "...",
                "change": "...",
                "after": "...",
                "entities": ["..."],
                "actions": ["..."],
            }
            for item in proposal_map
        ]
        return (
            "You are repairing missing or invalid assessments for explicitly paired video change proposals. "
            "The images are ordered by image index. Assess only the listed proposals; do not produce an overview, "
            "do not merge proposals, and do not invent proposal IDs. Return exactly one valid assessment for every "
            f"listed proposal, including explicit no_meaningful_change decisions. Return exactly {len(proposal_map)} "
            "assessments and preserve every proposal_id in the output skeleton.\n\n"
            "Choose one consistent hierarchical decision: local_semantic_event with object_action/state_change; "
            "global_visual_change with scene_cut/camera_motion; or no_meaningful_change with "
            "no_meaningful_change. Use empty transition text and lists for no_meaningful_change.\n\n"
            f"Proposals:\n{json.dumps(proposal_map)}\n\n"
            "Return exactly one JSON object:\n"
            f"{json.dumps({'proposal_assessments': assessment_skeleton})}"
        )

    def _validate_proposal_assessments(
        self,
        raw_rows: Any,
        proposals: Sequence[ChangeProposal],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
        expected = {item.proposal_id for item in proposals}
        grouped: dict[str, list[dict[str, Any]]] = {item: [] for item in expected}
        extras: list[dict[str, Any]] = []
        if isinstance(raw_rows, list):
            for raw in raw_rows:
                if not isinstance(raw, dict):
                    extras.append({"proposal_id": None, "status": "invalid_non_object"})
                    continue
                proposal_id = str(raw.get("proposal_id", ""))
                if proposal_id not in expected:
                    extras.append({"proposal_id": proposal_id or None, "status": "unexpected_id"})
                    continue
                grouped[proposal_id].append(raw)
        valid: dict[str, dict[str, Any]] = {}
        statuses: dict[str, str] = {}
        for proposal_id in sorted(expected):
            rows = grouped[proposal_id]
            if not rows:
                statuses[proposal_id] = "missing"
                continue
            if len(rows) != 1:
                statuses[proposal_id] = "duplicate"
                continue
            row = dict(rows[0])
            admission = str(row.get("admission", "")).strip().lower()
            if admission not in self.ALLOWED_ADMISSIONS:
                statuses[proposal_id] = "invalid_admission"
                continue
            transition_type = str(row.get("type", "")).strip().lower()
            if transition_type not in self.ALLOWED_TRANSITIONS:
                statuses[proposal_id] = "invalid_type"
                continue
            expected_types = {
                "local_semantic_event": self.LOCAL_TRANSITIONS,
                "global_visual_change": self.GLOBAL_TRANSITIONS,
                "no_meaningful_change": {"no_meaningful_change"},
            }[admission]
            if transition_type not in expected_types:
                statuses[proposal_id] = "inconsistent_hierarchy"
                continue
            if admission != "no_meaningful_change" and not all(
                _clean_text(row.get(key), 400) for key in ("before", "change", "after")
            ):
                statuses[proposal_id] = "missing_transition_text"
                continue
            row["proposal_id"] = proposal_id
            row["admission"] = admission
            row["type"] = transition_type
            valid[proposal_id] = row
            statuses[proposal_id] = "assessed"
        return valid, statuses, extras

    def _parse_transitions(
        self,
        raw_rows: Any,
        proposals: Sequence[ChangeProposal],
        overview: OverviewMemory,
        store: RawFrameStore,
    ) -> list[TransitionMemory]:
        if not isinstance(raw_rows, list) or not proposals:
            return []
        result: list[TransitionMemory] = []
        used: set[int] = set()
        proposal_ids = {item.proposal_id: index for index, item in enumerate(proposals)}
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            proposal_index = proposal_ids.get(str(raw.get("proposal_id", "")))
            if proposal_index is None:
                try:
                    requested_time = float(raw.get("proposal_time"))
                except (TypeError, ValueError):
                    continue
                proposal_index = min(range(len(proposals)), key=lambda index: abs(proposals[index].peak_time - requested_time))
                if abs(proposals[proposal_index].peak_time - requested_time) > self.block_seconds:
                    continue
            if proposal_index in used:
                continue
            proposal = proposals[proposal_index]
            used.add(proposal_index)
            transition_type = str(raw.get("type", "no_meaningful_change")).strip().lower()
            if transition_type not in self.ALLOWED_TRANSITIONS:
                transition_type = "no_meaningful_change"
            admission = str(raw.get("admission", "no_meaningful_change")).strip().lower()
            if admission == "no_meaningful_change" or transition_type == "no_meaningful_change":
                continue
            before_record, after_record = store.record(proposal.before_id), store.record(proposal.after_id)
            if before_record is None or after_record is None:
                continue
            prior = self._visual_change_prior(proposal, store)
            gate_status, evidence_admissible = self._resolve_semantic_gate(
                admission, str(prior["category"])
            )
            result.append(
                TransitionMemory(
                    memory_id="",
                    video_id=overview.video_id,
                    parent_overview_id=overview.memory_id,
                    start_time=before_record.timestamp,
                    peak_time=proposal.peak_time,
                    end_time=after_record.timestamp,
                    transition_type=transition_type,
                    before_state=_clean_text(raw.get("before"), 400),
                    change=_clean_text(raw.get("change"), 400),
                    after_state=_clean_text(raw.get("after"), 400),
                    entities=_clean_list(raw.get("entities")),
                    actions=_clean_list(raw.get("actions")),
                    l0_frame_ids=[proposal.before_id, proposal.after_id],
                    semantic_admission=admission,
                    evidence_admissible=evidence_admissible,
                    semantic_gate_status=gate_status,
                    visual_change_prior=str(prior["category"]),
                    visual_change_score=float(prior["score"]),
                    visual_change_fraction=float(prior["fraction"]),
                    visual_change_concentration=float(prior["concentration"]),
                )
            )
        return result

    @staticmethod
    def _link_nodes(overviews: list[OverviewMemory], transitions: list[TransitionMemory]) -> None:
        for index, overview in enumerate(overviews):
            overview.transition_ids = []
            overview.previous_overview_id = overviews[index - 1].memory_id if index else None
            overview.next_overview_id = overviews[index + 1].memory_id if index + 1 < len(overviews) else None
        transitions.sort(key=lambda item: (item.peak_time, item.parent_overview_id))
        per_parent: Counter[str] = Counter()
        for transition in transitions:
            per_parent[transition.parent_overview_id] += 1
            transition.memory_id = f"{transition.parent_overview_id}:t{per_parent[transition.parent_overview_id]:02d}"
        for index, transition in enumerate(transitions):
            transition.previous_transition_id = transitions[index - 1].memory_id if index else None
            transition.next_transition_id = transitions[index + 1].memory_id if index + 1 < len(transitions) else None
            parent = next((item for item in overviews if item.memory_id == transition.parent_overview_id), None)
            if parent is not None:
                parent.transition_ids.append(transition.memory_id)

    def _embed_memories(
        self,
        index: CoverageChangeIndex,
        reuse_index: CoverageChangeIndex | None = None,
    ) -> None:
        nodes = index.all_nodes(include_scene_changes=True)
        if not nodes:
            return
        pending: list[MemoryNode] = []
        for node in nodes:
            previous = reuse_index.get(node.memory_id) if reuse_index is not None else None
            reusable = reuse_index.text_embeddings.get(node.memory_id) if reuse_index is not None else None
            if previous is not None and reusable is not None and previous.searchable_text() == node.searchable_text():
                index.text_embeddings[node.memory_id] = _normalize(reusable)
            else:
                pending.append(node)
        if pending:
            text_vectors = self.text_encoder.encode([item.searchable_text() for item in pending])
            for node, vector in zip(pending, text_vectors):
                index.text_embeddings[node.memory_id] = _normalize(vector)
        for node in nodes:
            frame_vectors = [index.frame_store.clip_embeddings[item] for item in node.l0_frame_ids if item in index.frame_store.clip_embeddings]
            if frame_vectors:
                index.visual_embeddings[node.memory_id] = _normalize(torch.stack(frame_vectors).mean(dim=0))


NEED_TYPES = {
    "current", "history_event", "transition", "temporal_order", "neighbor_event",
    "state", "attribute", "spatial", "ocr", "count",
}
NEED_SCOPES = {"current", "historical", "both"}
TEMPORAL_RELATIONS = {"current", "before", "after", "during", "first", "last", "repeated", "any"}
TEMPORAL_GRAPH_RELATIONS = {"before", "after"}
VISUAL_EVIDENCE_TYPES = {"history_event", "transition", "state", "attribute", "spatial", "ocr", "count"}
DETERMINISTIC_VISUAL_EVIDENCE_TYPES = {"attribute", "spatial", "state", "ocr"}


def _has_historical_cue(text: str) -> bool:
    value = str(text).lower()
    patterns = (
        r"\bwhat did\b",
        r"\bwhere did\b",
        r"\bwho did\b",
        r"\bwhen did\b",
        r"\bwhy did\b",
        r"\bhow did\b",
        r"\bwhat was\b",
        r"\bwhere was\b",
        r"\bwho was\b",
        r"\bwhat were\b",
        r"\bwhere were\b",
        r"\b(before|after|earlier|previously|first|last|again|repeated|how many times)\b",
        r"\b(had|did|was|were)\b.+\b(put|place|drop|leave|take|pick|move|chop|install|remove|open|close)\w*\b",
    )
    return any(re.search(pattern, value) for pattern in patterns)


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


@dataclass
class EvidenceNeed:
    need_id: str
    need_type: str
    query: str
    entities: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    temporal_relation: str = "any"
    priority: int = 1
    scope: str = ""
    visual_verification: bool | None = None
    confidence: float = 1.0
    model_scope: str = ""
    scope_override_reason: str = ""
    model_visual_verification: bool | None = None
    visual_policy_reason: str = ""

    def __post_init__(self) -> None:
        if self.scope not in NEED_SCOPES:
            self.scope = "current" if self.need_type == "current" else "historical"
        if self.model_scope not in NEED_SCOPES:
            self.model_scope = self.scope
        if self.visual_verification is None:
            self.visual_verification = self.need_type in VISUAL_EVIDENCE_TYPES
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def requires_history(self) -> bool:
        return self.scope in {"historical", "both"}

    @property
    def is_current_only(self) -> bool:
        return self.scope == "current"

    def search_text(self) -> str:
        return " ".join([self.query, *self.entities, *self.actions]).strip()


@dataclass
class TemporalRelationNeed:
    relation_id: str
    relation_type: str
    source_need_id: str
    target_need_id: str
    query: str = ""
    confidence: float = 1.0


@dataclass
class EvidencePlan:
    needs: list[EvidenceNeed]
    relations: list[TemporalRelationNeed] = field(default_factory=list)


def parse_evidence_plan(raw: str, fallback_query: str) -> EvidencePlan:
    payload = _parse_json_object(raw) or {}
    rows = payload.get("needs", [])
    needs: list[EvidenceNeed] = []
    raw_id_map: dict[str, str] = {}
    if isinstance(rows, list):
        for index, row in enumerate(rows[:4], start=1):
            if not isinstance(row, dict):
                continue
            raw_need_id = _clean_text(row.get("need_id"), 32) or f"N{index}"
            need_id = f"N{len(needs) + 1}"
            raw_type = str(row.get("evidence_type", row.get("type", "history_event"))).strip().lower()
            proposed_scope = str(row.get("scope", "")).strip().lower()
            if not proposed_scope:
                proposed_scope = "current" if raw_type == "current" else "historical"
            if proposed_scope not in NEED_SCOPES:
                proposed_scope = "historical"
            temporal = str(row.get("temporal_relation", "any")).strip().lower()
            query = _clean_text(row.get("query"), 400)
            if raw_type not in NEED_TYPES or not query:
                continue
            model_scope = proposed_scope
            override_reason = ""
            if _has_historical_cue(fallback_query) or _has_historical_cue(query):
                if proposed_scope == "current":
                    proposed_scope = "historical"
                    override_reason = "deterministic_historical_cue"
            need_type = raw_type
            if need_type == "current" and proposed_scope != "current":
                need_type = "temporal_order" if temporal in {"before", "after", "first", "last"} else "history_event"
            try:
                confidence = float(row.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 0.5
            model_visual_verification = _parse_bool(
                row.get("visual_verification"), need_type in VISUAL_EVIDENCE_TYPES
            )
            needs.append(
                EvidenceNeed(
                    need_id=need_id,
                    need_type=need_type,
                    query=query,
                    entities=_clean_list(row.get("entities"), 8),
                    actions=_clean_list(row.get("actions"), 8),
                    temporal_relation=temporal if temporal in TEMPORAL_RELATIONS else "any",
                    priority=max(1, min(3, int(row.get("priority", index)))),
                    scope=proposed_scope,
                    visual_verification=model_visual_verification,
                    confidence=confidence,
                    model_scope=model_scope,
                    scope_override_reason=override_reason,
                    model_visual_verification=model_visual_verification,
                )
            )
            raw_id_map[raw_need_id] = need_id
            raw_id_map[need_id] = need_id
    if not needs:
        scope = "historical" if _has_historical_cue(fallback_query) else "both"
        needs = [
            EvidenceNeed(
                "N1",
                "history_event",
                _clean_text(fallback_query, 400) or "observable video evidence",
                scope=scope,
                confidence=0.0,
                model_scope="",
                scope_override_reason="planner_parse_fallback",
            )
        ]
    relations: list[TemporalRelationNeed] = []
    raw_relations = payload.get("relations", [])
    if isinstance(raw_relations, list):
        for index, row in enumerate(raw_relations[:2], start=1):
            if not isinstance(row, dict):
                continue
            relation_type = str(row.get("type", row.get("relation_type", ""))).strip().lower()
            source = raw_id_map.get(str(row.get("source_need_id", row.get("source", ""))))
            target = raw_id_map.get(str(row.get("target_need_id", row.get("target", ""))))
            if relation_type not in TEMPORAL_GRAPH_RELATIONS or not source or not target or source == target:
                continue
            try:
                relation_confidence = float(row.get("confidence", 1.0))
            except (TypeError, ValueError):
                relation_confidence = 0.5
            relations.append(
                TemporalRelationNeed(
                    relation_id=f"R{len(relations) + 1}",
                    relation_type=relation_type,
                    source_need_id=source,
                    target_need_id=target,
                    query=_clean_text(row.get("query"), 400),
                    confidence=max(0.0, min(1.0, relation_confidence)),
                )
            )
    relation_endpoint_ids = {
        need_id
        for relation in relations
        for need_id in (relation.source_need_id, relation.target_need_id)
    }
    for need in needs:
        if need.need_id in relation_endpoint_ids and need.need_type == "history_event":
            need.need_type = "transition"
            need.visual_verification = True
    return EvidencePlan(needs=needs, relations=relations)


def apply_visual_verification_policy(
    plan: EvidencePlan,
    policy: str = "deterministic",
) -> EvidencePlan:
    """Apply an auditable admission policy after parsing the planner output."""
    if policy not in {"deterministic", "planner"}:
        raise ValueError("visual verification policy must be 'deterministic' or 'planner'")
    relation_endpoint_ids = {
        need_id
        for relation in plan.relations
        for need_id in (relation.source_need_id, relation.target_need_id)
    }
    for need in plan.needs:
        model_decision = (
            bool(need.model_visual_verification)
            if need.model_visual_verification is not None
            else bool(need.visual_verification)
        )
        need.model_visual_verification = model_decision
        if need.need_id in relation_endpoint_ids:
            need.visual_verification = True
            need.visual_policy_reason = "temporal_graph_endpoint"
        elif policy == "planner":
            need.visual_verification = model_decision
            need.visual_policy_reason = "planner_decision"
        elif need.need_type in DETERMINISTIC_VISUAL_EVIDENCE_TYPES:
            need.visual_verification = True
            need.visual_policy_reason = f"visual_type:{need.need_type}"
        else:
            need.visual_verification = False
            need.visual_policy_reason = "l1_text_admissible"
    return plan


def parse_neighbor_event_question(question: str) -> tuple[str, str] | None:
    """Return (direction, anchor) for OVO ASI predecessor/successor questions."""
    text = " ".join(str(question).strip().split())
    patterns = (
        r"^what\s+(?:does|did)\s+.+?\s+do\s+(before|after)\s+(.+?)\s*[?.!]*$",
        r"^what\s+(?:happens|happened|occurred)\s+(before|after)\s+(.+?)\s*[?.!]*$",
        r"^(?:what\s+did\s+.+?|which\s+object\s+did\s+.+?)\s+(before|after)\s+(.+?)\s*[?.!]*$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            anchor = _clean_text(match.group(2), 400).rstrip("?.!")
            if anchor:
                return match.group(1).lower(), anchor
    return None


def apply_neighbor_event_policy(plan: EvidencePlan, question: str) -> EvidencePlan:
    parsed = parse_neighbor_event_question(question)
    if parsed is None:
        return plan
    direction, anchor = parsed
    return EvidencePlan(
        needs=[
            EvidenceNeed(
                need_id="N1",
                need_type="neighbor_event",
                query=anchor,
                actions=[anchor],
                temporal_relation=direction,
                priority=1,
                scope="historical",
                visual_verification=False,
                confidence=1.0,
                model_scope=plan.needs[0].model_scope if plan.needs else "",
                scope_override_reason="deterministic_neighbor_event",
                model_visual_verification=(
                    plan.needs[0].model_visual_verification if plan.needs else None
                ),
                visual_policy_reason="l1_text_admissible",
            )
        ],
        relations=[],
    )


def parse_evidence_needs(raw: str, fallback_query: str) -> list[EvidenceNeed]:
    return parse_evidence_plan(raw, fallback_query).needs


@dataclass
class SearchSpec:
    need_id: str
    query: str
    memory_types: list[str] = field(default_factory=lambda: ["overview", "transition"])
    temporal_relation: str = "any"
    anchor_memory_id: str | None = None


@dataclass
class CandidateHit:
    memory_id: str
    kind: str
    start_time: float
    end_time: float
    text: str
    need_scores: dict[str, float] = field(default_factory=dict)
    component_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    ranks: dict[str, dict[str, int]] = field(default_factory=dict)
    utility: float = 0.0


class BM25Index:
    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.tokens = {key: _tokens(value) for key, value in documents.items()}
        self.lengths = {key: len(value) for key, value in self.tokens.items()}
        self.average_length = sum(self.lengths.values()) / len(self.lengths) if self.lengths else 1.0
        frequencies: Counter[str] = Counter()
        for values in self.tokens.values():
            frequencies.update(set(values))
        count = max(1, len(self.tokens))
        self.idf = {token: math.log(1.0 + (count - freq + 0.5) / (freq + 0.5)) for token, freq in frequencies.items()}

    def scores(self, query: str) -> dict[str, float]:
        query_tokens = _tokens(query)
        result: dict[str, float] = {}
        for key, values in self.tokens.items():
            counts = Counter(values)
            length = self.lengths[key]
            score = 0.0
            for token in query_tokens:
                tf = counts[token]
                if tf <= 0:
                    continue
                denominator = tf + self.k1 * (1.0 - self.b + self.b * length / max(self.average_length, 1e-6))
                score += self.idf.get(token, 0.0) * tf * (self.k1 + 1.0) / denominator
            result[key] = score
        return result


class MemoryRetriever:
    """EvidenceNeed-aware retrieval over L1 memories."""

    VISUAL_NEEDS = {"attribute", "spatial", "state"}

    def __init__(
        self,
        index: CoverageChangeIndex,
        text_encoder: TextEmbeddingModel,
        image_encoder: ImageEmbeddingModel,
        candidate_metadata_tokens: int = 768,
        rrf_constant: float = 10.0,
        visual_weight: float = 0.25,
        redundancy_weight: float = 0.25,
        minimum_semantic_similarity: float = 0.25,
        minimum_visual_similarity: float = 0.20,
        minimum_bm25_score: float = 0.10,
        max_candidates_per_retriever: int = 16,
        use_bge: bool = True,
        use_bm25: bool = True,
        use_clip_fallback: bool = True,
        use_complementary_selection: bool = True,
    ) -> None:
        self.index = index
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.candidate_metadata_tokens = max(64, int(candidate_metadata_tokens))
        self.rrf_constant = max(1.0, float(rrf_constant))
        self.visual_weight = max(0.0, float(visual_weight))
        self.redundancy_weight = max(0.0, float(redundancy_weight))
        self.minimum_semantic_similarity = float(minimum_semantic_similarity)
        self.minimum_visual_similarity = float(minimum_visual_similarity)
        self.minimum_bm25_score = max(0.0, float(minimum_bm25_score))
        self.max_candidates_per_retriever = max(1, int(max_candidates_per_retriever))
        self.use_bge = bool(use_bge)
        self.use_bm25 = bool(use_bm25)
        self.use_clip_fallback = bool(use_clip_fallback)
        self.use_complementary_selection = bool(use_complementary_selection)
        self.documents = {item.memory_id: item.searchable_text() for item in index.all_nodes(include_scene_changes=False)}
        self.bm25 = BM25Index(self.documents)

    def search(
        self,
        needs: Sequence[EvidenceNeed],
        specs: Sequence[SearchSpec],
        exclude_ids: Iterable[str] = (),
        reference_ids: Iterable[str] = (),
    ) -> list[CandidateHit]:
        excluded = set(exclude_ids)
        need_map = {item.need_id: item for item in needs}
        candidates: dict[str, CandidateHit] = {}
        for spec in specs:
            need = need_map.get(spec.need_id)
            if need is None or not need.requires_history:
                continue
            query = _clean_text(spec.query, 400) or need.search_text()
            nodes = [
                item
                for item in self.index.all_nodes(include_scene_changes=False)
                if item.memory_id not in excluded
                and item.kind in set(spec.memory_types or ["overview", "transition"])
                and self._passes_temporal_filter(item, spec)
            ]
            if not nodes:
                continue
            semantic: dict[str, float] = {}
            if self.use_bge:
                query_vector = self.text_encoder.encode([query])[0]
                semantic = {
                    item.memory_id: _cosine(query_vector, self.index.text_embeddings.get(item.memory_id)) for item in nodes
                }
            lexical: dict[str, float] = {}
            if self.use_bm25:
                lexical_all = self.bm25.scores(" ".join([query, *need.entities, *need.actions]))
                lexical = {item.memory_id: lexical_all.get(item.memory_id, 0.0) for item in nodes}
            visual: dict[str, float] = {}
            if self.use_clip_fallback and need.need_type in self.VISUAL_NEEDS:
                query_visual = self.image_encoder.text_embedding(query)
                visual = {
                    item.memory_id: _cosine(query_visual, self.index.visual_embeddings.get(item.memory_id)) for item in nodes
                }
            semantic_ranks = self._ranks(semantic, limit=self.max_candidates_per_retriever)
            lexical_ranks = self._ranks(lexical, positive_only=True, limit=self.max_candidates_per_retriever)
            visual_ranks = self._ranks(visual, limit=self.max_candidates_per_retriever) if visual else {}
            for node in nodes:
                memory_id = node.memory_id
                component_scores = {
                    "bge": float(semantic.get(memory_id, 0.0)),
                    "bm25": float(lexical.get(memory_id, 0.0)),
                    "clip": float(visual.get(memory_id, 0.0)),
                }
                passes_absolute_gate = (
                    (self.use_bge and component_scores["bge"] >= self.minimum_semantic_similarity)
                    or (self.use_bm25 and component_scores["bm25"] >= self.minimum_bm25_score)
                    or (bool(visual) and component_scores["clip"] >= self.minimum_visual_similarity)
                )
                if not passes_absolute_gate:
                    continue
                score = 0.0
                if memory_id in semantic_ranks:
                    score += 1.0 / (self.rrf_constant + semantic_ranks[memory_id])
                if memory_id in lexical_ranks:
                    score += 1.0 / (self.rrf_constant + lexical_ranks[memory_id])
                if memory_id in visual_ranks:
                    score += self.visual_weight / (self.rrf_constant + visual_ranks[memory_id])
                if score <= 0:
                    continue
                if need.need_type in {"transition", "temporal_order", "neighbor_event", "count"} and node.kind == "transition":
                    score += 0.5 / (self.rrf_constant + 1.0)
                elif need.need_type == "history_event" and node.kind == "overview":
                    score += 0.25 / (self.rrf_constant + 1.0)
                hit = candidates.setdefault(
                    memory_id,
                    CandidateHit(memory_id, node.kind, node.start_time, node.end_time, node.searchable_text()),
                )
                hit.need_scores[need.need_id] = score
                hit.component_scores[need.need_id] = component_scores
                hit.ranks[need.need_id] = {
                    key: rank
                    for key, rank in {
                        "bge": semantic_ranks.get(memory_id),
                        "bm25": lexical_ranks.get(memory_id),
                        "clip": visual_ranks.get(memory_id),
                    }.items()
                    if rank is not None
                }
        rows = list(candidates.values())
        return (
            self._select_complementary(rows, needs, reference_ids)
            if self.use_complementary_selection
            else self._select_by_rank(rows)
        )

    def _passes_temporal_filter(self, node: MemoryNode, spec: SearchSpec) -> bool:
        if spec.anchor_memory_id is None or spec.temporal_relation not in {"before", "after"}:
            return True
        anchor = self.index.get(spec.anchor_memory_id)
        if anchor is None:
            return False
        return node.end_time <= anchor.start_time if spec.temporal_relation == "before" else node.start_time >= anchor.end_time

    @staticmethod
    def _ranks(scores: dict[str, float], positive_only: bool = False, limit: int | None = None) -> dict[str, int]:
        rows = [(score, key) for key, score in scores.items() if not positive_only or score > 0]
        rows.sort(key=lambda item: (-item[0], item[1]))
        if limit is not None:
            rows = rows[: max(0, int(limit))]
        return {key: rank for rank, (_, key) in enumerate(rows, start=1)}

    def _select_complementary(
        self,
        hits: list[CandidateHit],
        needs: Sequence[EvidenceNeed],
        reference_ids: Iterable[str] = (),
    ) -> list[CandidateHit]:
        if not hits:
            return []
        non_current = [item for item in needs if item.requires_history]
        maxima = {
            need.need_id: max((hit.need_scores.get(need.need_id, 0.0) for hit in hits), default=0.0)
            for need in non_current
        }
        targets = {need.need_id: self._evidence_target(need) for need in non_current}
        selected: list[CandidateHit] = []
        references: list[CandidateHit] = []
        for memory_id in dict.fromkeys(str(item) for item in reference_ids):
            node = self.index.get(memory_id)
            if node is not None:
                references.append(
                    CandidateHit(memory_id, node.kind, node.start_time, node.end_time, node.searchable_text())
                )
        covered_slots = {need.need_id: 0 for need in non_current}
        remaining = list(hits)
        used_tokens = 0
        while remaining:
            best: CandidateHit | None = None
            best_utility = -math.inf
            for hit in remaining:
                gain_numerator = 0.0
                priority_total = 0.0
                for need in non_current:
                    weight = float(4 - need.priority)
                    maximum = maxima[need.need_id]
                    relevance = hit.need_scores.get(need.need_id, 0.0) / maximum if maximum > 0 else 0.0
                    if covered_slots[need.need_id] < targets[need.need_id]:
                        gain_numerator += weight * relevance / targets[need.need_id]
                    priority_total += weight
                gain = gain_numerator / max(priority_total, 1.0)
                redundancy = max(
                    (self._redundancy(hit, other) for other in [*references, *selected]), default=0.0
                )
                utility = gain - self.redundancy_weight * redundancy
                if utility > best_utility or (utility == best_utility and best and hit.memory_id < best.memory_id):
                    best, best_utility = hit, utility
            if best is None or best_utility <= 0:
                break
            cost = max(1, len(_tokens(best.text))) + 8
            if selected and used_tokens + cost > self.candidate_metadata_tokens:
                break
            best.utility = best_utility
            selected.append(best)
            used_tokens += cost
            for need in non_current:
                if best.need_scores.get(need.need_id, 0.0) > 0 and covered_slots[need.need_id] < targets[need.need_id]:
                    covered_slots[need.need_id] += 1
            remaining.remove(best)
        return selected

    @staticmethod
    def _evidence_target(need: EvidenceNeed) -> int:
        if need.need_type == "count" or need.temporal_relation == "repeated":
            return 3
        if need.need_type == "temporal_order":
            return 2
        return 1

    def _select_by_rank(self, hits: list[CandidateHit]) -> list[CandidateHit]:
        rows = sorted(hits, key=lambda item: (-max(item.need_scores.values(), default=0.0), item.memory_id))
        selected: list[CandidateHit] = []
        used_tokens = 0
        for hit in rows:
            cost = max(1, len(_tokens(hit.text))) + 8
            if selected and used_tokens + cost > self.candidate_metadata_tokens:
                break
            hit.utility = max(hit.need_scores.values(), default=0.0)
            selected.append(hit)
            used_tokens += cost
        return selected

    def _redundancy(self, left: CandidateHit, right: CandidateHit) -> float:
        text = max(0.0, _cosine(self.index.text_embeddings.get(left.memory_id), self.index.text_embeddings.get(right.memory_id)))
        visual = max(0.0, _cosine(self.index.visual_embeddings.get(left.memory_id), self.index.visual_embeddings.get(right.memory_id)))
        temporal = _temporal_iou(left.start_time, left.end_time, right.start_time, right.end_time)
        return 0.5 * text + 0.25 * visual + 0.25 * temporal


def _temporal_iou(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if union > 0 else float(left_start == right_start)


@dataclass
class WorkingEvidence:
    evidence_id: str
    need_ids: list[str]
    source_type: str
    memory_ids: list[str]
    start_time: float
    end_time: float
    text: str
    provenance_frame_ids: list[str]
    retrieval_score: float = 0.0
    status: str = "retrieved"


class WorkingEvidenceTable:
    def __init__(self) -> None:
        self.entries: dict[str, WorkingEvidence] = {}
        self._next_id = 0

    def add(
        self,
        need_ids: Sequence[str],
        source_type: str,
        memory_ids: Sequence[str],
        start_time: float,
        end_time: float,
        text: str,
        provenance_frame_ids: Sequence[str] = (),
        retrieval_score: float = 0.0,
        status: str = "retrieved",
    ) -> WorkingEvidence:
        cleaned = _clean_text(text, 1200)
        normalized_memories = sorted(set(memory_ids))
        for entry in self.entries.values():
            if entry.status == status and entry.memory_ids == normalized_memories and entry.text == cleaned:
                entry.need_ids = sorted(set(entry.need_ids) | set(need_ids))
                entry.retrieval_score = max(entry.retrieval_score, float(retrieval_score))
                return entry
        self._next_id += 1
        entry = WorkingEvidence(
            evidence_id=f"W{self._next_id:04d}",
            need_ids=sorted(set(need_ids)),
            source_type=source_type,
            memory_ids=normalized_memories,
            start_time=float(start_time),
            end_time=float(end_time),
            text=cleaned,
            provenance_frame_ids=sorted(set(provenance_frame_ids)),
            retrieval_score=float(retrieval_score),
            status=status,
        )
        self.entries[entry.evidence_id] = entry
        return entry

    def for_memory(self, memory_id: str) -> list[WorkingEvidence]:
        return [item for item in self.entries.values() if memory_id in item.memory_ids]

    def for_need(self, need_id: str, include_rejected: bool = False) -> list[WorkingEvidence]:
        return [
            item
            for item in self.entries.values()
            if need_id in item.need_ids and (include_rejected or item.status != "rejected")
        ]

    def context(self, token_budget: int = 1000) -> str:
        rows: list[str] = []
        used = 0
        ordered = sorted(
            self.entries.values(),
            key=lambda item: (
                {"compared": 0, "inspected": 1, "partial": 2, "retrieved": 3, "rejected": 4}.get(item.status, 5),
                -item.retrieval_score,
                item.start_time,
            ),
        )
        for item in ordered:
            row = (
                f"- evidence_id={item.evidence_id}; needs={','.join(item.need_ids) or 'none'}; "
                f"memory_ids={','.join(item.memory_ids)}; time={item.start_time:.1f}-{item.end_time:.1f}; "
                f"status={item.status}; text={item.text}"
            )
            cost = len(_tokens(row))
            if rows and used + cost > token_budget:
                continue
            rows.append(row)
            used += cost
        return "\n".join(rows) or "(empty)"


ALLOWED_TOOL_ACTIONS = {"search_memory", "inspect_memory", "expand_memory", "compare_memory", "finish_retrieval"}


@dataclass
class BudgetedToolCall:
    action: str
    payload: dict[str, Any]
    raw: str = ""


def parse_budgeted_tool_call(raw: str) -> BudgetedToolCall | None:
    payload = _parse_json_object(raw)
    if payload is None:
        return None
    action = str(payload.get("action", "")).strip()
    if action not in ALLOWED_TOOL_ACTIONS:
        return None
    return BudgetedToolCall(action=action, payload=payload, raw=str(raw))


@dataclass
class BudgetedTrace:
    needs: list[EvidenceNeed] = field(default_factory=list)
    relations: list[TemporalRelationNeed] = field(default_factory=list)
    calls: list[BudgetedToolCall] = field(default_factory=list)
    retrievals: list[dict[str, Any]] = field(default_factory=list)
    working_evidence: list[dict[str, Any]] = field(default_factory=list)
    selected_memory_ids: list[str] = field(default_factory=list)
    planner_calls: int = 0
    controller_calls: int = 0
    perception_calls: int = 0
    final_calls: int = 0
    controller_failures: int = 0
    executed_tools: int = 0
    history_visual_frames: int = 0
    final_recent_frames: int = 0
    search_rounds: int = 0
    navigation_steps: int = 0
    visual_actions: int = 0
    fast_path: bool = False
    controller_mode: str = "llm"
    need_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    relation_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    planner_raw: str = ""
    budget_exhausted: bool = False
    retrieval_stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs": [asdict(item) for item in self.needs],
            "relations": [asdict(item) for item in self.relations],
            "calls": [{"action": item.action, "payload": item.payload, "raw": item.raw} for item in self.calls],
            "retrievals": self.retrievals,
            "working_evidence": self.working_evidence,
            "selected_memory_ids": self.selected_memory_ids,
            "planner_calls": self.planner_calls,
            "controller_calls": self.controller_calls,
            "perception_calls": self.perception_calls,
            "final_calls": self.final_calls,
            "controller_failures": self.controller_failures,
            "executed_tools": self.executed_tools,
            "history_visual_frames": self.history_visual_frames,
            "final_recent_frames": self.final_recent_frames,
            "search_rounds": self.search_rounds,
            "navigation_steps": self.navigation_steps,
            "visual_actions": self.visual_actions,
            "fast_path": self.fast_path,
            "controller_mode": self.controller_mode,
            "need_states": self.need_states,
            "relation_states": self.relation_states,
            "planner_raw": self.planner_raw,
            "budget_exhausted": self.budget_exhausted,
            "retrieval_stop_reason": self.retrieval_stop_reason,
        }


class BudgetedVeriStreamAgent:
    """Restricted retrieval agent over a coverage/change memory index."""

    INSPECTION_MODES = {"temporal", "state", "attribute", "spatial", "ocr"}
    EXPANSION_RELATIONS = {"before", "after", "both", "parent", "children"}
    COMPARISON_MODES = {"temporal_order", "state_change", "identity", "spatial_change"}
    COMPARISON_RELATIONS = {
        "A_before_B", "B_before_A", "same", "changed", "different", "uncertain"
    }

    def __init__(
        self,
        perception: FrameModel,
        reasoning: TextModel,
        frame_store: RawFrameStore,
        index: CoverageChangeIndex,
        retriever: MemoryRetriever,
        image_encoder: ImageEmbeddingModel,
        max_tool_actions: int = 4,
        max_history_visual_frames: int = 8,
        history_context_tokens: int = 768,
        controller_mode: str = "llm",
        max_search_rounds: int = 2,
        max_navigation_steps: int = 2,
        max_visual_actions: int = 2,
        current_fast_path_confidence: float = 0.7,
        visual_verification_policy: str = "deterministic",
    ) -> None:
        if frame_store.video_id != index.video_id:
            raise ValueError("frame store and index video IDs differ")
        self.perception = perception
        self.reasoning = reasoning
        self.frame_store = frame_store
        self.index = index
        self.retriever = retriever
        self.image_encoder = image_encoder
        self.max_tool_actions = max(1, int(max_tool_actions))
        self.max_history_visual_frames = max(0, int(max_history_visual_frames))
        self.history_context_tokens = max(64, int(history_context_tokens))
        if controller_mode not in {"deterministic", "llm"}:
            raise ValueError("controller_mode must be 'deterministic' or 'llm'")
        self.controller_mode = controller_mode
        self.max_search_rounds = max(1, int(max_search_rounds))
        self.max_navigation_steps = max(0, int(max_navigation_steps))
        self.max_visual_actions = max(0, int(max_visual_actions))
        self.current_fast_path_confidence = max(0.0, min(1.0, float(current_fast_path_confidence)))
        if visual_verification_policy not in {"deterministic", "planner"}:
            raise ValueError("visual_verification_policy must be 'deterministic' or 'planner'")
        self.visual_verification_policy = visual_verification_policy

    def answer(
        self,
        question: str,
        options: str = "",
        baseline_prompt: str | None = None,
        current_images: Sequence[Image.Image] | None = None,
    ) -> tuple[str, BudgetedTrace]:
        trace = BudgetedTrace(controller_mode=self.controller_mode)
        planner_raw = self.reasoning.generate_from_text(self._planner_prompt(question, options))
        trace.planner_raw = str(planner_raw)
        trace.planner_calls += 1
        plan = apply_neighbor_event_policy(parse_evidence_plan(planner_raw, question), question)
        plan = apply_visual_verification_policy(plan, self.visual_verification_policy)
        needs = plan.needs
        relations = plan.relations
        for need in needs:
            if need.is_current_only and need.confidence < self.current_fast_path_confidence:
                need.scope = "both"
                need.scope_override_reason = "low_scope_confidence"
        trace.needs = needs
        trace.relations = relations
        table = WorkingEvidenceTable()
        selected_memory_ids: list[str] = []
        tool_history: list[str] = []
        history_frames_used = 0
        current_fast_path = not relations and bool(needs) and all(
            item.is_current_only and item.confidence >= self.current_fast_path_confidence for item in needs
        )
        if current_fast_path:
            trace.fast_path = True
        elif any(item.requires_history for item in needs) and self.controller_mode == "deterministic":
            selected_memory_ids, history_frames_used = self._run_deterministic_retrieval(
                question, needs, relations, table, trace
            )
        elif any(item.requires_history for item in needs):
            for _ in range(self.max_tool_actions + 1):
                remaining_actions = self.max_tool_actions - trace.executed_tools
                remaining_frames = self.max_history_visual_frames - history_frames_used
                raw = self.reasoning.generate_from_text(
                    self._controller_prompt(
                        question,
                        options,
                        needs,
                        relations,
                        table,
                        tool_history,
                        remaining_actions,
                        remaining_frames,
                        self.max_visual_actions - trace.visual_actions,
                        self.max_search_rounds - trace.search_rounds,
                        self.max_navigation_steps - trace.navigation_steps,
                    )
                )
                trace.controller_calls += 1
                call = parse_budgeted_tool_call(raw)
                if call is None:
                    trace.controller_failures += 1
                    repair = self.reasoning.generate_from_text(self._repair_prompt(raw))
                    trace.controller_calls += 1
                    call = parse_budgeted_tool_call(repair)
                if call is None:
                    call = self._fallback_call(needs, table)
                trace.calls.append(call)
                if call.action == "finish_retrieval":
                    selected_memory_ids = self._valid_selected_ids(
                        call.payload.get("selected_memory_ids", []), table, needs
                    )
                    trace.retrieval_stop_reason = "controller_finish"
                    break
                if trace.executed_tools >= self.max_tool_actions:
                    trace.budget_exhausted = True
                    trace.retrieval_stop_reason = "tool_budget_exhausted"
                    break
                if call.action == "search_memory":
                    if trace.search_rounds >= self.max_search_rounds:
                        result = "search_memory stopped at the search-round budget. Choose another action or finish."
                    else:
                        result = self._execute_search(call, needs, table, trace)
                        trace.search_rounds += 1
                elif call.action == "inspect_memory":
                    if trace.visual_actions >= self.max_visual_actions:
                        result, consumed = "inspect_memory stopped at the visual-action budget.", 0
                    else:
                        result, consumed = self._execute_inspect(call, needs, table, remaining_frames, trace)
                    history_frames_used += consumed
                    if consumed > 0:
                        trace.visual_actions += 1
                elif call.action == "expand_memory":
                    if trace.navigation_steps >= self.max_navigation_steps:
                        result = "expand_memory stopped at the navigation-step budget. Choose another action or finish."
                    else:
                        result = self._execute_expand(call, needs, table)
                        trace.navigation_steps += 1
                else:
                    if trace.visual_actions >= self.max_visual_actions:
                        result, consumed = "compare_memory stopped at the visual-action budget.", 0
                    else:
                        result, consumed = self._execute_compare(call, needs, table, remaining_frames, trace)
                    history_frames_used += consumed
                    if consumed > 0:
                        trace.visual_actions += 1
                trace.executed_tools += 1
                tool_history.append(result)
            if relations:
                trace.relation_states = self._evaluate_temporal_relations(relations, needs, table)
        trace.history_visual_frames = history_frames_used
        if not selected_memory_ids:
            selected_memory_ids = self._default_selected_ids(table, needs)
        trace.selected_memory_ids = selected_memory_ids
        historical_context = self._final_history_context(table, selected_memory_ids, needs)
        recent_images = list(current_images) if current_images is not None else self.frame_store.recent_images()
        final_prompt = (
            baseline_prompt or self._recent_only_prompt(question, options)
            if current_fast_path
            else self._final_prompt(question, options, historical_context)
        )
        if recent_images and hasattr(self.reasoning, "generate_from_frames"):
            response = self.reasoning.generate_from_frames(recent_images, final_prompt)
        else:
            response = self.reasoning.generate_from_text(final_prompt)
        trace.final_calls += 1
        trace.final_recent_frames = len(recent_images)
        trace.working_evidence = [asdict(item) for item in table.entries.values()]
        trace.need_states = self._summarize_need_states(needs, table)
        if relations and not trace.relation_states:
            trace.relation_states = {
                item.relation_id: {
                    "state": "unresolved",
                    "satisfied": False,
                    "source_need_id": item.source_need_id,
                    "target_need_id": item.target_need_id,
                }
                for item in relations
            }
        if not trace.retrieval_stop_reason:
            trace.retrieval_stop_reason = "fast_path" if current_fast_path else "retrieval_complete"
        return str(response).strip(), trace

    def _run_deterministic_retrieval(
        self,
        question: str,
        needs: Sequence[EvidenceNeed],
        relations: Sequence[TemporalRelationNeed],
        table: WorkingEvidenceTable,
        trace: BudgetedTrace,
    ) -> tuple[list[str], int]:
        historical_needs = sorted(
            (item for item in needs if item.requires_history), key=lambda item: (item.priority, item.need_id)
        )
        if trace.executed_tools < self.max_tool_actions:
            initial_call = self._automatic_search_call(historical_needs)
            trace.calls.append(initial_call)
            self._execute_search(initial_call, needs, table, trace)
            trace.search_rounds += 1
            trace.executed_tools += 1

        missing = [item for item in historical_needs if not self._has_enough_locations(item, table)]
        need_map = {item.need_id: item for item in historical_needs}
        for relation in relations:
            if self._relation_has_distinct_locations(relation, table):
                continue
            for need_id in (relation.source_need_id, relation.target_need_id):
                need = need_map.get(need_id)
                if need is not None and need not in missing:
                    missing.append(need)
        if (
            missing
            and trace.search_rounds < self.max_search_rounds
            and trace.executed_tools < self.max_tool_actions
        ):
            retry_call = self._automatic_search_call(missing, fallback_query=question)
            trace.calls.append(retry_call)
            self._execute_search(retry_call, needs, table, trace)
            trace.search_rounds += 1
            trace.executed_tools += 1

        for need in historical_needs:
            if (
                trace.navigation_steps >= self.max_navigation_steps
                or trace.executed_tools >= self.max_tool_actions
            ):
                break
            entries = self._ranked_need_entries(table, need.need_id)
            if not entries:
                continue
            memory_id = entries[0].memory_ids[0]
            node = self.index.get(memory_id)
            relation: str | None = None
            if need.need_type in {"transition", "temporal_order", "count"} and isinstance(node, OverviewMemory):
                relation = "children"
            elif need.temporal_relation in {"before", "after"}:
                relation = need.temporal_relation
            if relation is None:
                continue
            expand_call = BudgetedToolCall(
                "expand_memory",
                {"action": "expand_memory", "memory_id": memory_id, "need_id": need.need_id, "relation": relation},
            )
            trace.calls.append(expand_call)
            self._execute_expand(expand_call, needs, table)
            trace.navigation_steps += 1
            trace.executed_tools += 1

        endpoint_assignments = self._relation_endpoint_assignments(relations, table)
        history_frames_used = 0
        graph_need_ids = {
            need_id
            for relation in relations
            for need_id in (relation.source_need_id, relation.target_need_id)
        }
        inspection_needs = sorted(
            historical_needs,
            key=lambda item: (item.need_id not in graph_need_ids, item.priority, item.need_id),
        )
        for need in inspection_needs:
            if not need.visual_verification or trace.visual_actions >= self.max_visual_actions:
                continue
            desired = 2 if need.need_type in {"temporal_order", "count"} else 1
            ranked_ids = self._ranked_need_memory_ids(table, need.need_id)
            assigned = endpoint_assignments.get(need.need_id)
            memory_ids = list(dict.fromkeys(([assigned] if assigned else []) + ranked_ids))[:desired]
            for memory_id in memory_ids:
                if (
                    trace.visual_actions >= self.max_visual_actions
                    or history_frames_used >= self.max_history_visual_frames
                    or trace.executed_tools >= self.max_tool_actions
                ):
                    break
                inspect_call = BudgetedToolCall(
                    "inspect_memory",
                    {
                        "action": "inspect_memory",
                        "requests": [
                            {
                                "memory_id": memory_id,
                                "need_id": need.need_id,
                                "mode": self._inspection_mode_for_need(need),
                                "focus": need.search_text(),
                            }
                        ],
                    },
                )
                trace.calls.append(inspect_call)
                _, consumed = self._execute_inspect(
                    inspect_call,
                    needs,
                    table,
                    self.max_history_visual_frames - history_frames_used,
                    trace,
                )
                trace.executed_tools += 1
                if consumed <= 0:
                    continue
                history_frames_used += consumed
                trace.visual_actions += 1

        trace.relation_states = self._evaluate_temporal_relations(relations, needs, table)
        for need in historical_needs:
            if need.need_type != "temporal_order" or need.need_id in graph_need_ids:
                continue
            memory_ids = self._ranked_need_memory_ids(table, need.need_id)
            if len(memory_ids) < 2:
                continue
            first, second = sorted(
                (self.index.get(item) for item in memory_ids[:2]),
                key=lambda item: (item.start_time, item.end_time) if item is not None else (math.inf, math.inf),
            )
            if first is None or second is None:
                continue
            relation = "ordered_non_overlapping" if first.end_time <= second.start_time else "overlapping_or_uncertain"
            relation_text = (
                f"{first.memory_id} precedes {second.memory_id}"
                if relation == "ordered_non_overlapping"
                else f"{first.memory_id} and {second.memory_id} overlap or cannot be strictly ordered"
            )
            table.add(
                [need.need_id],
                "temporal_order",
                [first.memory_id, second.memory_id],
                first.start_time,
                second.end_time,
                f"Timestamp relation: {relation_text}. Intervals: {first.memory_id} "
                f"({first.start_time:.1f}-{first.end_time:.1f}s), {second.memory_id} "
                f"({second.start_time:.1f}-{second.end_time:.1f}s).",
                status="compared" if relation == "ordered_non_overlapping" else "partial",
            )

        selected = self._default_selected_ids(table, needs)
        states = self._summarize_need_states(historical_needs, table)
        finish_call = BudgetedToolCall(
            "finish_retrieval",
            {
                "action": "finish_retrieval",
                "satisfied_need_ids": [
                    item.need_id for item in historical_needs if states[item.need_id]["satisfied"]
                ],
                "satisfied_relation_ids": [
                    relation_id
                    for relation_id, state in trace.relation_states.items()
                    if state.get("satisfied")
                ],
                "selected_memory_ids": selected,
                "reason": "Deterministic evidence-state policy reached its evidence or budget boundary.",
            },
        )
        trace.calls.append(finish_call)
        states = self._summarize_need_states(historical_needs, table)
        all_needs_satisfied = all(state["satisfied"] for state in states.values())
        all_relations_satisfied = all(
            state.get("satisfied", False) for state in trace.relation_states.values()
        )
        all_evidence_satisfied = all_needs_satisfied and all_relations_satisfied
        if trace.executed_tools >= self.max_tool_actions and not all_evidence_satisfied:
            trace.budget_exhausted = True
            trace.retrieval_stop_reason = "tool_budget_exhausted"
        else:
            trace.retrieval_stop_reason = (
                "evidence_satisfied" if all_evidence_satisfied else "no_admissible_action"
            )
        return selected, history_frames_used

    def _relation_has_distinct_locations(
        self,
        relation: TemporalRelationNeed,
        table: WorkingEvidenceTable,
    ) -> bool:
        source_ids = self._ranked_need_memory_ids(table, relation.source_need_id)[:4]
        target_ids = self._ranked_need_memory_ids(table, relation.target_need_id)[:4]
        return self._best_relation_pair(source_ids, target_ids) is not None

    def _best_relation_pair(
        self,
        source_ids: Sequence[str],
        target_ids: Sequence[str],
    ) -> tuple[str, str] | None:
        candidates: list[tuple[tuple[float, ...], tuple[str, str]]] = []
        for source_rank, source_id in enumerate(source_ids):
            source = self.index.get(source_id)
            if source is None:
                continue
            for target_rank, target_id in enumerate(target_ids):
                target = self.index.get(target_id)
                if target is None or source_id == target_id or self._same_event_family(source, target):
                    continue
                overview_penalty = float(isinstance(source, OverviewMemory)) + float(
                    isinstance(target, OverviewMemory)
                )
                overlap = _temporal_iou(source.start_time, source.end_time, target.start_time, target.end_time)
                score = (
                    float(source_rank + target_rank),
                    overview_penalty,
                    overlap,
                    float(abs(source.start_time - target.start_time) < 1e-6),
                )
                candidates.append((score, (source_id, target_id)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (*item[0], *item[1]))
        return candidates[0][1]

    @staticmethod
    def _same_event_family(source: MemoryNode, target: MemoryNode) -> bool:
        if isinstance(source, TransitionMemory) and isinstance(target, OverviewMemory):
            return source.parent_overview_id == target.memory_id
        if isinstance(source, OverviewMemory) and isinstance(target, TransitionMemory):
            return target.parent_overview_id == source.memory_id
        return False

    def _relation_endpoint_assignments(
        self,
        relations: Sequence[TemporalRelationNeed],
        table: WorkingEvidenceTable,
    ) -> dict[str, str]:
        assignments: dict[str, str] = {}
        for relation in relations:
            source_ids = self._ranked_need_memory_ids(table, relation.source_need_id)[:4]
            target_ids = self._ranked_need_memory_ids(table, relation.target_need_id)[:4]
            pair = self._best_relation_pair(source_ids, target_ids)
            if pair is not None:
                assignments.setdefault(relation.source_need_id, pair[0])
                assignments.setdefault(relation.target_need_id, pair[1])
        return assignments

    def _evaluate_temporal_relations(
        self,
        relations: Sequence[TemporalRelationNeed],
        needs: Sequence[EvidenceNeed],
        table: WorkingEvidenceTable,
    ) -> dict[str, dict[str, Any]]:
        need_states = self._summarize_need_states(needs, table)
        need_map = {item.need_id: item for item in needs}
        result: dict[str, dict[str, Any]] = {}
        for relation in relations:
            base = {
                "source_need_id": relation.source_need_id,
                "target_need_id": relation.target_need_id,
                "relation_type": relation.relation_type,
                "satisfied": False,
            }
            if not need_states.get(relation.source_need_id, {}).get("satisfied") or not need_states.get(
                relation.target_need_id, {}
            ).get("satisfied"):
                result[relation.relation_id] = {**base, "state": "endpoints_not_verified"}
                continue
            source_need = need_map.get(relation.source_need_id)
            target_need = need_map.get(relation.target_need_id)
            source_ids = [
                memory_id
                for memory_id in self._ranked_need_memory_ids(table, relation.source_need_id)[:4]
                if source_need is not None
                and self._memory_ready_for_need(relation.source_need_id, memory_id, source_need, table)
            ]
            target_ids = [
                memory_id
                for memory_id in self._ranked_need_memory_ids(table, relation.target_need_id)[:4]
                if target_need is not None
                and self._memory_ready_for_need(relation.target_need_id, memory_id, target_need, table)
            ]
            pair = self._best_relation_pair(source_ids, target_ids)
            if pair is None:
                result[relation.relation_id] = {**base, "state": "distinct_endpoints_not_found"}
                continue
            source = self.index.get(pair[0])
            target = self.index.get(pair[1])
            if source is None or target is None:
                result[relation.relation_id] = {**base, "state": "endpoint_memory_missing"}
                continue
            if isinstance(source, TransitionMemory) and isinstance(target, TransitionMemory):
                source_time = source.peak_time
                target_time = target.peak_time
                temporal_basis = "transition_peak_time"
                if relation.relation_type == "before":
                    supported = source_time < target_time
                    contradicted = target_time < source_time
                else:
                    supported = target_time < source_time
                    contradicted = source_time < target_time
            else:
                source_time = (source.start_time + source.end_time) / 2.0
                target_time = (target.start_time + target.end_time) / 2.0
                temporal_basis = "non_overlapping_interval"
                if relation.relation_type == "before":
                    supported = source.end_time <= target.start_time
                    contradicted = target.end_time <= source.start_time
                else:
                    supported = target.end_time <= source.start_time
                    contradicted = source.end_time <= target.start_time
            if supported:
                outcome, status = "supported", "compared"
            elif contradicted:
                outcome, status = "contradicted", "compared"
            else:
                outcome, status = "overlapping_or_uncertain", "partial"
            text = (
                f"Temporal relation {relation.relation_id}: {relation.source_need_id} ({source.memory_id}, "
                f"{source.start_time:.1f}-{source.end_time:.1f}s) {relation.relation_type} "
                f"{relation.target_need_id} ({target.memory_id}, {target.start_time:.1f}-{target.end_time:.1f}s) "
                f"is {outcome}; basis={temporal_basis}, event_times={source_time:.1f}s/{target_time:.1f}s."
            )
            table.add(
                [relation.relation_id],
                "temporal_relation",
                [source.memory_id, target.memory_id],
                min(source.start_time, target.start_time),
                max(source.end_time, target.end_time),
                text,
                status=status,
            )
            result[relation.relation_id] = {
                **base,
                "state": "relation_verified" if status == "compared" else "uncertain",
                "satisfied": status == "compared",
                "outcome": outcome,
                "source_memory_id": source.memory_id,
                "target_memory_id": target.memory_id,
                "source_event_time": source_time,
                "target_event_time": target_time,
                "temporal_basis": temporal_basis,
            }
        return result

    @staticmethod
    def _memory_ready_for_need(
        need_id: str,
        memory_id: str,
        need: EvidenceNeed,
        table: WorkingEvidenceTable,
    ) -> bool:
        entries = [
            entry
            for entry in table.for_need(need_id)
            if memory_id in entry.memory_ids
        ]
        if need.visual_verification:
            return any(entry.status in {"inspected", "compared"} for entry in entries)
        return bool(entries)

    @staticmethod
    def _has_enough_locations(need: EvidenceNeed, table: WorkingEvidenceTable) -> bool:
        memory_ids = {
            memory_id
            for entry in table.for_need(need.need_id)
            for memory_id in entry.memory_ids
        }
        target = 3 if need.need_type == "count" or need.temporal_relation == "repeated" else 1
        if need.need_type == "temporal_order":
            target = 2
        return len(memory_ids) >= target

    @staticmethod
    def _summarize_need_states(
        needs: Sequence[EvidenceNeed],
        table: WorkingEvidenceTable,
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for need in needs:
            entries = table.for_need(need.need_id)
            statuses = {entry.status for entry in entries}
            memory_ids = sorted({memory_id for entry in entries for memory_id in entry.memory_ids})
            if need.is_current_only:
                state = "current_lane"
                satisfied = True
            elif "compared" in statuses:
                state = "relation_verified"
                satisfied = True
            elif "inspected" in statuses:
                state = "visually_verified"
                satisfied = need.need_type != "temporal_order"
            elif entries:
                state = "located"
                satisfied = not need.visual_verification and need.need_type != "temporal_order"
            else:
                state = "unlocated"
                satisfied = False
            result[need.need_id] = {
                "state": state,
                "satisfied": satisfied,
                "memory_ids": memory_ids,
            }
        return result

    @staticmethod
    def _automatic_search_call(
        needs: Sequence[EvidenceNeed], fallback_query: str = ""
    ) -> BudgetedToolCall:
        requests = [
            {
                "need_id": item.need_id,
                "query": " ".join(part for part in (item.search_text(), fallback_query) if part).strip(),
                "memory_types": ["overview", "transition"],
                "temporal_relation": "any",
                "anchor_memory_id": None,
            }
            for item in needs
        ]
        return BudgetedToolCall(
            "search_memory", {"action": "search_memory", "requests": requests, "exclude_memory_ids": []}
        )

    @staticmethod
    def _ranked_need_entries(table: WorkingEvidenceTable, need_id: str) -> list[WorkingEvidence]:
        return sorted(
            table.for_need(need_id),
            key=lambda item: (
                {"compared": 0, "inspected": 1, "partial": 2, "retrieved": 3}.get(item.status, 4),
                -item.retrieval_score,
                item.start_time,
            ),
        )

    def _ranked_need_memory_ids(self, table: WorkingEvidenceTable, need_id: str) -> list[str]:
        return list(
            dict.fromkeys(
                memory_id
                for entry in self._ranked_need_entries(table, need_id)
                for memory_id in entry.memory_ids
                if self.index.get(memory_id) is not None
            )
        )

    @staticmethod
    def _inspection_mode_for_need(need: EvidenceNeed) -> str:
        if need.need_type in {"attribute", "spatial", "state", "ocr"}:
            return need.need_type
        return "temporal"

    @staticmethod
    def _recent_only_prompt(question: str, options: str) -> str:
        option_rows = [item.strip() for item in str(options).splitlines() if item.strip()]
        option_text = "; ".join(option_rows)
        if option_text and not option_text.endswith(";"):
            option_text += ";"
        return f"{question}\nOptions: {option_text}\nOnly give the best option's letter directly."

    @staticmethod
    def _planner_prompt(question: str, options: str) -> str:
        return (
            "You are an evidence planner for causal long-video question answering.\n\n"
            "Decompose the question into the smallest set of observable evidence needs. "
            "Do not answer the question. Do not choose a multiple-choice option. Do not create "
            "separate needs for every answer option unless the options refer to different observable events.\n\n"
            "Assign scope=current only when the evidence is explicitly about what is visible or happening now. "
            "Assign scope=historical for past events, earlier object locations, persistent states established earlier, transitions, "
            "temporal order, or repetition. Assign scope=both when either current frames or historical evidence may be required. "
            "Never rewrite a past-tense question as a current-state query. Questions beginning with 'What did', 'Where did', "
            "'Who did', or containing before, after, first, last, earlier, previously, again, or how many times require historical scope. "
            "For example, 'Where did I put the shoe?' is historical, while 'What is the person doing now?' is current.\n\n"
            "Use visual_verification=true when raw images are needed to verify an event, attribute, spatial relation, state, OCR, "
            "or count. Each query must preserve the question's temporal meaning and describe observable video evidence rather than "
            "an expected answer. Confidence is the confidence in the scope assignment, not answer confidence.\n\n"
            "For a question that compares the order of two distinct events, create one atomic transition need for each event. "
            "Do not combine both events into one temporal_order query. Add a relation whose source_need_id and target_need_id "
            "reference those atomic needs. The relation type states the proposition to test; a later timestamp check may support "
            "or contradict it. Example: N1='the person places the cup', N2='the person leaves', "
            "R1={type:'before', source_need_id:'N1', target_need_id:'N2'}.\n\n"
            f"Question:\n{question}\n\nOptions:\n{options or '(included in the question text)'}\n\n"
            "Return exactly one JSON object:\n"
            '{"needs":[{"need_id":"N1","scope":"current|historical|both",'
            '"evidence_type":"current|history_event|transition|temporal_order|state|attribute|spatial|ocr|count",'
            '"query":"...","entities":["..."],"actions":["..."],"visual_verification":true,'
            '"temporal_relation":"current|before|after|during|first|last|repeated|any","priority":1,"confidence":0.9}],'
            '"relations":[{"relation_id":"R1","type":"before|after","source_need_id":"N1",'
            '"target_need_id":"N2","query":"test whether N1 occurs before N2","confidence":0.9}]}'
        )

    def _controller_prompt(
        self,
        question: str,
        options: str,
        needs: Sequence[EvidenceNeed],
        relations: Sequence[TemporalRelationNeed],
        table: WorkingEvidenceTable,
        tool_history: Sequence[str],
        remaining_actions: int,
        remaining_frames: int,
        remaining_visual_actions: int,
        remaining_search_rounds: int,
        remaining_navigation_steps: int,
    ) -> str:
        need_text = json.dumps([asdict(item) for item in needs], ensure_ascii=False)
        relation_text = json.dumps([asdict(item) for item in relations], ensure_ascii=False)
        return (
            "You are a retrieval controller for causal long-video question answering. "
            "Gather sufficient historical evidence; do not answer the question.\n\n"
            "Use search_memory when a need lacks a reliable temporal anchor. Use inspect_memory when a relevant memory lacks visual detail. "
            "Use expand_memory for evidence before, after, inside, or around a known memory. Use compare_memory when two localized memories "
            "must be related. Use finish_retrieval when the important needs are sufficiently covered or no useful action remains.\n\n"
            "Do not guess timestamps or raw frame IDs. Do not repeat identical calls without new evidence. "
            "Do not choose a multiple-choice answer. Return exactly one JSON tool call.\n\n"
            "Allowed schemas:\n"
            '{"action":"search_memory","requests":[{"need_id":"N1","query":"...","memory_types":["overview","transition"],'
            '"temporal_relation":"any|before|after","anchor_memory_id":null}],"exclude_memory_ids":[]}\n'
            '{"action":"inspect_memory","requests":[{"memory_id":"...","need_id":"N1",'
            '"mode":"temporal|state|attribute|spatial|ocr","focus":"..."}]}\n'
            '{"action":"expand_memory","memory_id":"...","need_id":"N1",'
            '"relation":"before|after|both|parent|children"}\n'
            '{"action":"compare_memory","memory_ids":["...","..."],"need_id":"N1",'
            '"mode":"temporal_order|state_change|identity|spatial_change","target":"..."}\n'
            '{"action":"finish_retrieval","satisfied_need_ids":["N1"],"selected_memory_ids":["..."],"reason":"..."}\n\n'
            f"Question:\n{question}\n\nOptions:\n{options or '(included in the question text)'}\n\n"
            f"Evidence needs:\n{need_text}\n\nTemporal relations:\n{relation_text}\n\n"
            f"Working evidence:\n{table.context()}\n\n"
            f"Previous tool results:\n{json.dumps(list(tool_history[-4:]), ensure_ascii=False)}\n\n"
            f"Remaining executed-tool budget: {remaining_actions}\n"
            f"Remaining search-round budget: {remaining_search_rounds}\n"
            f"Remaining navigation-step budget: {remaining_navigation_steps}\n"
            f"Remaining historical visual-frame budget: {remaining_frames}\n"
            f"Remaining visual-action budget: {remaining_visual_actions}"
        )

    @staticmethod
    def _repair_prompt(raw: str) -> str:
        return (
            "Convert the following invalid controller output into exactly one valid JSON tool call. "
            "Use only search_memory, inspect_memory, expand_memory, compare_memory, or finish_retrieval. "
            "Do not add Markdown or explanation.\n\nInvalid output:\n" + str(raw)
        )

    @staticmethod
    def _fallback_call(needs: Sequence[EvidenceNeed], table: WorkingEvidenceTable) -> BudgetedToolCall:
        if not table.entries:
            requests = [
                {"need_id": item.need_id, "query": item.search_text(), "memory_types": ["overview", "transition"], "temporal_relation": "any", "anchor_memory_id": None}
                for item in needs
                if item.requires_history
            ]
            return BudgetedToolCall("search_memory", {"action": "search_memory", "requests": requests, "exclude_memory_ids": []})
        memory_ids = sorted({memory_id for item in table.entries.values() for memory_id in item.memory_ids})
        return BudgetedToolCall("finish_retrieval", {"action": "finish_retrieval", "selected_memory_ids": memory_ids})

    def _execute_search(
        self,
        call: BudgetedToolCall,
        needs: Sequence[EvidenceNeed],
        table: WorkingEvidenceTable,
        trace: BudgetedTrace,
    ) -> str:
        need_map = {item.need_id: item for item in needs}
        raw_specs = call.payload.get("requests", [])
        specs: list[SearchSpec] = []
        if isinstance(raw_specs, list):
            for raw in raw_specs:
                if not isinstance(raw, dict) or raw.get("need_id") not in need_map:
                    continue
                types = [item for item in raw.get("memory_types", []) if item in {"overview", "transition"}]
                temporal = str(raw.get("temporal_relation", "any"))
                specs.append(
                    SearchSpec(
                        need_id=str(raw["need_id"]),
                        query=_clean_text(raw.get("query"), 400) or need_map[str(raw["need_id"])].search_text(),
                        memory_types=types or ["overview", "transition"],
                        temporal_relation=temporal if temporal in {"any", "before", "after"} else "any",
                        anchor_memory_id=(
                            str(raw["anchor_memory_id"])
                            if table.for_memory(str(raw.get("anchor_memory_id", "")))
                            else None
                        ),
                    )
                )
        if not specs:
            specs = [SearchSpec(item.need_id, item.search_text()) for item in needs if item.requires_history]
        exclude = call.payload.get("exclude_memory_ids", [])
        requested_excludes = [str(item) for item in exclude if self.index.get(str(item))] if isinstance(exclude, list) else []
        existing_ids = {
            memory_id for entry in table.entries.values() for memory_id in entry.memory_ids if self.index.get(memory_id)
        }
        exclude_ids = sorted(existing_ids | set(requested_excludes))
        hits = self.retriever.search(needs, specs, exclude_ids, reference_ids=existing_ids)
        for hit in hits:
            need_ids = sorted(hit.need_scores)
            table.add(
                need_ids,
                hit.kind,
                [hit.memory_id],
                hit.start_time,
                hit.end_time,
                hit.text,
                retrieval_score=hit.utility,
                status="retrieved",
            )
            trace.retrievals.append(asdict(hit))
        return f"search_memory returned {len(hits)} complementary L1 memories: {[item.memory_id for item in hits]}"

    def _execute_inspect(
        self,
        call: BudgetedToolCall,
        needs: Sequence[EvidenceNeed],
        table: WorkingEvidenceTable,
        remaining_frames: int,
        trace: BudgetedTrace,
    ) -> tuple[str, int]:
        need_map = {item.need_id: item for item in needs}
        requests = call.payload.get("requests", [])
        if not isinstance(requests, list) or remaining_frames <= 0:
            return "inspect_memory could not run because its requests or visual budget were invalid.", 0
        details: list[str] = []
        consumed = 0
        for request in requests:
            if not isinstance(request, dict) or consumed >= remaining_frames:
                continue
            memory_id = str(request.get("memory_id", ""))
            need_id = str(request.get("need_id", ""))
            mode = str(request.get("mode", "temporal"))
            node = self.index.get(memory_id)
            need = need_map.get(need_id)
            if node is None or not table.for_memory(memory_id) or need is None or mode not in self.INSPECTION_MODES:
                continue
            focus = _clean_text(request.get("focus"), 400) or need.search_text()
            frame_ids = self._inspection_frame_ids(node, mode, focus)
            frame_ids = frame_ids[: max(0, remaining_frames - consumed)]
            frames = self.frame_store.frames_for_ids(frame_ids)
            if not frames:
                continue
            raw = self.perception.generate_from_frames(frames, self._inspection_prompt(node, need, mode, focus, frame_ids))
            trace.perception_calls += 1
            consumed += len(frames)
            payload = _parse_json_object(raw) or {}
            result = str(payload.get("result", "partial")).strip().lower()
            supported_need_ids = payload.get("supported_need_ids")
            explicitly_supported = (
                isinstance(supported_need_ids, list)
                and need_id in {str(item) for item in supported_need_ids}
            )
            if result == "not_visible":
                status = "rejected"
            elif result == "supported" and explicitly_supported:
                status = "inspected"
            else:
                status = "partial"
            observation = _clean_text(payload.get("observation"), 1000) or _clean_text(raw, 1000)
            table.add(
                [need_id], mode, [memory_id], node.start_time, node.end_time, observation,
                provenance_frame_ids=frame_ids, status=status,
            )
            details.append(f"{memory_id}:{status}:{observation}")
        return "inspect_memory results: " + ("; ".join(details) if details else "none"), consumed

    def _inspection_frame_ids(self, node: MemoryNode, mode: str, focus: str) -> list[str]:
        if isinstance(node, TransitionMemory):
            ids = list(node.l0_frame_ids)
            if mode == "temporal" and ids:
                last = self.frame_store.record(ids[-1])
                if last is not None:
                    following = next((item.frame_id for item in self.frame_store.history_records if item.timestamp > last.timestamp), None)
                    if following:
                        ids.append(following)
            return self._ordered_frame_ids(ids)[: 3 if mode == "temporal" else 2]
        if mode == "temporal":
            ids = list(node.change_frame_ids[:2])
            if node.coverage_frame_ids:
                ids.append(node.coverage_frame_ids[-1])
            return self._ordered_frame_ids(ids)[:3]
        if mode == "state":
            return self._uniform_frame_ids(node.l0_frame_ids, 3)
        if mode == "ocr":
            return self._ordered_frame_ids(node.coverage_frame_ids)[:4]
        candidates = node.l0_frame_ids
        query_vector = self.image_encoder.text_embedding(focus)
        ranked = sorted(
            candidates,
            key=lambda item: (-_cosine(query_vector, self.frame_store.clip_embeddings.get(item)), self.frame_store.record(item).timestamp if self.frame_store.record(item) else math.inf),
        )
        if not ranked:
            return []
        selected = [ranked[0]]
        record = self.frame_store.record(ranked[0])
        if record is not None:
            neighbor = min(
                (item for item in self.frame_store.history_records if item.frame_id != ranked[0] and node.start_time <= item.timestamp <= node.end_time),
                key=lambda item: abs(item.timestamp - record.timestamp),
                default=None,
            )
            if neighbor is not None:
                selected.append(neighbor.frame_id)
        return self._ordered_frame_ids(selected)

    def _execute_expand(
        self,
        call: BudgetedToolCall,
        needs: Sequence[EvidenceNeed],
        table: WorkingEvidenceTable,
    ) -> str:
        memory_id = str(call.payload.get("memory_id", ""))
        need_id = str(call.payload.get("need_id", ""))
        relation = str(call.payload.get("relation", ""))
        if (
            not any(item.need_id == need_id for item in needs)
            or relation not in self.EXPANSION_RELATIONS
            or not table.for_memory(memory_id)
        ):
            return "expand_memory parameters were invalid."
        neighbors = self.index.neighbors(memory_id, relation)
        neighbors = [
            item
            for item in neighbors
            if not isinstance(item, TransitionMemory)
            or item.evidence_admissible
        ]
        for node in neighbors:
            table.add([need_id], node.kind, [node.memory_id], node.start_time, node.end_time, node.searchable_text(), status="retrieved")
        return f"expand_memory({memory_id}, {relation}) returned {[item.memory_id for item in neighbors]}"

    def _execute_compare(
        self,
        call: BudgetedToolCall,
        needs: Sequence[EvidenceNeed],
        table: WorkingEvidenceTable,
        remaining_frames: int,
        trace: BudgetedTrace,
    ) -> tuple[str, int]:
        raw_ids = call.payload.get("memory_ids", [])
        memory_ids = [str(item) for item in raw_ids] if isinstance(raw_ids, list) else []
        need_id = str(call.payload.get("need_id", ""))
        mode = str(call.payload.get("mode", ""))
        target = _clean_text(call.payload.get("target"), 400)
        need = next((item for item in needs if item.need_id == need_id), None)
        nodes = [self.index.get(item) for item in memory_ids[:2]]
        if (
            len(memory_ids) != 2
            or any(item is None for item in nodes)
            or any(not table.for_memory(item) for item in memory_ids)
            or need is None
            or mode not in self.COMPARISON_MODES
            or remaining_frames < 2
        ):
            return "compare_memory parameters or visual budget were invalid.", 0
        valid_nodes = [item for item in nodes if item is not None]
        frame_groups = [self._comparison_frame_ids(item) for item in valid_nodes]
        while sum(len(item) for item in frame_groups) > remaining_frames:
            longest = max(range(len(frame_groups)), key=lambda index: len(frame_groups[index]))
            frame_groups[longest] = frame_groups[longest][:-1]
        frame_ids = [frame_id for group in frame_groups for frame_id in group]
        frames = self.frame_store.frames_for_ids(frame_ids)
        if len(frames) < 2:
            return "compare_memory could not load both segments.", 0
        raw = self.perception.generate_from_frames(
            frames, self._comparison_prompt(valid_nodes[0], valid_nodes[1], need, mode, target, frame_groups)
        )
        trace.perception_calls += 1
        payload = _parse_json_object(raw) or {}
        raw_relation = (_clean_text(payload.get("relation"), 80) or "uncertain").lower()
        relation = next(
            (item for item in self.COMPARISON_RELATIONS if item.lower() == raw_relation),
            "uncertain",
        )
        supported_need_ids = payload.get("supported_need_ids")
        explicitly_supported = (
            isinstance(supported_need_ids, list)
            and need_id in {str(item) for item in supported_need_ids}
        )
        observation = _clean_text(payload.get("observation"), 1000) or _clean_text(raw, 1000)
        table.add(
            [need_id], mode, memory_ids, min(item.start_time for item in valid_nodes), max(item.end_time for item in valid_nodes),
            f"Relation={relation}. {observation}", provenance_frame_ids=frame_ids,
            status="compared" if relation != "uncertain" and explicitly_supported else "partial",
        )
        return f"compare_memory result: relation={relation}; observation={observation}", len(frames)

    def _comparison_frame_ids(self, node: MemoryNode) -> list[str]:
        if isinstance(node, TransitionMemory):
            return self._ordered_frame_ids(node.l0_frame_ids)[:2]
        return self._uniform_frame_ids(node.coverage_frame_ids or node.l0_frame_ids, 2)

    def _ordered_frame_ids(self, frame_ids: Sequence[str]) -> list[str]:
        valid = {item for item in frame_ids if self.frame_store.record(item) is not None}
        return sorted(valid, key=lambda item: self.frame_store.record(item).timestamp)

    def _uniform_frame_ids(self, frame_ids: Sequence[str], count: int) -> list[str]:
        ordered = self._ordered_frame_ids(frame_ids)
        if len(ordered) <= count:
            return ordered
        positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)] if count > 1 else [len(ordered) // 2]
        return [ordered[position] for position in positions]

    @staticmethod
    def _inspection_prompt(
        node: MemoryNode,
        need: EvidenceNeed,
        mode: str,
        focus: str,
        frame_ids: Sequence[str],
    ) -> str:
        return (
            "You are inspecting a localized historical video memory.\n\n"
            f"Memory ID: {node.memory_id}\nTime interval: {node.start_time:.1f}-{node.end_time:.1f} seconds\n"
            f"Evidence need ID: {need.need_id}\nEvidence need: {need.search_text()}\n"
            f"Inspection mode: {mode}\nFocus: {focus}\n"
            f"Frame references in temporal order: {list(frame_ids)}\n\n"
            "Describe only directly visible evidence relevant to the evidence need. Do not answer the multiple-choice question. "
            "Do not infer facts outside this interval. If the requested detail is not visible, state that it is not visible.\n\n"
            "Return exactly one JSON object:\n"
            '{"observation":"...","supported_need_ids":["N1"],"visible_entities":["..."],'
            '"visible_actions":["..."],"visible_text":["..."],"frame_references":["..."],'
            '"result":"supported|partial|not_visible"}'
        )

    @staticmethod
    def _comparison_prompt(
        first: MemoryNode,
        second: MemoryNode,
        need: EvidenceNeed,
        mode: str,
        target: str,
        frame_groups: Sequence[Sequence[str]],
    ) -> str:
        return (
            "You are comparing two localized historical video segments.\n\n"
            f"Segment A: memory={first.memory_id}, time={first.start_time:.1f}-{first.end_time:.1f}, frames={list(frame_groups[0])}\n"
            f"Segment B: memory={second.memory_id}, time={second.start_time:.1f}-{second.end_time:.1f}, frames={list(frame_groups[1])}\n\n"
            f"Evidence need: {need.search_text()}\nComparison mode: {mode}\nTarget: {target}\n\n"
            "The images are grouped as Segment A followed by Segment B and ordered by time within each segment. "
            "Compare only directly visible evidence. Do not answer the multiple-choice question. "
            "If the relation cannot be established, return uncertain.\n\n"
            "Return exactly one JSON object:\n"
            '{"relation":"A_before_B|B_before_A|same|changed|different|uncertain",'
            '"observation":"...","supported_need_ids":["N1"],"frame_references":["..."]}'
        )

    def _valid_selected_ids(
        self,
        raw_ids: Any,
        table: WorkingEvidenceTable,
        needs: Sequence[EvidenceNeed],
    ) -> list[str]:
        if not isinstance(raw_ids, list):
            return []
        need_map = {item.need_id: item for item in needs}
        return list(
            dict.fromkeys(
                str(item)
                for item in raw_ids
                if self.index.get(str(item)) is not None
                and any(
                    self._entry_is_admissible(entry, need_map)
                    for entry in table.for_memory(str(item))
                )
            )
        )

    @staticmethod
    def _entry_is_admissible(item: WorkingEvidence, need_map: dict[str, EvidenceNeed]) -> bool:
        if item.status in {"inspected", "compared"}:
            return True
        if item.status != "retrieved":
            return False
        linked_needs = [need_map[need_id] for need_id in item.need_ids if need_id in need_map]
        return bool(linked_needs) and any(not need.visual_verification for need in linked_needs)

    @classmethod
    def _default_selected_ids(
        cls,
        table: WorkingEvidenceTable,
        needs: Sequence[EvidenceNeed],
    ) -> list[str]:
        need_map = {item.need_id: item for item in needs}
        ordered = sorted(
            (item for item in table.entries.values() if cls._entry_is_admissible(item, need_map)),
            key=lambda item: (
                {"compared": 0, "inspected": 1, "partial": 2, "retrieved": 3, "rejected": 4}.get(item.status, 5),
                -item.retrieval_score,
            ),
        )
        return list(dict.fromkeys(memory_id for item in ordered for memory_id in item.memory_ids))

    def _final_history_context(
        self,
        table: WorkingEvidenceTable,
        selected_memory_ids: Sequence[str],
        needs: Sequence[EvidenceNeed],
    ) -> str:
        selected = set(selected_memory_ids)
        need_map = {item.need_id: item for item in needs}
        eligible = [
            item for item in table.entries.values()
            if self._entry_is_admissible(item, need_map)
            and (not selected or selected.intersection(item.memory_ids))
        ]
        detailed_memories = {
            memory_id
            for item in eligible
            if item.status in {"inspected", "compared", "partial"}
            for memory_id in item.memory_ids
        }
        eligible = [
            item for item in eligible
            if item.status != "retrieved" or not detailed_memories.intersection(item.memory_ids)
        ]
        eligible.sort(key=lambda item: (item.start_time, item.end_time, item.evidence_id))
        rows: list[str] = []
        used = 0
        seen: set[str] = set()
        for item in eligible:
            normalized = " ".join(_tokens(item.text))
            if not normalized or normalized in seen:
                continue
            row = f"- time={item.start_time:.1f}-{item.end_time:.1f}; source={item.status}; evidence={item.text}"
            cost = len(_tokens(row))
            if rows and used + cost > self.history_context_tokens:
                continue
            rows.append(row)
            used += cost
            seen.add(normalized)
        return "\n".join(rows) or "(none)"

    @staticmethod
    def _final_prompt(question: str, options: str, historical_context: str) -> str:
        return (
            "You are the final multimodal reasoning model for causal long-video question answering.\n\n"
            "The attached recent frames are the current visual context. Use them as the primary evidence for current actions, "
            "current state, spatial relations, and visible text. The historical evidence describes earlier video observations "
            "with timestamps and provenance. Use it only for earlier events, persistent object history, transitions, counts, "
            "and temporal relations. Do not assume that a historical action is still happening now. If recent visual evidence "
            "conflicts with a historical summary about the current state, trust the recent visual evidence.\n\n"
            f"Question:\n{question}\n\nOptions:\n{options or '(included in the question text)'}\n\n"
            f"Historical evidence:\n{historical_context}\n\n"
            "If this is a multiple-choice question, output only the option letter. Otherwise output the shortest direct answer."
        )
