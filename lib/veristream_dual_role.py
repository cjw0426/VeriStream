"""VeriStream 的感知-推理解耦编排。

本模块不依赖 vLLM。感知端和推理端只要实现相同的最小接口，就可以是
本地 Transformers 模型、两个独立的 vLLM 服务，或单元测试中的假模型。
视频级索引保存可回溯证据，问题级工作集只保存当前问题的少量候选。
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from PIL import Image

from lib.veristream import EvidenceStatus


def _tokens(text: str) -> set[str]:
    """提取训练免调优的中英文检索词元。"""

    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()))


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


@dataclass
class RawEvidenceRef:
    """不可变的原始视频证据指针；原始证据不随记忆淘汰而删除。"""

    video_id: str
    chunk_indices: list[int]
    start_time: float
    end_time: float
    source_uri: str = ""


@dataclass
class ObservationMemory:
    """感知模型写入的局部直接观察。"""

    memory_id: str
    ref: RawEvidenceRef
    observation: str
    entities: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    status: EvidenceStatus = EvidenceStatus.CANDIDATE
    access_count: int = 0
    verified_at: float | None = None
    conflicts_with: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservationMemory":
        copied = dict(payload)
        copied["ref"] = RawEvidenceRef(**copied["ref"])
        copied["status"] = EvidenceStatus(copied.get("status", EvidenceStatus.CANDIDATE.value))
        return cls(**copied)


@dataclass
class EventMemory:
    """由多个局部观察聚合的事件/状态变化记忆。"""

    event_id: str
    video_id: str
    start_time: float
    end_time: float
    summary: str
    entities: list[str]
    actions: list[str]
    evidence_ids: list[str]
    confidence: float = 0.5
    status: EvidenceStatus = EvidenceStatus.CANDIDATE
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventMemory":
        copied = dict(payload)
        copied["status"] = EvidenceStatus(copied.get("status", EvidenceStatus.CANDIDATE.value))
        return cls(**copied)


@dataclass
class MemoryHit:
    """检索命中及其可解释得分。"""

    memory_id: str
    kind: str
    score: float
    text: str
    start_time: float
    end_time: float
    status: EvidenceStatus


@dataclass
class ToolCall:
    """推理模型的受限工具请求。"""

    action: str
    memory_ids: list[str] = field(default_factory=list)
    query: str = ""
    start_time: float | None = None
    end_time: float | None = None
    target: str = ""
    answer: str = ""
    raw: str = ""


@dataclass
class DualRoleTrace:
    """问题级工作流轨迹，供效率和错误分析使用。"""

    calls: list[ToolCall] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    verified_ids: list[str] = field(default_factory=list)
    perception_calls: int = 0
    reasoning_calls: int = 0
    controller_failures: int = 0
    final_vision_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": [asdict(call) for call in self.calls],
            "retrieved_ids": self.retrieved_ids,
            "verified_ids": self.verified_ids,
            "perception_calls": self.perception_calls,
            "reasoning_calls": self.reasoning_calls,
            "controller_failures": self.controller_failures,
            "final_vision_frames": self.final_vision_frames,
        }


class FrameModel(Protocol):
    """感知服务和多模态推理服务共用的最小帧接口。"""

    def generate_from_frames(self, frames: list[Image.Image], prompt: str) -> str: ...


class TextModel(Protocol):
    """推理控制器所需的最小文本接口。"""

    def generate_from_text(self, prompt: str) -> str: ...


class ChunkRepository:
    """当前视频解码后的 chunk 仓库，负责按证据指针回取局部帧。"""

    def __init__(self, video_id: str, chunks: Sequence[Any], source_uri: str = "") -> None:
        self.video_id = video_id
        self.source_uri = source_uri
        self._chunks = {int(chunk.chunk_index): chunk for chunk in chunks}

    def frames_for_indices(self, indices: Sequence[int], max_frames: int = 8) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for index in sorted({int(item) for item in indices}):
            chunk = self._chunks.get(index)
            if chunk is not None:
                frames.extend(list(chunk.frames))
        return self._uniform_sample(frames, max_frames)

    @staticmethod
    def _uniform_sample(frames: Sequence[Image.Image], max_frames: int) -> list[Image.Image]:
        """在候选帧的完整时间范围内均匀采样，避免只看到区间开头。"""

        limit = max(1, int(max_frames))
        candidates = list(frames)
        if len(candidates) <= limit:
            return candidates
        if limit == 1:
            return [candidates[len(candidates) // 2]]
        positions = [round(index * (len(candidates) - 1) / (limit - 1)) for index in range(limit)]
        return [candidates[position] for position in positions]

    def select_interval(self, start_time: float, end_time: float, max_frames: int = 8) -> tuple[list[int], list[Image.Image]]:
        selected = [
            chunk
            for chunk in self._chunks.values()
            if float(chunk.end_time) > float(start_time) and float(chunk.start_time) < float(end_time)
        ]
        selected.sort(key=lambda chunk: int(chunk.chunk_index))
        indices = [int(chunk.chunk_index) for chunk in selected]
        frames: list[Image.Image] = []
        for chunk in selected:
            frames.extend(list(chunk.frames))
        return indices, self._uniform_sample(frames, max_frames)

    def recent_frames(self, max_frames: int = 4) -> list[Image.Image]:
        """返回当前因果视频仓库末端的最近帧，作为最终推理的当前视觉上下文。"""

        if max_frames <= 0:
            return []
        frames: list[Image.Image] = []
        for chunk in sorted(self._chunks.values(), key=lambda item: int(item.chunk_index)):
            frames.extend(list(chunk.frames))
        return frames[-int(max_frames) :]


class VideoEvidenceIndex:
    """按视频隔离的 L0-L2 证据索引。

    L0 是 ``RawEvidenceRef``，L1 是 ``ObservationMemory``，L2 是
    ``EventMemory``。当前问题的检索结果不写回该对象，属于短生命周期 L3。
    """

    def __init__(self, video_id: str, max_observations: int = 256, max_events: int = 96) -> None:
        if not video_id:
            raise ValueError("video_id 不能为空")
        self.video_id = video_id
        self.max_observations = max(1, int(max_observations))
        self.max_events = max(1, int(max_events))
        self.observations: dict[str, ObservationMemory] = {}
        self.events: dict[str, EventMemory] = {}
        self._next_observation_id = 0
        self._next_event_id = 0

    def _new_observation_id(self) -> str:
        self._next_observation_id += 1
        return f"{self.video_id}:m{self._next_observation_id:05d}"

    def _new_event_id(self) -> str:
        self._next_event_id += 1
        return f"{self.video_id}:e{self._next_event_id:05d}"

    def _assert_video(self, ref: RawEvidenceRef) -> None:
        if ref.video_id != self.video_id:
            raise ValueError(f"证据视频 {ref.video_id!r} 与索引 {self.video_id!r} 不一致")

    def _find_duplicate(self, ref: RawEvidenceRef, observation: str) -> ObservationMemory | None:
        observation_tokens = _tokens(observation)
        requested = set(ref.chunk_indices)
        for memory in self.observations.values():
            if memory.status == EvidenceStatus.EVICTED:
                continue
            if not (requested & set(memory.ref.chunk_indices)):
                continue
            if _jaccard(observation_tokens, _tokens(memory.observation)) >= 0.45:
                return memory
        return None

    def add_observation(
        self,
        ref: RawEvidenceRef,
        observation: str,
        entities: Sequence[str] = (),
        actions: Sequence[str] = (),
        confidence: float = 0.5,
    ) -> ObservationMemory:
        """写入 L1 观察；重复观察合并，原始指针始终保留。"""

        self._assert_video(ref)
        cleaned = " ".join(str(observation).split()).strip()
        if not cleaned:
            raise ValueError("观察文本不能为空")
        normalized_entities = sorted({str(item).strip().lower() for item in entities if str(item).strip()})[:12]
        normalized_actions = sorted({str(item).strip().lower() for item in actions if str(item).strip()})[:8]
        duplicate = self._find_duplicate(ref, cleaned)
        now = time.time()
        if duplicate is not None:
            duplicate.ref.chunk_indices = sorted(set(duplicate.ref.chunk_indices) | set(ref.chunk_indices))
            duplicate.ref.start_time = min(duplicate.ref.start_time, ref.start_time)
            duplicate.ref.end_time = max(duplicate.ref.end_time, ref.end_time)
            duplicate.observation = cleaned
            duplicate.entities = sorted(set(duplicate.entities) | set(normalized_entities))
            duplicate.actions = sorted(set(duplicate.actions) | set(normalized_actions))
            duplicate.confidence = max(duplicate.confidence, float(confidence))
            duplicate.updated_at = now
            self._upsert_event(duplicate)
            return duplicate

        memory = ObservationMemory(
            memory_id=self._new_observation_id(),
            ref=ref,
            observation=cleaned,
            entities=normalized_entities,
            actions=normalized_actions,
            confidence=max(0.0, min(1.0, float(confidence))),
        )
        self.observations[memory.memory_id] = memory
        self._upsert_event(memory)
        self.evict_if_needed()
        return memory

    def _upsert_event(self, memory: ObservationMemory) -> EventMemory:
        """用实体/动作和相邻时间合并 L1 观察，得到轻量 L2 事件。"""

        for event in self.events.values():
            nearby = memory.ref.start_time <= event.end_time + 3.0 and memory.ref.end_time >= event.start_time - 3.0
            related = bool(set(memory.entities) & set(event.entities)) or bool(set(memory.actions) & set(event.actions))
            if nearby and related:
                event.start_time = min(event.start_time, memory.ref.start_time)
                event.end_time = max(event.end_time, memory.ref.end_time)
                event.entities = sorted(set(event.entities) | set(memory.entities))
                event.actions = sorted(set(event.actions) | set(memory.actions))
                event.evidence_ids = sorted(set(event.evidence_ids) | {memory.memory_id})
                event.summary = memory.observation
                event.confidence = max(event.confidence, memory.confidence)
                event.updated_at = time.time()
                return event
        event = EventMemory(
            event_id=self._new_event_id(),
            video_id=self.video_id,
            start_time=memory.ref.start_time,
            end_time=memory.ref.end_time,
            summary=memory.observation,
            entities=list(memory.entities),
            actions=list(memory.actions),
            evidence_ids=[memory.memory_id],
            confidence=memory.confidence,
        )
        self.events[event.event_id] = event
        self._evict_events_if_needed()
        return event

    def get_observation(self, memory_id: str) -> ObservationMemory | None:
        memory = self.observations.get(memory_id)
        return memory if memory is not None and memory.status != EvidenceStatus.EVICTED else None

    def mark_verified(self, memory_id: str) -> ObservationMemory:
        memory = self.observations[memory_id]
        memory.status = EvidenceStatus.VERIFIED
        memory.verified_at = time.time()
        memory.updated_at = memory.verified_at
        for event in self.events.values():
            if memory_id in event.evidence_ids:
                event.status = EvidenceStatus.VERIFIED
                event.updated_at = memory.updated_at
        return memory

    def quarantine(self, memory_id: str, conflicts_with: Iterable[str] = ()) -> ObservationMemory:
        memory = self.observations[memory_id]
        memory.status = EvidenceStatus.QUARANTINED
        memory.conflicts_with = sorted(set(memory.conflicts_with) | {item for item in conflicts_with if item in self.observations})
        memory.updated_at = time.time()
        return memory

    def search(self, query: str, top_k: int = 4, time_range: tuple[float, float] | None = None) -> list[MemoryHit]:
        """混合词元、实体、动作、时间、置信度和状态的可解释检索。"""

        query_tokens = _tokens(query)
        hits: list[MemoryHit] = []
        for memory in self.observations.values():
            if memory.status in {EvidenceStatus.EVICTED, EvidenceStatus.STALE, EvidenceStatus.QUARANTINED}:
                continue
            if time_range and (memory.ref.end_time <= time_range[0] or memory.ref.start_time >= time_range[1]):
                continue
            memory_tokens = _tokens(memory.observation) | set(memory.entities) | set(memory.actions)
            lexical = len(query_tokens & memory_tokens)
            if lexical == 0:
                continue
            score = float(lexical) + 0.25 * memory.confidence + 0.02 * memory.access_count
            if memory.status == EvidenceStatus.VERIFIED:
                score += 0.6
            hits.append(MemoryHit(memory.memory_id, "observation", score, memory.observation, memory.ref.start_time, memory.ref.end_time, memory.status))
        for event in self.events.values():
            if event.status in {EvidenceStatus.EVICTED, EvidenceStatus.STALE, EvidenceStatus.QUARANTINED}:
                continue
            if time_range and (event.end_time <= time_range[0] or event.start_time >= time_range[1]):
                continue
            event_tokens = _tokens(event.summary) | set(event.entities) | set(event.actions)
            lexical = len(query_tokens & event_tokens)
            if lexical:
                score = float(lexical) + 0.2 * event.confidence + (0.5 if event.status == EvidenceStatus.VERIFIED else 0.0)
                hits.append(MemoryHit(event.event_id, "event", score, event.summary, event.start_time, event.end_time, event.status))
        hits.sort(key=lambda hit: (-hit.score, -hit.end_time, hit.memory_id))
        selected = hits[: max(0, int(top_k))]
        for hit in selected:
            if hit.kind == "observation":
                self.observations[hit.memory_id].access_count += 1
        return selected

    def observation_ids_for_hit(self, hit: MemoryHit) -> list[str]:
        if hit.kind == "observation":
            return [hit.memory_id]
        event = self.events.get(hit.memory_id)
        return list(event.evidence_ids) if event is not None else []

    def evict_if_needed(self) -> list[str]:
        """只淘汰高层摘要；L0 原始视频指针仍可由源视频重新获得。"""

        active = [item for item in self.observations.values() if item.status != EvidenceStatus.EVICTED]
        evicted: list[str] = []
        while len(active) > self.max_observations:
            victim = min(
                active,
                key=lambda item: (
                    item.status == EvidenceStatus.VERIFIED,
                    item.access_count,
                    item.confidence,
                    item.ref.end_time,
                ),
            )
            victim.status = EvidenceStatus.EVICTED
            victim.updated_at = time.time()
            evicted.append(victim.memory_id)
            active.remove(victim)
        return evicted

    def _evict_events_if_needed(self) -> None:
        active = [item for item in self.events.values() if item.status != EvidenceStatus.EVICTED]
        while len(active) > self.max_events:
            victim = min(active, key=lambda item: (item.status == EvidenceStatus.VERIFIED, item.confidence, item.end_time))
            victim.status = EvidenceStatus.EVICTED
            active.remove(victim)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "version": 1,
                    "video_id": self.video_id,
                    "max_observations": self.max_observations,
                    "max_events": self.max_events,
                    "next_observation_id": self._next_observation_id,
                    "next_event_id": self._next_event_id,
                    "observations": [item.to_dict() for item in self.observations.values()],
                    "events": [item.to_dict() for item in self.events.values()],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VideoEvidenceIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(payload["video_id"], payload.get("max_observations", 256), payload.get("max_events", 96))
        index._next_observation_id = int(payload.get("next_observation_id", 0))
        index._next_event_id = int(payload.get("next_event_id", 0))
        index.observations = {item["memory_id"]: ObservationMemory.from_dict(item) for item in payload.get("observations", [])}
        index.events = {item["event_id"]: EventMemory.from_dict(item) for item in payload.get("events", [])}
        return index


def _parse_observation(raw: str) -> tuple[str, list[str], list[str], float]:
    """解析感知端 JSON；模型不服从时保留原始描述，避免静默丢失证据。"""

    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            observation = str(payload.get("observation", "")).strip()
            entities = payload.get("entities", [])
            actions = payload.get("actions", [])
            confidence = float(payload.get("confidence", 0.5))
            if observation and isinstance(entities, list) and isinstance(actions, list):
                return observation, [str(item) for item in entities], [str(item) for item in actions], confidence
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return raw.strip(), [], [], 0.5


class CoarseEvidenceIndexer:
    """感知角色：建立问题无关、可回溯的粗粒度全视频索引。"""

    def __init__(self, perception: FrameModel, coarse_stride: int = 12, max_frames_per_call: int = 4) -> None:
        self.perception = perception
        self.coarse_stride = max(1, int(coarse_stride))
        self.max_frames_per_call = max(1, int(max_frames_per_call))

    def build(self, repository: ChunkRepository, index: VideoEvidenceIndex) -> int:
        """扫描固定锚点；更细观察由推理阶段的 ``inspect_segment`` 按需触发。"""

        if repository.video_id != index.video_id:
            raise ValueError("仓库与索引的视频命名空间不一致")
        calls = 0
        chunks = sorted(repository._chunks.values(), key=lambda chunk: int(chunk.chunk_index))
        for offset in range(0, len(chunks), self.coarse_stride):
            group = chunks[offset : offset + self.coarse_stride]
            if not group:
                continue
            indices = [int(chunk.chunk_index) for chunk in group]
            frames = repository.frames_for_indices(indices, self.max_frames_per_call)
            if not frames:
                continue
            raw = self.perception.generate_from_frames(
                frames,
                "You are a long-video perception model. Describe only directly visible objects, "
                "actions, and states in the provided time segment. Do not answer any question. "
                "Return JSON only: {\"observation\":\"...\",\"entities\":[\"...\"],"
                "\"actions\":[\"...\"],\"confidence\":0.0}.",
            )
            calls += 1
            observation, entities, actions, confidence = _parse_observation(raw)
            if observation:
                index.add_observation(
                    RawEvidenceRef(repository.video_id, indices, float(group[0].start_time), float(group[-1].end_time), repository.source_uri),
                    observation,
                    entities,
                    actions,
                    confidence,
                )
        return calls


_ALLOWED_ACTIONS = {"search_memory", "inspect_segment", "verify_evidence", "compare_segments", "answer"}


def parse_dual_role_action(raw: str) -> ToolCall | None:
    """只接受白名单 JSON 动作，模型文本不能直接操作文件或系统。"""

    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    action = str(payload.get("action", "")).strip()
    memory_ids = payload.get("memory_ids", [])
    if action not in _ALLOWED_ACTIONS or not isinstance(memory_ids, list) or not all(isinstance(item, str) for item in memory_ids):
        return None
    def optional_number(name: str) -> float | None:
        value = payload.get(name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return ToolCall(
        action=action,
        memory_ids=memory_ids[:4],
        query=str(payload.get("query", ""))[:400],
        start_time=optional_number("start_time"),
        end_time=optional_number("end_time"),
        target=str(payload.get("target", ""))[:400],
        answer=str(payload.get("answer", ""))[:1000],
        raw=raw,
    )


class DualRoleVeriStreamAgent:
    """推理角色：检索索引，并按需回溯多模态感知服务。"""

    def __init__(
        self,
        perception: FrameModel,
        reasoning: TextModel,
        repository: ChunkRepository,
        index: VideoEvidenceIndex,
        max_actions: int = 4,
        max_working_memories: int = 4,
        final_recent_frames: int = 4,
    ) -> None:
        if repository.video_id != index.video_id:
            raise ValueError("仓库与索引的视频命名空间不一致")
        self.perception = perception
        self.reasoning = reasoning
        self.repository = repository
        self.index = index
        self.max_actions = max(1, int(max_actions))
        self.max_working_memories = max(1, int(max_working_memories))
        self.final_recent_frames = max(0, int(final_recent_frames))

    def _working_context(self, ids: Sequence[str]) -> str:
        rows: list[str] = []
        for memory_id in ids:
            memory = self.index.get_observation(memory_id)
            if memory is not None:
                rows.append(
                    f"- id={memory.memory_id}; time={memory.ref.start_time:.1f}-{memory.ref.end_time:.1f}; "
                    f"status={memory.status.value}; observation={memory.observation}"
                )
        return "\n".join(rows) or "(no candidate evidence)"

    def _controller_prompt(self, question: str, working_ids: Sequence[str], step: int, tool_result: str = "") -> str:
        return (
            "You are the reasoning controller for long-video question answering. Candidate summaries "
            "are not final facts; verify any unverified summary before relying on it. "
            "Do not verify a memory whose status is already VERIFIED or QUARANTINED; choose another "
            "candidate, inspect a new time interval, search again, compare segments, or answer. "
            "Return exactly one JSON object, with no Markdown. Allowed actions: "
            "search_memory, inspect_segment, verify_evidence, compare_segments, answer.\n"
            "Format: {\"action\":\"...\",\"memory_ids\":[\"...\"],\"query\":\"...\","
            "\"start_time\":null,\"end_time\":null,\"target\":\"...\",\"answer\":\"...\"}\n"
            f"Step {step}. Question: {question}\nCandidate evidence:\n{self._working_context(working_ids)}\n"
            f"Previous tool result: {tool_result or '(none)'}"
        )

    def _add_hits(self, working_ids: list[str], hits: Sequence[MemoryHit]) -> None:
        for hit in hits:
            for memory_id in self.index.observation_ids_for_hit(hit):
                if memory_id not in working_ids:
                    working_ids.append(memory_id)
        del working_ids[self.max_working_memories :]

    def _inspect(self, call: ToolCall, trace: DualRoleTrace) -> tuple[str, list[str]]:
        if call.start_time is None or call.end_time is None or call.end_time <= call.start_time:
            return "Invalid inspect_segment parameters.", []
        indices, frames = self.repository.select_interval(call.start_time, call.end_time)
        if not frames:
            return "No frames are available for the requested time interval.", []
        raw = self.perception.generate_from_frames(
            frames,
            "Re-inspect the provided local video frames and state only directly visible facts. "
            "Return JSON only: {\"observation\":\"...\",\"entities\":[\"...\"],"
            "\"actions\":[\"...\"],\"confidence\":0.0}. "
            f"Focus target: {call.target or 'changes relevant to the question'}",
        )
        trace.perception_calls += 1
        observation, entities, actions, confidence = _parse_observation(raw)
        if not observation:
            return "The local re-inspection produced no valid observation.", []
        memory = self.index.add_observation(
            RawEvidenceRef(self.index.video_id, indices, call.start_time, call.end_time, self.repository.source_uri),
            observation,
            entities,
            actions,
            confidence,
        )
        return f"Local re-inspection: {memory.observation}", [memory.memory_id]

    def _verify(self, call: ToolCall, trace: DualRoleTrace) -> tuple[str, list[str]]:
        verified: list[str] = []
        details: list[str] = []
        for memory_id in call.memory_ids[: self.max_working_memories]:
            memory = self.index.get_observation(memory_id)
            if memory is None:
                continue
            if memory.status == EvidenceStatus.VERIFIED:
                details.append(f"{memory_id}: already verified")
                verified.append(memory_id)
                continue
            if memory.status == EvidenceStatus.QUARANTINED:
                details.append(f"{memory_id}: already quarantined")
                continue
            frames = self.repository.frames_for_indices(memory.ref.chunk_indices)
            if not frames:
                self.index.quarantine(memory_id)
                details.append(f"{memory_id}: raw frames unavailable; quarantined")
                continue
            raw = self.perception.generate_from_frames(
                frames,
                "Check whether the following historical observation is directly supported by the "
                "provided video frames. Output SUPPORTED, UNSUPPORTED, or CONFLICT on the first line; "
                "give a short directly visible fact on the second line.\n"
                f"Historical observation: {memory.observation}\nFocus target: {call.target or 'facts relevant to the question'}",
            ).strip()
            trace.perception_calls += 1
            label = raw.upper().splitlines()[0] if raw else ""
            if label.startswith("SUPPORTED"):
                self.index.mark_verified(memory_id)
                verified.append(memory_id)
                details.append(f"{memory_id}: verified")
            else:
                self.index.quarantine(memory_id)
                details.append(f"{memory_id}: unsupported or conflicting; quarantined")
        return "; ".join(details) or "No evidence is available for verification.", verified

    def _compare(self, call: ToolCall, trace: DualRoleTrace) -> str:
        selected = [self.index.get_observation(item) for item in call.memory_ids[:2]]
        selected = [item for item in selected if item is not None]
        if len(selected) < 2:
            return "compare_segments requires at least two valid evidence items."
        frames: list[Image.Image] = []
        for memory in selected:
            frames.extend(self.repository.frames_for_indices(memory.ref.chunk_indices, max_frames=2))
        if not frames:
            return "The raw frames required for comparison are unavailable."
        raw = self.perception.generate_from_frames(
            frames[:4],
            "Compare the two groups of video frames in temporal order and state whether the "
            "visible state changed. "
            f"Comparison target: {call.target or 'the state relevant to the question'}",
        ).strip()
        trace.perception_calls += 1
        return f"Local comparison: {raw}"

    def answer(self, question: str) -> tuple[str, DualRoleTrace]:
        """执行问题级 L3 工作记忆与受限工具循环。"""

        trace = DualRoleTrace()
        working_ids: list[str] = []
        initial_hits = self.index.search(question, self.max_working_memories)
        self._add_hits(working_ids, initial_hits)
        trace.retrieved_ids.extend(item.memory_id for item in initial_hits)
        tool_result = "Initial memory search completed."
        requested_answer = ""

        for step in range(1, self.max_actions + 1):
            raw = self.reasoning.generate_from_text(self._controller_prompt(question, working_ids, step, tool_result))
            trace.reasoning_calls += 1
            call = parse_dual_role_action(raw)
            if call is None:
                trace.controller_failures += 1
                call = ToolCall(action="search_memory", query=question, raw=raw)
            trace.calls.append(call)
            if call.action == "search_memory":
                hits = self.index.search(call.query or question, self.max_working_memories)
                self._add_hits(working_ids, hits)
                trace.retrieved_ids.extend(item.memory_id for item in hits)
                tool_result = f"Retrieved {len(hits)} candidate evidence items."
            elif call.action == "inspect_segment":
                tool_result, new_ids = self._inspect(call, trace)
                for memory_id in new_ids:
                    if memory_id not in working_ids:
                        working_ids.append(memory_id)
                del working_ids[self.max_working_memories :]
            elif call.action == "verify_evidence":
                tool_result, verified = self._verify(call, trace)
                trace.verified_ids.extend(verified)
            elif call.action == "compare_segments":
                tool_result = self._compare(call, trace)
            else:
                requested_answer = call.answer
                break

        verified_ids = [
            memory_id
            for memory_id in working_ids
            if (memory := self.index.get_observation(memory_id)) is not None and memory.status == EvidenceStatus.VERIFIED
        ]
        if not verified_ids and working_ids:
            # 控制器提前结束时，仍核验最高相关候选，阻止候选摘要直接进入结论。
            _, verified = self._verify(ToolCall(action="verify_evidence", memory_ids=[working_ids[0]], target=question), trace)
            trace.verified_ids.extend(verified)
            verified_ids.extend(verified)
        final_context = self._working_context(verified_ids)
        recent_frames = self.repository.recent_frames(self.final_recent_frames)
        final_prompt = (
            "You are the final multimodal reasoning model for long-video question answering. "
            "The attached recent frames are the current causal visual context. Use them to judge "
            "the current state, fine-grained actions, spatial relations, and visible text. "
            "The verified evidence below describes earlier video content and must be used for "
            "long-term history only. Do not treat an unverified memory as fact. "
            "If this is a multiple-choice question, output only the option letter (A, B, C, or D); "
            "otherwise provide a concise answer.\n"
            f"Question: {question}\nVerified historical evidence:\n{final_context or '(none)'}\n"
            "Answer using the recent frames together with the verified historical evidence."
        )
        trace.final_vision_frames = len(recent_frames)
        if recent_frames and hasattr(self.reasoning, "generate_from_frames"):
            final_answer = self.reasoning.generate_from_frames(recent_frames, final_prompt).strip()
        else:
            final_answer = self.reasoning.generate_from_text(final_prompt).strip()
        trace.reasoning_calls += 1
        return final_answer or requested_answer, trace


class OpenAICompatibleVLMClient:
    """vLLM OpenAI 兼容服务的轻量客户端。

    服务端只负责生成；工具执行、记忆更新和权限控制全部留在本地编排器中。
    """

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout_seconds: float = 180.0, max_tokens: int = 256) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)

    def _request(self, content: list[dict[str, Any]]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM 服务不可用：{self.base_url}") from exc
        try:
            return str(parsed["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"vLLM 返回格式异常：{parsed}") from exc

    def generate_from_text(self, prompt: str) -> str:
        return self._request([{"type": "text", "text": prompt}])

    def generate_from_frames(self, frames: list[Image.Image], prompt: str) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in frames:
            buffer = io.BytesIO()
            frame.convert("RGB").save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        return self._request(content)
