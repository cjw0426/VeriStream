"""VeriStream 的训练免调优证据记忆与流式工具编排。

模块刻意不依赖现成视频 Agent 框架。记忆层只保存可回溯的证据指针，
推理层必须重新观察候选证据后才能把它送入最终答案上下文。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from PIL import Image


class EvidenceStatus(str, Enum):
    """证据卡片在生命周期中的状态。"""

    CANDIDATE = "candidate"
    VERIFIED = "verified"
    STALE = "stale"
    QUARANTINED = "quarantined"
    EVICTED = "evicted"


@dataclass
class EvidenceCard:
    """带时间和原始帧指针的最小证据单元。"""

    card_id: str
    video_id: str
    start_time: float
    end_time: float
    chunk_indices: list[int]
    observation: str
    labels: list[str]
    write_confidence: float = 0.5
    status: EvidenceStatus = EvidenceStatus.CANDIDATE
    conflicts_with: list[str] = field(default_factory=list)
    access_count: int = 0
    verified_at: float | None = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceCard":
        copied = dict(payload)
        copied["status"] = EvidenceStatus(copied.get("status", EvidenceStatus.CANDIDATE.value))
        return cls(**copied)


@dataclass
class ToolAction:
    """控制器动作及其可复现实验轨迹。"""

    action: str
    card_ids: list[str] = field(default_factory=list)
    target: str = ""
    reason: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolTrace:
    """一次回答使用的工具、证据和预算统计。"""

    actions: list[ToolAction] = field(default_factory=list)
    retrieved_card_ids: list[str] = field(default_factory=list)
    verified_card_ids: list[str] = field(default_factory=list)
    final_card_ids: list[str] = field(default_factory=list)
    qwen_calls: int = 0
    qwen_frames: int = 0
    controller_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [item.to_dict() for item in self.actions],
            "retrieved_card_ids": self.retrieved_card_ids,
            "verified_card_ids": self.verified_card_ids,
            "final_card_ids": self.final_card_ids,
            "qwen_calls": self.qwen_calls,
            "qwen_frames": self.qwen_frames,
            "controller_failures": self.controller_failures,
        }


class QAModel(Protocol):
    """VeriStream 所需的冻结多模态模型最小接口。"""

    def generate_from_frames(self, frames: list[Image.Image], question: str) -> str: ...

    def generate_from_text(self, prompt: str) -> str: ...


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()))


def _extract_labels(text: str, limit: int = 8) -> list[str]:
    """用保守的词元标签支持无训练的重复检测与查询匹配。"""

    labels = [item for item in _tokens(text) if len(item) >= 2]
    return sorted(labels)[:limit]


class EvidenceMemoryStore:
    """按视频命名空间隔离的持久证据库。"""

    def __init__(self, video_id: str, max_cards: int = 48) -> None:
        if not video_id:
            raise ValueError("video_id 不能为空")
        self.video_id = video_id
        self.max_cards = max(1, int(max_cards))
        self.cards: dict[str, EvidenceCard] = {}
        self._next_id = 0

    def _new_id(self) -> str:
        self._next_id += 1
        return f"{self.video_id}:e{self._next_id:05d}"

    def _find_duplicate(self, observation: str, chunk_indices: Sequence[int]) -> EvidenceCard | None:
        query_tokens = _tokens(observation)
        query_chunks = set(chunk_indices)
        for card in self.cards.values():
            if card.status == EvidenceStatus.EVICTED:
                continue
            overlap = bool(query_chunks & set(card.chunk_indices))
            card_tokens = _tokens(card.observation)
            union = query_tokens | card_tokens
            similarity = len(query_tokens & card_tokens) / len(union) if union else 0.0
            if overlap and similarity >= 0.45:
                return card
        return None

    def write_candidate(
        self,
        start_time: float,
        end_time: float,
        chunk_indices: Sequence[int],
        observation: str,
        confidence: float = 0.5,
        labels: Sequence[str] | None = None,
    ) -> EvidenceCard:
        """写入候选证据；同一时段的重复观察只更新已有卡片。"""

        cleaned = " ".join(str(observation).split()).strip()
        if not cleaned:
            raise ValueError("候选证据不能为空")
        indices = sorted({int(index) for index in chunk_indices})
        duplicate = self._find_duplicate(cleaned, indices)
        if duplicate is not None:
            duplicate.observation = cleaned
            duplicate.labels = sorted(set(duplicate.labels) | set(labels or _extract_labels(cleaned)))
            duplicate.chunk_indices = sorted(set(duplicate.chunk_indices) | set(indices))
            duplicate.start_time = min(duplicate.start_time, float(start_time))
            duplicate.end_time = max(duplicate.end_time, float(end_time))
            duplicate.write_confidence = max(duplicate.write_confidence, float(confidence))
            duplicate.updated_at = time.time()
            return duplicate

        card = EvidenceCard(
            card_id=self._new_id(),
            video_id=self.video_id,
            start_time=float(start_time),
            end_time=float(end_time),
            chunk_indices=indices,
            observation=cleaned,
            labels=list(labels or _extract_labels(cleaned)),
            write_confidence=max(0.0, min(1.0, float(confidence))),
        )
        self.cards[card.card_id] = card
        self.evict_if_needed()
        return card

    def retrieve(self, query: str, limit: int = 4, include_quarantined: bool = False) -> list[EvidenceCard]:
        """按词元重合、置信度和新近性排序，且不跨视频检索。"""

        query_tokens = _tokens(query)
        rows: list[tuple[float, EvidenceCard]] = []
        for card in self.cards.values():
            if card.status in {EvidenceStatus.EVICTED, EvidenceStatus.STALE}:
                continue
            if card.status == EvidenceStatus.QUARANTINED and not include_quarantined:
                continue
            overlap = len(query_tokens & (_tokens(card.observation) | set(card.labels)))
            status_bonus = 0.4 if card.status == EvidenceStatus.VERIFIED else 0.0
            score = overlap + status_bonus + 0.1 * card.write_confidence + 0.01 * card.access_count
            if score > 0:
                rows.append((score, card))
        rows.sort(key=lambda item: (-item[0], -item[1].end_time, item[1].card_id))
        selected = [card for _, card in rows[: max(0, int(limit))]]
        for card in selected:
            card.access_count += 1
        return selected

    def mark_verified(self, card_id: str) -> EvidenceCard:
        card = self.cards[card_id]
        card.status = EvidenceStatus.VERIFIED
        card.verified_at = time.time()
        card.updated_at = card.verified_at
        return card

    def quarantine(self, card_id: str, conflict_ids: Iterable[str] = ()) -> EvidenceCard:
        """隔离未被重观察支持的记忆，阻止其污染后续回答。"""

        card = self.cards[card_id]
        card.status = EvidenceStatus.QUARANTINED
        card.conflicts_with = sorted(set(card.conflicts_with) | {item for item in conflict_ids if item in self.cards})
        card.updated_at = time.time()
        return card

    def evict_if_needed(self) -> list[str]:
        """优先删除未验证、低访问且更早的候选，不删除原始视频指针。"""

        active = [card for card in self.cards.values() if card.status != EvidenceStatus.EVICTED]
        evicted: list[str] = []
        while len(active) > self.max_cards:
            victim = min(
                active,
                key=lambda card: (
                    card.status == EvidenceStatus.VERIFIED,
                    card.access_count,
                    card.write_confidence,
                    card.end_time,
                ),
            )
            victim.status = EvidenceStatus.EVICTED
            victim.updated_at = time.time()
            evicted.append(victim.card_id)
            active.remove(victim)
        return evicted

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "video_id": self.video_id,
                    "max_cards": self.max_cards,
                    "next_id": self._next_id,
                    "cards": [card.to_dict() for card in self.cards.values()],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceMemoryStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(video_id=payload["video_id"], max_cards=payload.get("max_cards", 48))
        store._next_id = int(payload.get("next_id", 0))
        store.cards = {card["card_id"]: EvidenceCard.from_dict(card) for card in payload.get("cards", [])}
        return store


def parse_tool_action(raw: str) -> ToolAction | None:
    """只接受受限 JSON，避免模型自由文本直接驱动文件或视频操作。"""

    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    action = str(payload.get("action", "")).strip()
    allowed = {"retrieve", "inspect_recent", "verify_and_zoom", "compare", "write_or_update", "answer"}
    if action not in allowed:
        return None
    card_ids = payload.get("card_ids", [])
    if not isinstance(card_ids, list) or not all(isinstance(item, str) for item in card_ids):
        return None
    return ToolAction(
        action=action,
        card_ids=card_ids[:4],
        target=str(payload.get("target", ""))[:240],
        reason=str(payload.get("reason", ""))[:240],
        raw=raw,
    )


class VeriStreamAgent:
    """以验证门控为中心的在线长视频理解 Agent。"""

    def __init__(
        self,
        qa: QAModel,
        video_id: str,
        recent_frames: int = 4,
        max_cards: int = 48,
        anchor_interval: int = 12,
        novelty_threshold: float = 0.12,
        max_actions: int = 6,
        max_final_cards: int = 4,
        novelty_fn: Callable[[Image.Image, Image.Image], float] | None = None,
    ) -> None:
        self.qa = qa
        self.store = EvidenceMemoryStore(video_id=video_id, max_cards=max_cards)
        self.recent_frames = max(1, int(recent_frames))
        self.anchor_interval = max(1, int(anchor_interval))
        self.novelty_threshold = float(novelty_threshold)
        self.max_actions = max(1, int(max_actions))
        self.max_final_cards = max(1, int(max_final_cards))
        self.novelty_fn = novelty_fn or self._default_novelty
        self._chunks: list[Any] = []
        self._last_frame: Image.Image | None = None

    @staticmethod
    def _default_novelty(previous: Image.Image, current: Image.Image) -> float:
        """无额外依赖的像素变化回退策略；正式实验可注入 CLIP 距离。"""

        prev = previous.resize((32, 32)).convert("L")
        curr = current.resize((32, 32)).convert("L")
        previous_values = list(prev.getdata())
        current_values = list(curr.getdata())
        return sum(abs(a - b) for a, b in zip(previous_values, current_values)) / (255.0 * len(previous_values))

    def ingest(self, chunks: Sequence[Any], trace: ToolTrace | None = None) -> None:
        """按时间顺序接收新 chunk，绝不读取尚未到达的未来 chunk。"""

        for chunk in chunks:
            self._chunks.append(chunk)
            if not chunk.frames:
                continue
            frame = chunk.frames[-1]
            is_anchor = int(chunk.chunk_index) % self.anchor_interval == 0
            novelty = 1.0 if self._last_frame is None else self.novelty_fn(self._last_frame, frame)
            self._last_frame = frame
            if not is_anchor and novelty < self.novelty_threshold:
                continue
            prompt = (
                "Describe only directly visible actions, object states, or positions in this historical "
                "video segment. Do not answer a question, speculate, or output an option letter."
            )
            observation = self.qa.generate_from_frames(list(chunk.frames), prompt)
            if trace is not None:
                trace.qwen_calls += 1
                trace.qwen_frames += len(chunk.frames)
                trace.actions.append(ToolAction(action="write_or_update", target=f"chunk:{chunk.chunk_index}"))
            if observation and observation.strip():
                self.store.write_candidate(
                    start_time=float(chunk.start_time),
                    end_time=float(chunk.end_time),
                    chunk_indices=[int(chunk.chunk_index)],
                    observation=observation,
                    confidence=min(1.0, 0.5 + novelty / 2.0),
                )

    def _recent_frames(self) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for chunk in reversed(self._chunks):
            frames[0:0] = list(chunk.frames)
            if len(frames) >= self.recent_frames:
                return frames[-self.recent_frames :]
        return frames

    def _controller_prompt(self, question: str, candidates: Sequence[EvidenceCard], step: int) -> str:
        memory = "\n".join(
            f"- id={card.card_id}; time={card.start_time:.1f}-{card.end_time:.1f}; state={card.status.value}; obs={card.observation}"
            for card in candidates
        ) or "(no candidate historical evidence)"
        return (
            "You are a video-evidence controller. Given the question and candidate historical evidence, "
            "choose the minimum necessary next action. Unverified memory cannot be used as a final fact. "
            "Return exactly one JSON object with no Markdown.\n"
            "Allowed actions: retrieve, inspect_recent, verify_and_zoom, compare, answer.\n"
            "Format: {\"action\":\"...\",\"card_ids\":[\"...\"],\"target\":\"...\",\"reason\":\"...\"}\n"
            f"Step {step}; question: {question}\nCandidate evidence:\n{memory}"
        )

    def _frames_for_card(self, card: EvidenceCard) -> list[Image.Image]:
        wanted = set(card.chunk_indices)
        selected: list[Image.Image] = []
        for chunk in self._chunks:
            if int(chunk.chunk_index) in wanted:
                selected.extend(chunk.frames)
        return selected

    def _verify(self, card: EvidenceCard, target: str, trace: ToolTrace) -> bool:
        """用原始局部帧重新观察候选条目，而非相信历史文本。"""

        frames = self._frames_for_card(card)
        if not frames:
            self.store.quarantine(card.card_id)
            return False
        prompt = (
            "Check whether the following historical observation is directly supported by the provided "
            "video frames. Output SUPPORTED, UNSUPPORTED, or CONFLICT on the first line; give a short "
            "directly visible fact on the second line.\n"
            f"Observation to verify: {card.observation}\nFocus target: {target or 'facts relevant to the question'}"
        )
        response = self.qa.generate_from_frames(frames, prompt).strip()
        trace.qwen_calls += 1
        trace.qwen_frames += len(frames)
        trace.actions.append(ToolAction(action="verify_and_zoom", card_ids=[card.card_id], target=target, raw=response))
        label = response.upper().splitlines()[0] if response else ""
        if label.startswith("SUPPORTED"):
            self.store.mark_verified(card.card_id)
            return True
        self.store.quarantine(card.card_id)
        return False

    @staticmethod
    def _evidence_text(cards: Sequence[EvidenceCard]) -> str:
        return "\n".join(
            f"- [{card.start_time:.1f}-{card.end_time:.1f}s] {card.observation}"
            for card in cards
        )

    def answer(self, question: str) -> tuple[str, ToolTrace]:
        """执行有预算的模型驱动工具循环并给出最终答案。"""

        trace = ToolTrace()
        candidates: list[EvidenceCard] = []
        verified: dict[str, EvidenceCard] = {}
        for step in range(1, self.max_actions + 1):
            raw = self.qa.generate_from_text(self._controller_prompt(question, candidates, step))
            trace.qwen_calls += 1
            action = parse_tool_action(raw)
            if action is None:
                trace.controller_failures += 1
                action = ToolAction(action="retrieve" if not candidates else "answer", reason="控制器输出无效，使用安全回退", raw=raw)
            trace.actions.append(action)
            if action.action == "retrieve":
                candidates = self.store.retrieve(question, limit=self.max_final_cards)
                trace.retrieved_card_ids = [card.card_id for card in candidates]
                continue
            if action.action == "verify_and_zoom":
                requested = action.card_ids or [card.card_id for card in candidates]
                for card_id in requested[: self.max_final_cards]:
                    card = self.store.cards.get(card_id)
                    if card is not None and card.status != EvidenceStatus.QUARANTINED and self._verify(card, action.target, trace):
                        verified[card.card_id] = card
                continue
            if action.action == "compare":
                # 比较动作复用局部验证，避免引入未受控的额外视觉上下文。
                for card_id in action.card_ids[:2]:
                    card = self.store.cards.get(card_id)
                    if card is not None and self._verify(card, action.target, trace):
                        verified[card.card_id] = card
                continue
            if action.action == "answer":
                break
            # inspect_recent 与 write_or_update 不需要在回答阶段额外写入历史记忆。

        if not verified and candidates:
            # 安全回退仍进行一次验证，避免控制器过早结束导致候选文本泄漏。
            first = candidates[0]
            if self._verify(first, question, trace):
                verified[first.card_id] = first
        final_cards = list(verified.values())[: self.max_final_cards]
        final_frames = self._recent_frames()
        for card in final_cards:
            final_frames.extend(self._frames_for_card(card)[:1])
        final_frames = final_frames[-(self.recent_frames + self.max_final_cards) :]
        prompt = (
            "Answer the question using the recent video frames and verified historical evidence. "
            "Historical evidence refers only to the past; do not confuse past actions with the current state.\n"
            f"Verified historical evidence:\n{self._evidence_text(final_cards) or '(none)'}\n\nQuestion: {question}"
        )
        response = self.qa.generate_from_frames(final_frames, prompt)
        trace.qwen_calls += 1
        trace.qwen_frames += len(final_frames)
        trace.final_card_ids = [card.card_id for card in final_cards]
        trace.verified_card_ids = list(verified)
        return response, trace
