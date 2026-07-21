from __future__ import annotations

import copy
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from lib.clip_topk_selector import CLIPTopKFrameSelector
from lib.recent_window_eval import (
    RecentWindowQAModel as _BaseRecentWindowQAModel,
    RecentWindowResult,
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
    flatten_gathered_results,
    print_ovo_results,
)


class RecentWindowQAModel(_BaseRecentWindowQAModel):
    """Qwen3 release wrapper aligned with the per-frame vision-token builder."""

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self._last_ttft_seconds = 0.0
        self._last_num_vision_tokens = 0
        self._last_num_vision_frames = 0

        proc_kwargs: dict[str, object] = {}
        if os.environ.get("MIN_PIXELS"):
            proc_kwargs["min_pixels"] = int(os.environ["MIN_PIXELS"])
        if os.environ.get("MAX_PIXELS"):
            proc_kwargs["max_pixels"] = int(os.environ["MAX_PIXELS"])
        self.processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)

        model_kwargs: dict[str, object] = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"

        saved_world_size = os.environ.pop("WORLD_SIZE", None)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        finally:
            if saved_world_size is not None:
                os.environ["WORLD_SIZE"] = saved_world_size
        if device != "auto":
            self.model.to(device)
        self.model.eval()

        self._hf_model = self.model
        self._visual = self.model.model.visual
        self._text_model = self.model.model
        self.image_token_id = self.model.config.image_token_id
        self.vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.merge_size = self.model.model.visual.spatial_merge_size

    @torch.inference_mode()
    def encode_vision(self, frames: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep official preprocessing, but expose encoded vision for explicit input building."""
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": "."})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(self.model.device)
        image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
        image_embeds = self._flatten_vision_features(
            self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
        )

        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return image_embeds, image_grid_thw

    @torch.inference_mode()
    def encode_vision_batched(
        self,
        frames_per_chunk: list[list[Image.Image]],
        max_frames_per_batch: int = 8,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not frames_per_chunk:
            return []

        flat_pairs: list[tuple[int, Image.Image]] = []
        for chunk_index, frames in enumerate(frames_per_chunk):
            for frame in frames:
                flat_pairs.append((chunk_index, frame))

        hidden_size = int(getattr(self.model.config, "hidden_size", 4096))
        model_dtype = getattr(self.model, "dtype", torch.bfloat16)
        empty_emb = torch.empty((0, hidden_size), dtype=model_dtype, device="cpu")
        empty_grid = torch.empty((0, 3), dtype=torch.long, device="cpu")
        if not flat_pairs:
            return [(empty_emb, empty_grid) for _ in frames_per_chunk]

        merge_area = max(1, int(self.merge_size)) ** 2
        chunk_embeds: list[list[torch.Tensor]] = [[] for _ in frames_per_chunk]
        chunk_grids: list[list[torch.Tensor]] = [[] for _ in frames_per_chunk]

        batch_size = max(1, int(max_frames_per_batch))
        offset_flat = 0
        while offset_flat < len(flat_pairs):
            pairs = flat_pairs[offset_flat : offset_flat + batch_size]
            content = [{"type": "image", "image": frame} for _, frame in pairs]
            content.append({"type": "text", "text": "."})
            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self.model.device)
            image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
            image_embeds = self._flatten_vision_features(
                self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
            )

            frame_token_counts = [
                max(1, int(row[0].item() * row[1].item() * row[2].item()) // merge_area)
                for row in image_grid_thw
            ]
            expected_tokens = sum(frame_token_counts)
            if expected_tokens != int(image_embeds.shape[0]) or len(frame_token_counts) != len(pairs):
                grouped: dict[int, list[Image.Image]] = {}
                for chunk_index, frame in pairs:
                    grouped.setdefault(chunk_index, []).append(frame)
                for chunk_index, frames in grouped.items():
                    emb, grid = self.encode_vision(frames)
                    chunk_embeds[chunk_index].append(emb.to(dtype=torch.bfloat16, device="cpu"))
                    chunk_grids[chunk_index].append(grid.cpu())
                offset_flat += len(pairs)
                del pixel_values, image_grid_thw, image_embeds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            offset = 0
            for (chunk_index, _), token_count, row in zip(pairs, frame_token_counts, image_grid_thw):
                end = offset + token_count
                chunk_embeds[chunk_index].append(image_embeds[offset:end].to(dtype=torch.bfloat16, device="cpu"))
                chunk_grids[chunk_index].append(row.unsqueeze(0).cpu())
                offset = end
            offset_flat += len(pairs)

            del pixel_values, image_grid_thw, image_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for chunk_index in range(len(frames_per_chunk)):
            if chunk_embeds[chunk_index]:
                outputs.append(
                    (
                        torch.cat(chunk_embeds[chunk_index], dim=0),
                        torch.cat(chunk_grids[chunk_index], dim=0),
                    )
                )
            else:
                outputs.append((empty_emb, empty_grid))
        return outputs

    @torch.inference_mode()
    def generate_with_vision_features(
        self,
        vision_embeds: torch.Tensor,
        vision_grid_thw: torch.Tensor,
        question: str,
    ) -> str:
        device = self.model.device
        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model()

        num_vision_tokens = int(vision_embeds.shape[0])
        self._last_num_vision_tokens = num_vision_tokens
        self._last_num_vision_frames = int(vision_grid_thw.shape[0]) if vision_grid_thw is not None else 0

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        grid_rows = vision_grid_thw.to(device)
        tokens_per_frame = (grid_rows.prod(dim=-1) // (self.merge_size**2)).tolist()
        expected_tokens = sum(int(n) for n in tokens_per_frame)
        if expected_tokens != num_vision_tokens:
            raise ValueError(
                "vision token count mismatch: "
                f"embeds={num_vision_tokens} vs grid={expected_tokens}"
            )

        input_ids_list: list[int] = []
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        for frame_token_count in tokens_per_frame:
            input_ids_list.append(self.vision_start_id)
            input_ids_list.extend([self.image_token_id] * int(frame_token_count))
            input_ids_list.append(self.vision_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend(question_ids)
        input_ids_list.append(self.im_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, vision_embeds)

        position_ids, _ = text_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=grid_rows,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        return self._generate_from_model_inputs(
            prompt_length=len(input_ids[0]),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    @torch.inference_mode()
    def generate_from_frames(self, frames: list[Image.Image], question: str) -> str:
        vision_embeds, vision_grid_thw = self.encode_vision(frames)
        return self.generate_with_vision_features(vision_embeds, vision_grid_thw, question)


@dataclass
class EncodedChunk:
    vision_emb: torch.Tensor
    grid_thw: torch.Tensor
    chunk_index: int
    start_time: float
    end_time: float


@dataclass
class TextMemoryBundle:
    lines: list[str]
    raw_summary: str
    captions: list[str]
    history_chunk_ids: list[int]
    memory_type: str = "action"


@dataclass
class VSTMemoryBundle:
    entries: list[str]
    text: str
    history_chunk_ids: list[int]
    backend: str
    clip_size: int
    max_clips: int
    version: str = "v1"
    raw_entries: list[str] | None = None


def _combine_window_embeddings(
    window: deque[EncodedChunk],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_embeds = torch.cat([item.vision_emb.to(device) for item in window], dim=0)
    combined_grid_thw = torch.cat([item.grid_thw.to(device) for item in window], dim=0)
    return combined_embeds, combined_grid_thw


def query_recent_window(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str]:
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames_only,
        video_start=video_start,
        video_end=video_end,
    )
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    window_size = max(1, int(recent_frames_only))
    recent_chunks = chunks[-window_size:]
    encoded_chunks: list[EncodedChunk] = []
    encoded_outputs = qa.encode_vision_batched([chunk.frames for chunk in recent_chunks], max_frames_per_batch=8)
    for chunk, (vision_emb, grid_thw) in zip(recent_chunks, encoded_outputs):
        if int(vision_emb.shape[0]) == 0 or int(grid_thw.shape[0]) == 0:
            continue
        encoded_chunks.append(
            EncodedChunk(
                vision_emb=vision_emb,
                grid_thw=grid_thw,
                chunk_index=chunk.chunk_index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
        )
    if not encoded_chunks:
        raise ValueError(f"No vision chunks encoded from video: {video_path}")

    encoded_window: deque[EncodedChunk] = deque(encoded_chunks, maxlen=window_size)
    t0 = time.perf_counter()
    combined_embeds, combined_grid_thw = _combine_window_embeddings(encoded_window, qa.model.device)
    answer = qa.generate_with_vision_features(combined_embeds, combined_grid_thw, prompt)
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = qa._last_num_vision_tokens
    num_frames = qa._last_num_vision_frames

    return (
        RecentWindowResult(
            answer=answer,
            final_chunk_ids=[item.chunk_index for item in encoded_window],
            generate_time=generate_time,
            ttft_seconds=ttft_seconds,
            num_vision_tokens=num_vision_tokens,
            num_vision_tokens_before=num_vision_tokens,
            num_vision_tokens_after=num_vision_tokens,
            num_frames=num_frames,
        ),
        decode_backend,
    )


def _flatten_chunk_frames(chunks: list[Any]) -> tuple[list[Image.Image], list[int]]:
    flat_frames: list[Image.Image] = []
    flat_chunk_ids: list[int] = []
    for chunk in chunks:
        for frame in chunk.frames:
            flat_frames.append(frame)
            flat_chunk_ids.append(chunk.chunk_index)
    return flat_frames, flat_chunk_ids


def _flatten_chunk_frames_with_times(chunks: list[Any]) -> tuple[list[Image.Image], list[int], list[tuple[float, float]]]:
    flat_frames: list[Image.Image] = []
    flat_chunk_ids: list[int] = []
    flat_times: list[tuple[float, float]] = []
    for chunk in chunks:
        for frame in chunk.frames:
            flat_frames.append(frame)
            flat_chunk_ids.append(int(chunk.chunk_index))
            flat_times.append((float(chunk.start_time), float(chunk.end_time)))
    return flat_frames, flat_chunk_ids, flat_times


def _select_uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    selected_count = min(max(1, int(count)), int(length))
    if selected_count == 1:
        return [int(length) // 2]
    return [round(i * (int(length) - 1) / (selected_count - 1)) for i in range(selected_count)]


def _decode_flat_frames(
    video_path: str,
    chunk_duration: float,
    fps: float,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[list[Image.Image], list[int], str]:
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=None,
        video_start=video_start,
        video_end=video_end,
    )
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    flat_frames, flat_chunk_ids = _flatten_chunk_frames(chunks)
    if not flat_frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return flat_frames, flat_chunk_ids, decode_backend


def _decode_flat_frames_with_times(
    video_path: str,
    chunk_duration: float,
    fps: float,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[list[Image.Image], list[int], list[tuple[float, float]], str]:
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=None,
        video_start=video_start,
        video_end=video_end,
    )
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    flat_frames, flat_chunk_ids, flat_times = _flatten_chunk_frames_with_times(chunks)
    if not flat_frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return flat_frames, flat_chunk_ids, flat_times, decode_backend


def _build_result_from_selection(
    qa: RecentWindowQAModel,
    selected_frames: list[Image.Image],
    selected_chunk_ids: list[int],
    prompt: str,
    decode_backend: str,
) -> tuple[RecentWindowResult, str]:
    t0 = time.perf_counter()
    answer = qa.generate_from_frames(selected_frames, prompt)
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = qa._last_num_vision_tokens
    num_frames = qa._last_num_vision_frames

    return (
        RecentWindowResult(
            answer=answer,
            final_chunk_ids=selected_chunk_ids,
            generate_time=generate_time,
            ttft_seconds=ttft_seconds,
            num_vision_tokens=num_vision_tokens,
            num_vision_tokens_before=num_vision_tokens,
            num_vision_tokens_after=num_vision_tokens,
            num_frames=num_frames,
        ),
        decode_backend,
    )


def _build_memory_caption_prompt() -> str:
    return (
        "Describe only directly visible historical facts from these frames in one short sentence. "
        "Focus on concrete actions, object locations, object states, and temporal events. "
        "Do not answer any question. Do not mention option letters, choices, yes/no, or counts unless they are visually explicit. "
        "Do not guess, infer hidden facts, or state a final conclusion."
    )


def _build_memory_summary_prompt(question_prompt: str, captions: list[str], max_items: int) -> str:
    question_text = _sanitize_memory_question(question_prompt)
    joined = "\n".join(f"{idx + 1}. {caption}" for idx, caption in enumerate(captions))
    return (
        "You are compressing historical observations into a conservative memory.\n"
        f"Question context:\n{question_text}\n\n"
        f"Historical observations:\n{joined}\n\n"
        f"Rewrite them into at most {max_items} short bullet points.\n"
        "Keep only directly supported facts. Preserve temporal order when possible. "
        "Remove duplicates. Drop any observation that is speculative, answers the question directly, "
        "mentions option letters, or conflicts with another observation. "
        "If observations conflict, keep only the shared conservative fact or omit the item entirely. "
        "Do not mention confidence, confirmation, or meta commentary.\n"
        "Output bullet points only."
    )


def _build_state_memory_caption_prompt() -> str:
    return (
        "Describe only past visual state from these historical frames in one short sentence. "
        "Focus on object identity, object locations, scene continuity, actor identity, and completed progress. "
        "Avoid framing past observations as evidence that a queried action is happening now. "
        "Do not answer any question. Do not mention yes/no, option letters, or final conclusions."
    )


def _build_state_memory_summary_prompt(question_prompt: str, captions: list[str], max_items: int) -> str:
    question_text = _sanitize_memory_question(question_prompt)
    joined = "\n".join(f"{idx + 1}. {caption}" for idx, caption in enumerate(captions))
    return (
        "You are compressing historical observations into temporal state memory.\n"
        f"Question context:\n{question_text}\n\n"
        f"Historical observations:\n{joined}\n\n"
        f"Rewrite them into at most {max_items} short bullet points.\n"
        "Keep only past object state, scene continuity, actor identity, and completed progress. "
        "Do not say or imply that a past action is happening in the current recent clip. "
        "If the question asks whether something is happening now, preserve only earlier state and progress, not a current-action judgment. "
        "Remove duplicates, speculation, option letters, yes/no wording, and final answers. "
        "Output bullet points only."
    )


def _normalize_memory_lines(summary: str, max_items: int) -> list[str]:
    lines: list[str] = []
    for raw in str(summary).splitlines():
        text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw).strip()
        if text:
            lines.append(text)
    if not lines:
        for chunk in re.split(r"(?<=[.!?])\s+", str(summary).strip()):
            text = chunk.strip(" -\n\t")
            if text:
                lines.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if _is_invalid_memory_line(line):
            continue
        key = line.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(line)
        if len(deduped) >= max(1, int(max_items)):
            break
    return deduped


def _sanitize_memory_question(question_prompt: str) -> str:
    lines = [line.strip() for line in str(question_prompt).splitlines() if line.strip()]
    kept: list[str] = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("options:"):
            continue
        if "only give the best option" in lowered:
            continue
        if "answer yes or no only" in lowered:
            continue
        if "only give a number as answer" in lowered:
            continue
        kept.append(line)
    if not kept:
        return str(question_prompt).strip()
    return kept[0]


def _clean_caption_text(caption: str) -> str:
    text = " ".join(str(caption).split())
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
    if _is_invalid_memory_line(text):
        return ""
    return text


def _clean_state_caption_text(caption: str) -> str:
    text = " ".join(str(caption).split())
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
    if _is_invalid_state_memory_line(text):
        return ""
    return text


def _is_invalid_memory_line(text: str) -> bool:
    stripped = str(text).strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if re.fullmatch(r"[A-D][.)]?", stripped):
        return True
    if re.match(r"^[A-D][.)]\s", stripped):
        return True
    banned_phrases = (
        "option ",
        "answer is ",
        "the answer is ",
        "yes",
        "no",
        "confirmed by multiple observations",
        "not visible in any of the provided frames",
        "there is enough information",
    )
    if any(phrase in lowered for phrase in banned_phrases):
        return True
    return False


def _is_invalid_state_memory_line(text: str) -> bool:
    stripped = str(text).strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if re.fullmatch(r"(?:[A-D][.)]?|yes|no)", stripped, flags=re.IGNORECASE):
        return True
    if re.match(r"^[A-D][.)]\s", stripped):
        return True
    banned_phrases = (
        "option ",
        "answer is ",
        "the answer is ",
        "final answer",
        "not visible in any of the provided frames",
        "there is enough information",
    )
    if any(phrase in lowered for phrase in banned_phrases):
        return True
    return False


def _normalize_state_memory_lines(summary: str, max_items: int) -> list[str]:
    lines: list[str] = []
    for raw in str(summary).splitlines():
        text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw).strip()
        if text:
            lines.append(text)
    if not lines:
        for chunk in re.split(r"(?<=[.!?])\s+", str(summary).strip()):
            text = chunk.strip(" -\n\t")
            if text:
                lines.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if _is_invalid_state_memory_line(line):
            continue
        key = line.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(line)
        if len(deduped) >= max(1, int(max_items)):
            break
    return deduped


def _build_prompt_with_memory(base_prompt: str, memory_lines: list[str]) -> str:
    if not memory_lines:
        return base_prompt
    memory_block = "\n".join(f"- {line}" for line in memory_lines)
    return (
        "Use the recent observations as the primary evidence.\n"
        "Use the historical memory only as supporting context.\n"
        "If the recent observations conflict with the memory, trust the recent observations.\n\n"
        f"Historical memory:\n{memory_block}\n\n"
        f"{base_prompt}"
    )


def _build_prompt_with_state_memory(base_prompt: str, memory_lines: list[str]) -> str:
    if not memory_lines:
        return base_prompt
    memory_block = "\n".join(f"- {line}" for line in memory_lines)
    return (
        "Use the current recent video clip as the only evidence for current visual actions and current status.\n"
        "Use the historical state memory only for earlier object state, scene continuity, actor identity, and prior progress.\n"
        "If historical memory describes an action from earlier in the video, do not assume that action is happening now.\n"
        "If the recent clip conflicts with historical memory, trust the recent clip.\n\n"
        f"Historical state memory:\n{memory_block}\n\n"
        f"{base_prompt}"
    )


def _build_vst_memory_update_prompt(previous_memory: str, start_time: float, end_time: float) -> str:
    memory = previous_memory.strip() if previous_memory.strip() else "(empty)"
    return (
        "[System]\n"
        "You are a Streaming Video Analyst.\n\n"
        "Previous Video Memory:\n"
        f"{memory}\n\n"
        "Current segment:\n"
        f"Time={start_time:.1f}-{end_time:.1f}s\n"
        "[VideoClip]\n\n"
        "Streaming Thinking Rules:\n"
        "1. Update Only: observe the current segment and record only new visual clues.\n"
        "2. Do not repeat previous memory.\n"
        "3. Do not answer the final question yet.\n"
        "4. Keep concrete actions, object locations, object states, and temporal changes.\n\n"
        "Updated memory:"
    )


def _build_vst_final_prompt(base_prompt: str, memory_text: str) -> str:
    memory = memory_text.strip() if memory_text.strip() else "(empty)"
    return (
        "[System]\n"
        "You are a Streaming Video Analyst.\n\n"
        "Video Memory:\n"
        f"{memory}\n\n"
        "Based on the provided Video Memory and the current recent video clip, answer the following problem.\n"
        "Use the recent clip as primary evidence for current visual details.\n"
        "Use memory for earlier events and object history.\n\n"
        f"{base_prompt}"
    )


def _sanitize_vst_memory_question_v2(question_prompt: str) -> str:
    lines: list[str] = []
    for raw_line in str(question_prompt).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^[A-D][.)]\s+", line):
            continue
        lowered = line.lower()
        if "answer with the option" in lowered or "choose the correct" in lowered:
            continue
        lines.append(line)
    sanitized = " ".join(lines)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized[:800]


def _build_vst_memory_update_prompt_v2(
    previous_memory: str,
    question_prompt: str,
    start_time: float,
    end_time: float,
) -> str:
    memory = previous_memory.strip() if previous_memory.strip() else "(empty)"
    question = _sanitize_vst_memory_question_v2(question_prompt) or "(not provided)"
    return (
        "[System]\n"
        "You are a Streaming Video Analyst.\n\n"
        "Final question context (do not answer yet):\n"
        f"{question}\n\n"
        "Previous Video Memory:\n"
        f"{memory}\n\n"
        "Current selected historical observations:\n"
        f"Selected frame time range={start_time:.1f}-{end_time:.1f}s\n"
        "[VideoClip]\n\n"
        "Memory Update Rules:\n"
        "1. Write only NEW evidence that may help the final question.\n"
        "2. Use 1-2 short bullet points, maximum 35 words total.\n"
        "3. Do not restate previous memory.\n"
        "4. Do not infer between sparse frames; describe only visible facts.\n"
        "5. Do not mention option letters, answer choices, yes/no, or final answers.\n"
        "6. If nothing useful is visible, output: None\n\n"
        "New memory bullets:"
    )


def _build_vst_memory_compression_prompt_v2(question_prompt: str, memory_entries: list[str], max_items: int = 4) -> str:
    question = _sanitize_vst_memory_question_v2(question_prompt) or "(not provided)"
    entries = "\n".join(f"- {entry}" for entry in memory_entries) if memory_entries else "(empty)"
    return (
        "Compress the historical memory for final video QA.\n\n"
        "Final question context:\n"
        f"{question}\n\n"
        "Historical memory entries:\n"
        f"{entries}\n\n"
        f"Keep at most {max(1, int(max_items))} short bullet points.\n"
        "Keep only concrete visual evidence useful for the final question.\n"
        "Remove repetition, speculation, answer choices, option letters, and final answers.\n"
        "Output bullet points only. If no useful memory remains, output: None"
    )


def _build_vst_final_prompt_v2(base_prompt: str, memory_text: str) -> str:
    memory = memory_text.strip() if memory_text.strip() else "(empty)"
    return (
        "[System]\n"
        "You are a Streaming Video Analyst.\n\n"
        "Compact Video Memory:\n"
        f"{memory}\n\n"
        "Answer the following problem using the current recent video clip as primary evidence.\n"
        "Use Compact Video Memory only for earlier events and object history.\n"
        "If memory is irrelevant or conflicts with the recent clip, ignore the memory.\n\n"
        f"{base_prompt}"
    )


def _clean_vst_memory_update(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip()
    cleaned = re.sub(r"^\s*(?:Updated memory:|Memory update:)\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", cleaned).strip()
    if _is_invalid_memory_line(cleaned):
        return ""
    return cleaned


def _strip_repeated_memory_text(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"\b(\w{3,})(?:\s+\1\b){2,}", r"\1", cleaned, flags=re.IGNORECASE)
    for size in range(4, 21):
        pattern = re.compile(rf"\b([A-Za-z]{{{size}}})(?:\1){{2,}}\b")
        cleaned = pattern.sub(r"\1", cleaned)
    return cleaned


def _is_answer_like_memory_line(text: str) -> bool:
    stripped = str(text).strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered in {"none", "n/a", "na", "yes", "no"}:
        return True
    if re.fullmatch(r"[A-D][.)]?", stripped):
        return True
    if re.match(r"^[A-D][.)]\s", stripped):
        return True
    banned_phrases = (
        "option ",
        "answer choice",
        "answer is ",
        "the answer is ",
        "final answer",
        "correct answer",
        "yes/no",
        "there is enough information",
        "not visible in any of the provided frames",
    )
    return any(phrase in lowered for phrase in banned_phrases)


def _truncate_words(text: str, max_words: int) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,;:")


def _clean_vst_memory_lines_v2(text: str, max_items: int = 2, max_words: int = 45) -> list[str]:
    cleaned = _strip_repeated_memory_text(str(text))
    cleaned = re.sub(
        r"^\s*(?:New memory bullets?:|Updated memory:|Memory update:|Compact memory:)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    candidates: list[str] = []
    for raw_line in re.split(r"(?:\n|;)+", cleaned):
        line = raw_line.strip()
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        line = re.sub(r"\s+", " ", line)
        if not line or _is_answer_like_memory_line(line):
            continue
        line = _truncate_words(line, max_words)
        if line:
            candidates.append(line)

    if not candidates and cleaned and not _is_answer_like_memory_line(cleaned):
        candidates.append(_truncate_words(cleaned, max_words))

    result: list[str] = []
    seen: set[str] = set()
    remaining_words = max_words
    for line in candidates:
        normalized = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        words = line.split()
        if len(words) > remaining_words:
            line = " ".join(words[:remaining_words]).rstrip(" ,;:")
        result.append(line)
        remaining_words -= len(line.split())
        if len(result) >= max(1, int(max_items)) or remaining_words <= 0:
            break
    return result


def _temporary_max_new_tokens(qa: RecentWindowQAModel, max_new_tokens: int | None):
    class _TokenLimit:
        def __init__(self, model: RecentWindowQAModel, limit: int | None) -> None:
            self.model = model
            self.limit = limit
            self.previous = int(model.max_new_tokens)

        def __enter__(self) -> None:
            if self.limit is not None:
                self.model.max_new_tokens = max(1, int(self.limit))

        def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
            self.model.max_new_tokens = self.previous

    return _TokenLimit(qa, max_new_tokens)


def _select_vst_history_indices(
    flat_frames: list[Image.Image],
    flat_chunk_ids: list[int],
    question_prompt: str,
    recent_start: int,
    memory_clip_size: int,
    memory_max_clips: int,
    memory_backend: str,
    clip_selector: CLIPTopKFrameSelector | None,
) -> list[int]:
    history_budget = max(1, int(memory_clip_size)) * max(1, int(memory_max_clips))
    if recent_start <= 0 or history_budget <= 0:
        return []
    if memory_backend == "clip_topk":
        if clip_selector is None:
            raise ValueError("clip_selector is required when memory_backend='clip_topk'")
        selection = clip_selector.select_topk(
            frames=flat_frames[:recent_start],
            chunk_ids=flat_chunk_ids[:recent_start],
            text=question_prompt,
            top_k=history_budget,
        )
        return sorted(int(index) for index in selection.frame_indices)
    return _select_uniform_indices(recent_start, history_budget)


def _build_vst_memory(
    qa: RecentWindowQAModel,
    flat_frames: list[Image.Image],
    flat_chunk_ids: list[int],
    flat_times: list[tuple[float, float]],
    question_prompt: str,
    recent_frames_only: int,
    memory_clip_size: int,
    memory_max_clips: int,
    memory_backend: str,
    memory_max_tokens: int | None,
    clip_selector: CLIPTopKFrameSelector | None = None,
) -> VSTMemoryBundle:
    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    clip_size = max(1, int(memory_clip_size))
    max_clips = max(1, int(memory_max_clips))

    history_indices = _select_vst_history_indices(
        flat_frames=flat_frames,
        flat_chunk_ids=flat_chunk_ids,
        question_prompt=question_prompt,
        recent_start=recent_start,
        memory_clip_size=clip_size,
        memory_max_clips=max_clips,
        memory_backend=memory_backend,
        clip_selector=clip_selector,
    )
    history_indices = history_indices[: clip_size * max_clips]
    history_chunk_ids = [int(flat_chunk_ids[index]) for index in history_indices]

    memory_entries: list[str] = []
    with _temporary_max_new_tokens(qa, memory_max_tokens):
        for start in range(0, len(history_indices), clip_size):
            group_indices = history_indices[start : start + clip_size]
            if not group_indices:
                continue
            group_times = [flat_times[index] for index in group_indices]
            start_time = min(time_range[0] for time_range in group_times)
            end_time = max(time_range[1] for time_range in group_times)
            memory_text = "\n".join(memory_entries)
            update_prompt = _build_vst_memory_update_prompt(memory_text, start_time, end_time)
            update = qa.generate_from_frames([flat_frames[index] for index in group_indices], update_prompt)
            update = _clean_vst_memory_update(update)
            if update:
                memory_entries.append(f"Time={start_time:.1f}-{end_time:.1f}s: {update}")

    return VSTMemoryBundle(
        entries=memory_entries,
        text="\n".join(memory_entries),
        history_chunk_ids=history_chunk_ids,
        backend=memory_backend,
        clip_size=clip_size,
        max_clips=max_clips,
    )


def query_recent_vst_memory(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    memory_clip_size: int,
    memory_max_clips: int,
    memory_backend: str,
    memory_max_tokens: int | None = None,
    clip_selector: CLIPTopKFrameSelector | None = None,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str, VSTMemoryBundle]:
    flat_frames, flat_chunk_ids, flat_times, decode_backend = _decode_flat_frames_with_times(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        video_start=video_start,
        video_end=video_end,
    )
    memory_bundle = _build_vst_memory(
        qa=qa,
        flat_frames=flat_frames,
        flat_chunk_ids=flat_chunk_ids,
        flat_times=flat_times,
        question_prompt=prompt,
        recent_frames_only=recent_frames_only,
        memory_clip_size=memory_clip_size,
        memory_max_clips=memory_max_clips,
        memory_backend=memory_backend,
        memory_max_tokens=memory_max_tokens,
        clip_selector=clip_selector,
    )

    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    recent_indices = list(range(recent_start, total_frames))
    final_prompt = _build_vst_final_prompt(prompt, memory_bundle.text)
    result, decode_backend = _build_result_from_selection(
        qa=qa,
        selected_frames=[flat_frames[index] for index in recent_indices],
        selected_chunk_ids=[flat_chunk_ids[index] for index in recent_indices],
        prompt=final_prompt,
        decode_backend=decode_backend,
    )
    return result, decode_backend, memory_bundle


def _build_vst_memory_v2(
    qa: RecentWindowQAModel,
    flat_frames: list[Image.Image],
    flat_chunk_ids: list[int],
    flat_times: list[tuple[float, float]],
    question_prompt: str,
    recent_frames_only: int,
    memory_clip_size: int,
    memory_max_clips: int,
    memory_backend: str,
    memory_max_tokens: int | None,
    clip_selector: CLIPTopKFrameSelector | None = None,
) -> VSTMemoryBundle:
    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    clip_size = max(1, int(memory_clip_size))
    max_clips = max(1, int(memory_max_clips))

    history_indices = _select_vst_history_indices(
        flat_frames=flat_frames,
        flat_chunk_ids=flat_chunk_ids,
        question_prompt=question_prompt,
        recent_start=recent_start,
        memory_clip_size=clip_size,
        memory_max_clips=max_clips,
        memory_backend=memory_backend,
        clip_selector=clip_selector,
    )
    history_indices = history_indices[: clip_size * max_clips]
    history_chunk_ids = [int(flat_chunk_ids[index]) for index in history_indices]

    raw_entries: list[str] = []
    update_token_limit = int(memory_max_tokens) if memory_max_tokens is not None else int(qa.max_new_tokens)
    with _temporary_max_new_tokens(qa, update_token_limit):
        for start in range(0, len(history_indices), clip_size):
            group_indices = history_indices[start : start + clip_size]
            if not group_indices:
                continue
            group_times = [flat_times[index] for index in group_indices]
            start_time = min(time_range[0] for time_range in group_times)
            end_time = max(time_range[1] for time_range in group_times)
            previous_memory = "\n".join(raw_entries[-4:])
            update_prompt = _build_vst_memory_update_prompt_v2(
                previous_memory=previous_memory,
                question_prompt=question_prompt,
                start_time=start_time,
                end_time=end_time,
            )
            update = qa.generate_from_frames([flat_frames[index] for index in group_indices], update_prompt)
            for line in _clean_vst_memory_lines_v2(update, max_items=2, max_words=35):
                raw_entries.append(f"Time={start_time:.1f}-{end_time:.1f}s: {line}")

    if not raw_entries:
        return VSTMemoryBundle(
            entries=[],
            text="",
            history_chunk_ids=history_chunk_ids,
            backend=memory_backend,
            clip_size=clip_size,
            max_clips=max_clips,
            version="v2",
            raw_entries=[],
        )

    compression_prompt = _build_vst_memory_compression_prompt_v2(
        question_prompt=question_prompt,
        memory_entries=raw_entries,
        max_items=4,
    )
    with _temporary_max_new_tokens(qa, max(update_token_limit, 128)):
        compressed = qa.generate_from_text(compression_prompt)
    memory_entries = _clean_vst_memory_lines_v2(compressed, max_items=4, max_words=70)
    if not memory_entries:
        memory_entries = raw_entries[:4]

    memory_text = "\n".join(f"- {entry}" for entry in memory_entries)
    return VSTMemoryBundle(
        entries=memory_entries,
        text=memory_text,
        history_chunk_ids=history_chunk_ids,
        backend=memory_backend,
        clip_size=clip_size,
        max_clips=max_clips,
        version="v2",
        raw_entries=raw_entries,
    )


def query_recent_vst_memory_v2(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    memory_clip_size: int,
    memory_max_clips: int,
    memory_backend: str,
    memory_max_tokens: int | None = None,
    clip_selector: CLIPTopKFrameSelector | None = None,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str, VSTMemoryBundle]:
    flat_frames, flat_chunk_ids, flat_times, decode_backend = _decode_flat_frames_with_times(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        video_start=video_start,
        video_end=video_end,
    )
    memory_bundle = _build_vst_memory_v2(
        qa=qa,
        flat_frames=flat_frames,
        flat_chunk_ids=flat_chunk_ids,
        flat_times=flat_times,
        question_prompt=prompt,
        recent_frames_only=recent_frames_only,
        memory_clip_size=memory_clip_size,
        memory_max_clips=memory_max_clips,
        memory_backend=memory_backend,
        memory_max_tokens=memory_max_tokens,
        clip_selector=clip_selector,
    )

    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    recent_indices = list(range(recent_start, total_frames))
    final_prompt = _build_vst_final_prompt_v2(prompt, memory_bundle.text)
    result, decode_backend = _build_result_from_selection(
        qa=qa,
        selected_frames=[flat_frames[index] for index in recent_indices],
        selected_chunk_ids=[flat_chunk_ids[index] for index in recent_indices],
        prompt=final_prompt,
        decode_backend=decode_backend,
    )
    return result, decode_backend, memory_bundle


def _build_text_memory(
    qa: RecentWindowQAModel,
    flat_frames: list[Image.Image],
    flat_chunk_ids: list[int],
    question_prompt: str,
    recent_frames_only: int,
    supplemental_frames: int,
    memory_num_items: int,
    memory_group_size: int,
    memory_backend: str,
    clip_selector: CLIPTopKFrameSelector | None = None,
) -> TextMemoryBundle:
    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    history_chunk_ids: list[int] = []
    if recent_start <= 0 or int(supplemental_frames) <= 0:
        return TextMemoryBundle(lines=[], raw_summary="", captions=[], history_chunk_ids=history_chunk_ids)

    if memory_backend == "clip_topk":
        if clip_selector is None:
            raise ValueError("clip_selector is required when memory_backend='clip_topk'")
        selection = clip_selector.select_topk(
            frames=flat_frames[:recent_start],
            chunk_ids=flat_chunk_ids[:recent_start],
            text=question_prompt,
            top_k=int(supplemental_frames),
        )
        history_indices = selection.frame_indices
        history_chunk_ids = [int(chunk_id) for chunk_id in selection.chunk_ids]
    else:
        history_indices = _select_uniform_indices(recent_start, supplemental_frames)
        history_chunk_ids = [int(flat_chunk_ids[index]) for index in history_indices]

    history_indices = sorted(history_indices)
    if not history_indices:
        return TextMemoryBundle(lines=[], raw_summary="", captions=[], history_chunk_ids=history_chunk_ids)

    group_size = max(1, int(memory_group_size))
    captions: list[str] = []
    seen_captions: set[str] = set()
    for start in range(0, len(history_indices), group_size):
        group_indices = history_indices[start : start + group_size]
        group_frames = [flat_frames[index] for index in group_indices]
        caption = qa.generate_from_frames(group_frames, _build_memory_caption_prompt())
        caption = _clean_caption_text(caption)
        if caption and caption.lower() not in seen_captions:
            seen_captions.add(caption.lower())
            captions.append(caption)

    if not captions:
        return TextMemoryBundle(lines=[], raw_summary="", captions=[], history_chunk_ids=history_chunk_ids)

    raw_summary = qa.generate_from_text(
        _build_memory_summary_prompt(
            question_prompt=question_prompt,
            captions=captions,
            max_items=max(1, int(memory_num_items)),
        )
    )
    lines = _normalize_memory_lines(raw_summary, memory_num_items)
    return TextMemoryBundle(
        lines=lines,
        raw_summary=raw_summary,
        captions=captions,
        history_chunk_ids=history_chunk_ids,
    )


def query_recent_text_memory(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    supplemental_frames: int,
    memory_num_items: int,
    memory_group_size: int,
    memory_backend: str,
    clip_selector: CLIPTopKFrameSelector | None = None,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str, TextMemoryBundle]:
    flat_frames, flat_chunk_ids, decode_backend = _decode_flat_frames(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        video_start=video_start,
        video_end=video_end,
    )
    memory_bundle = _build_text_memory(
        qa=qa,
        flat_frames=flat_frames,
        flat_chunk_ids=flat_chunk_ids,
        question_prompt=prompt,
        recent_frames_only=recent_frames_only,
        supplemental_frames=supplemental_frames,
        memory_num_items=memory_num_items,
        memory_group_size=memory_group_size,
        memory_backend=memory_backend,
        clip_selector=clip_selector,
    )
    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    recent_indices = list(range(recent_start, total_frames))
    augmented_prompt = _build_prompt_with_memory(prompt, memory_bundle.lines)
    result, decode_backend = _build_result_from_selection(
        qa=qa,
        selected_frames=[flat_frames[index] for index in recent_indices],
        selected_chunk_ids=[flat_chunk_ids[index] for index in recent_indices],
        prompt=augmented_prompt,
        decode_backend=decode_backend,
    )
    return result, decode_backend, memory_bundle


def _select_stratified_state_memory_indices(recent_start: int, supplemental_frames: int) -> list[int]:
    total = max(0, int(supplemental_frames))
    if recent_start <= 0 or total <= 0:
        return []

    near_count = min(recent_start, max(1, total // 2))
    global_count = max(0, total - near_count)
    near_start = max(0, recent_start - near_count)
    near_indices = list(range(near_start, recent_start))
    global_indices = _select_uniform_indices(near_start, global_count) if global_count > 0 and near_start > 0 else []

    selected = sorted(set(global_indices + near_indices))
    if len(selected) < min(total, recent_start):
        selected_set = set(selected)
        for index in reversed(range(recent_start)):
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
            if len(selected) >= min(total, recent_start):
                break
    return sorted(selected[: min(total, recent_start)])


def _build_state_text_memory(
    qa: RecentWindowQAModel,
    flat_frames: list[Image.Image],
    flat_chunk_ids: list[int],
    question_prompt: str,
    recent_frames_only: int,
    supplemental_frames: int,
    memory_num_items: int,
    memory_group_size: int,
    memory_backend: str,
    clip_selector: CLIPTopKFrameSelector | None = None,
) -> TextMemoryBundle:
    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    history_chunk_ids: list[int] = []
    if recent_start <= 0 or int(supplemental_frames) <= 0:
        return TextMemoryBundle(
            lines=[],
            raw_summary="",
            captions=[],
            history_chunk_ids=history_chunk_ids,
            memory_type="state_v4",
        )

    if memory_backend == "clip_topk":
        if clip_selector is None:
            raise ValueError("clip_selector is required when memory_backend='clip_topk'")
        selection = clip_selector.select_topk(
            frames=flat_frames[:recent_start],
            chunk_ids=flat_chunk_ids[:recent_start],
            text=question_prompt,
            top_k=int(supplemental_frames),
        )
        history_indices = sorted(selection.frame_indices)
    elif memory_backend == "stratified":
        history_indices = _select_stratified_state_memory_indices(recent_start, supplemental_frames)
    else:
        history_indices = _select_uniform_indices(recent_start, supplemental_frames)

    history_indices = sorted(history_indices)
    history_chunk_ids = [int(flat_chunk_ids[index]) for index in history_indices]
    if not history_indices:
        return TextMemoryBundle(
            lines=[],
            raw_summary="",
            captions=[],
            history_chunk_ids=history_chunk_ids,
            memory_type="state_v4",
        )

    group_size = max(1, int(memory_group_size))
    captions: list[str] = []
    seen_captions: set[str] = set()
    for start in range(0, len(history_indices), group_size):
        group_indices = history_indices[start : start + group_size]
        group_frames = [flat_frames[index] for index in group_indices]
        caption = qa.generate_from_frames(group_frames, _build_state_memory_caption_prompt())
        caption = _clean_state_caption_text(caption)
        if caption and caption.lower() not in seen_captions:
            seen_captions.add(caption.lower())
            captions.append(caption)

    if not captions:
        return TextMemoryBundle(
            lines=[],
            raw_summary="",
            captions=[],
            history_chunk_ids=history_chunk_ids,
            memory_type="state_v4",
        )

    raw_summary = qa.generate_from_text(
        _build_state_memory_summary_prompt(
            question_prompt=question_prompt,
            captions=captions,
            max_items=max(1, int(memory_num_items)),
        )
    )
    lines = _normalize_state_memory_lines(raw_summary, memory_num_items)
    return TextMemoryBundle(
        lines=lines,
        raw_summary=raw_summary,
        captions=captions,
        history_chunk_ids=history_chunk_ids,
        memory_type="state_v4",
    )


def query_recent_state_text_memory(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    supplemental_frames: int,
    memory_num_items: int,
    memory_group_size: int,
    memory_backend: str,
    clip_selector: CLIPTopKFrameSelector | None = None,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str, TextMemoryBundle]:
    flat_frames, flat_chunk_ids, decode_backend = _decode_flat_frames(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        video_start=video_start,
        video_end=video_end,
    )
    memory_bundle = _build_state_text_memory(
        qa=qa,
        flat_frames=flat_frames,
        flat_chunk_ids=flat_chunk_ids,
        question_prompt=prompt,
        recent_frames_only=recent_frames_only,
        supplemental_frames=supplemental_frames,
        memory_num_items=memory_num_items,
        memory_group_size=memory_group_size,
        memory_backend=memory_backend,
        clip_selector=clip_selector,
    )
    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    recent_indices = list(range(recent_start, total_frames))
    augmented_prompt = _build_prompt_with_state_memory(prompt, memory_bundle.lines)
    result, decode_backend = _build_result_from_selection(
        qa=qa,
        selected_frames=[flat_frames[index] for index in recent_indices],
        selected_chunk_ids=[flat_chunk_ids[index] for index in recent_indices],
        prompt=augmented_prompt,
        decode_backend=decode_backend,
    )
    return result, decode_backend, memory_bundle


def query_uniform_frames(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    num_frames: int,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str]:
    flat_frames, flat_chunk_ids, decode_backend = _decode_flat_frames(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        video_start=video_start,
        video_end=video_end,
    )
    indices = _select_uniform_indices(len(flat_frames), num_frames)
    selected_frames = [flat_frames[index] for index in indices]
    selected_chunk_ids = [flat_chunk_ids[index] for index in indices]
    return _build_result_from_selection(
        qa=qa,
        selected_frames=selected_frames,
        selected_chunk_ids=selected_chunk_ids,
        prompt=prompt,
        decode_backend=decode_backend,
    )


def query_clip_topk(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    num_frames: int,
    clip_selector: CLIPTopKFrameSelector,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str]:
    flat_frames, flat_chunk_ids, decode_backend = _decode_flat_frames(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        video_start=video_start,
        video_end=video_end,
    )
    selection = clip_selector.select_topk(
        frames=flat_frames,
        chunk_ids=flat_chunk_ids,
        text=prompt,
        top_k=num_frames,
    )
    return _build_result_from_selection(
        qa=qa,
        selected_frames=selection.frames,
        selected_chunk_ids=selection.chunk_ids,
        prompt=prompt,
        decode_backend=decode_backend,
    )


def query_recent_uniform_frames(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    supplemental_frames: int,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str]:
    flat_frames, flat_chunk_ids, decode_backend = _decode_flat_frames(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        video_start=video_start,
        video_end=video_end,
    )
    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    recent_indices = list(range(recent_start, total_frames))
    past_indices = _select_uniform_indices(recent_start, supplemental_frames)
    final_indices = sorted(set(past_indices + recent_indices))

    return _build_result_from_selection(
        qa=qa,
        selected_frames=[flat_frames[index] for index in final_indices],
        selected_chunk_ids=[flat_chunk_ids[index] for index in final_indices],
        prompt=prompt,
        decode_backend=decode_backend,
    )


def query_recent_clip_topk_frames(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    supplemental_frames: int,
    clip_selector: CLIPTopKFrameSelector,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str]:
    flat_frames, flat_chunk_ids, decode_backend = _decode_flat_frames(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        video_start=video_start,
        video_end=video_end,
    )
    total_frames = len(flat_frames)
    recent_count = max(1, int(recent_frames_only))
    recent_start = max(0, total_frames - recent_count)
    recent_indices = list(range(recent_start, total_frames))

    if recent_start > 0 and int(supplemental_frames) > 0:
        history_selection = clip_selector.select_topk(
            frames=flat_frames[:recent_start],
            chunk_ids=flat_chunk_ids[:recent_start],
            text=prompt,
            top_k=int(supplemental_frames),
        )
        history_indices = history_selection.frame_indices
    else:
        history_indices = []

    final_indices = sorted(set(history_indices + recent_indices))
    return _build_result_from_selection(
        qa=qa,
        selected_frames=[flat_frames[index] for index in final_indices],
        selected_chunk_ids=[flat_chunk_ids[index] for index in final_indices],
        prompt=prompt,
        decode_backend=decode_backend,
    )


def evaluate_ovo_backward_realtime(
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    frame_selection: str = "recent",
    supplemental_frames: int = 0,
    clip_selector: CLIPTopKFrameSelector | None = None,
    memory_num_items: int = 3,
    memory_group_size: int = 4,
    memory_clip_size: int = 4,
    memory_max_clips: int = 4,
    memory_max_tokens: int | None = None,
) -> dict:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    response = None
    metadata: dict = {}
    if os.path.exists(video_path):
        base_prompt = build_ovo_prompt(anno["task"], anno)
        if frame_selection == "recent_clip_topk":
            if clip_selector is None:
                raise ValueError("clip_selector is required when frame_selection='recent_clip_topk'")
            result, decode_backend = query_recent_clip_topk_frames(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
                clip_selector=clip_selector,
            )
        elif frame_selection == "recent_memory_clip_topk":
            if clip_selector is None:
                raise ValueError("clip_selector is required when frame_selection='recent_memory_clip_topk'")
            result, decode_backend, memory_bundle = query_recent_text_memory(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
                memory_num_items=memory_num_items,
                memory_group_size=memory_group_size,
                memory_backend="clip_topk",
                clip_selector=clip_selector,
            )
        elif frame_selection in {
            "recent_state_memory_uniform_v4",
            "recent_state_memory_clip_topk_v4",
            "recent_state_memory_stratified_v4",
        }:
            if frame_selection == "recent_state_memory_clip_topk_v4" and clip_selector is None:
                raise ValueError(f"clip_selector is required when frame_selection='{frame_selection}'")
            state_memory_backend = {
                "recent_state_memory_uniform_v4": "uniform",
                "recent_state_memory_clip_topk_v4": "clip_topk",
                "recent_state_memory_stratified_v4": "stratified",
            }[frame_selection]
            result, decode_backend, memory_bundle = query_recent_state_text_memory(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
                memory_num_items=memory_num_items,
                memory_group_size=memory_group_size,
                memory_backend=state_memory_backend,
                clip_selector=clip_selector,
            )
        elif frame_selection == "recent_uniform":
            result, decode_backend = query_recent_uniform_frames(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
            )
        elif frame_selection == "recent_memory_uniform":
            result, decode_backend, memory_bundle = query_recent_text_memory(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
                memory_num_items=memory_num_items,
                memory_group_size=memory_group_size,
                memory_backend="uniform",
            )
        elif frame_selection in {"recent_vst_memory_clip_topk", "recent_vst_memory_clip_topk_v2"}:
            if clip_selector is None:
                raise ValueError(f"clip_selector is required when frame_selection='{frame_selection}'")
            query_vst_memory_fn = query_recent_vst_memory_v2 if frame_selection.endswith("_v2") else query_recent_vst_memory
            result, decode_backend, vst_memory_bundle = query_vst_memory_fn(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                memory_clip_size=memory_clip_size,
                memory_max_clips=memory_max_clips,
                memory_backend="clip_topk",
                memory_max_tokens=memory_max_tokens,
                clip_selector=clip_selector,
            )
        elif frame_selection in {"recent_vst_memory_uniform", "recent_vst_memory_uniform_v2"}:
            query_vst_memory_fn = query_recent_vst_memory_v2 if frame_selection.endswith("_v2") else query_recent_vst_memory
            result, decode_backend, vst_memory_bundle = query_vst_memory_fn(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                memory_clip_size=memory_clip_size,
                memory_max_clips=memory_max_clips,
                memory_backend="uniform",
                memory_max_tokens=memory_max_tokens,
            )
        elif frame_selection == "clip_topk":
            if clip_selector is None:
                raise ValueError("clip_selector is required when frame_selection='clip_topk'")
            result, decode_backend = query_clip_topk(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                num_frames=recent_frames_only,
                clip_selector=clip_selector,
            )
        elif frame_selection == "uniform":
            result, decode_backend = query_uniform_frames(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                num_frames=recent_frames_only,
            )
        else:
            result, decode_backend = query_recent_window(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
            )
        response = result.answer
        metadata = {
            "decode_backend": decode_backend,
            "final_chunk_ids": result.final_chunk_ids,
            "generate_time": result.generate_time,
            "ttft_seconds": result.ttft_seconds,
            "num_vision_tokens": result.num_vision_tokens,
            "num_vision_tokens_before": result.num_vision_tokens_before,
            "num_vision_tokens_after": result.num_vision_tokens_after,
            "num_frames": result.num_frames,
            "frame_selection": frame_selection,
            "supplemental_frames": int(supplemental_frames),
        }
        if frame_selection in {
            "recent_memory_uniform",
            "recent_memory_clip_topk",
            "recent_state_memory_uniform_v4",
            "recent_state_memory_clip_topk_v4",
            "recent_state_memory_stratified_v4",
        }:
            metadata["memory_lines"] = memory_bundle.lines
            metadata["memory_raw_summary"] = memory_bundle.raw_summary
            metadata["memory_captions"] = memory_bundle.captions
            metadata["history_chunk_ids"] = memory_bundle.history_chunk_ids
            metadata["memory_type"] = memory_bundle.memory_type
        if frame_selection in {
            "recent_vst_memory_uniform",
            "recent_vst_memory_clip_topk",
            "recent_vst_memory_uniform_v2",
            "recent_vst_memory_clip_topk_v2",
        }:
            metadata["memory_entries"] = vst_memory_bundle.entries
            metadata["memory_text"] = vst_memory_bundle.text
            metadata["memory_history_chunk_ids"] = vst_memory_bundle.history_chunk_ids
            metadata["memory_backend"] = vst_memory_bundle.backend
            metadata["memory_clip_size"] = vst_memory_bundle.clip_size
            metadata["memory_max_clips"] = vst_memory_bundle.max_clips
            metadata["memory_version"] = vst_memory_bundle.version
            if vst_memory_bundle.raw_entries is not None:
                metadata["memory_raw_entries"] = vst_memory_bundle.raw_entries
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "response": response,
        "ground_truth": chr(65 + anno["gt"]),
        **metadata,
    }


def evaluate_ovo_forward(
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    frame_selection: str = "recent",
    supplemental_frames: int = 0,
    clip_selector: CLIPTopKFrameSelector | None = None,
    memory_num_items: int = 3,
    memory_group_size: int = 4,
    memory_clip_size: int = 4,
    memory_max_clips: int = 4,
    memory_max_tokens: int | None = None,
) -> dict:
    result_anno = copy.deepcopy(anno)
    for index, test_info in enumerate(result_anno["test_info"]):
        video_path = os.path.join(chunked_dir, f"{anno['id']}_{index}.mp4")
        if not os.path.exists(video_path):
            test_info["response"] = None
            continue
        base_prompt = build_ovo_prompt(anno["task"], anno, index=index)
        if frame_selection == "recent_clip_topk":
            if clip_selector is None:
                raise ValueError("clip_selector is required when frame_selection='recent_clip_topk'")
            result, decode_backend = query_recent_clip_topk_frames(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
                clip_selector=clip_selector,
            )
        elif frame_selection == "recent_memory_clip_topk":
            if clip_selector is None:
                raise ValueError("clip_selector is required when frame_selection='recent_memory_clip_topk'")
            result, decode_backend, memory_bundle = query_recent_text_memory(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
                memory_num_items=memory_num_items,
                memory_group_size=memory_group_size,
                memory_backend="clip_topk",
                clip_selector=clip_selector,
            )
        elif frame_selection in {
            "recent_state_memory_uniform_v4",
            "recent_state_memory_clip_topk_v4",
            "recent_state_memory_stratified_v4",
        }:
            if frame_selection == "recent_state_memory_clip_topk_v4" and clip_selector is None:
                raise ValueError(f"clip_selector is required when frame_selection='{frame_selection}'")
            state_memory_backend = {
                "recent_state_memory_uniform_v4": "uniform",
                "recent_state_memory_clip_topk_v4": "clip_topk",
                "recent_state_memory_stratified_v4": "stratified",
            }[frame_selection]
            result, decode_backend, memory_bundle = query_recent_state_text_memory(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
                memory_num_items=memory_num_items,
                memory_group_size=memory_group_size,
                memory_backend=state_memory_backend,
                clip_selector=clip_selector,
            )
        elif frame_selection == "recent_uniform":
            result, decode_backend = query_recent_uniform_frames(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
            )
        elif frame_selection == "recent_memory_uniform":
            result, decode_backend, memory_bundle = query_recent_text_memory(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                supplemental_frames=supplemental_frames,
                memory_num_items=memory_num_items,
                memory_group_size=memory_group_size,
                memory_backend="uniform",
            )
        elif frame_selection in {"recent_vst_memory_clip_topk", "recent_vst_memory_clip_topk_v2"}:
            if clip_selector is None:
                raise ValueError(f"clip_selector is required when frame_selection='{frame_selection}'")
            query_vst_memory_fn = query_recent_vst_memory_v2 if frame_selection.endswith("_v2") else query_recent_vst_memory
            result, decode_backend, vst_memory_bundle = query_vst_memory_fn(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                memory_clip_size=memory_clip_size,
                memory_max_clips=memory_max_clips,
                memory_backend="clip_topk",
                memory_max_tokens=memory_max_tokens,
                clip_selector=clip_selector,
            )
        elif frame_selection in {"recent_vst_memory_uniform", "recent_vst_memory_uniform_v2"}:
            query_vst_memory_fn = query_recent_vst_memory_v2 if frame_selection.endswith("_v2") else query_recent_vst_memory
            result, decode_backend, vst_memory_bundle = query_vst_memory_fn(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                memory_clip_size=memory_clip_size,
                memory_max_clips=memory_max_clips,
                memory_backend="uniform",
                memory_max_tokens=memory_max_tokens,
            )
        elif frame_selection == "clip_topk":
            if clip_selector is None:
                raise ValueError("clip_selector is required when frame_selection='clip_topk'")
            result, decode_backend = query_clip_topk(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                num_frames=recent_frames_only,
                clip_selector=clip_selector,
            )
        elif frame_selection == "uniform":
            result, decode_backend = query_uniform_frames(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                num_frames=recent_frames_only,
            )
        else:
            result, decode_backend = query_recent_window(
                qa=qa,
                video_path=video_path,
                prompt=base_prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
            )
        test_info["response"] = result.answer
        test_info["decode_backend"] = decode_backend
        test_info["final_chunk_ids"] = result.final_chunk_ids
        test_info["generate_time"] = result.generate_time
        test_info["ttft_seconds"] = result.ttft_seconds
        test_info["num_vision_tokens"] = result.num_vision_tokens
        test_info["num_vision_tokens_before"] = result.num_vision_tokens_before
        test_info["num_vision_tokens_after"] = result.num_vision_tokens_after
        test_info["num_frames"] = result.num_frames
        test_info["frame_selection"] = frame_selection
        test_info["supplemental_frames"] = int(supplemental_frames)
        if frame_selection in {
            "recent_memory_uniform",
            "recent_memory_clip_topk",
            "recent_state_memory_uniform_v4",
            "recent_state_memory_clip_topk_v4",
            "recent_state_memory_stratified_v4",
        }:
            test_info["memory_lines"] = memory_bundle.lines
            test_info["memory_raw_summary"] = memory_bundle.raw_summary
            test_info["memory_captions"] = memory_bundle.captions
            test_info["history_chunk_ids"] = memory_bundle.history_chunk_ids
            test_info["memory_type"] = memory_bundle.memory_type
        if frame_selection in {
            "recent_vst_memory_uniform",
            "recent_vst_memory_clip_topk",
            "recent_vst_memory_uniform_v2",
            "recent_vst_memory_clip_topk_v2",
        }:
            test_info["memory_entries"] = vst_memory_bundle.entries
            test_info["memory_text"] = vst_memory_bundle.text
            test_info["memory_history_chunk_ids"] = vst_memory_bundle.history_chunk_ids
            test_info["memory_backend"] = vst_memory_bundle.backend
            test_info["memory_clip_size"] = vst_memory_bundle.clip_size
            test_info["memory_max_clips"] = vst_memory_bundle.max_clips
            test_info["memory_version"] = vst_memory_bundle.version
            if vst_memory_bundle.raw_entries is not None:
                test_info["memory_raw_entries"] = vst_memory_bundle.raw_entries
    return result_anno
