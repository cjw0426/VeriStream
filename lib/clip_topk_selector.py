from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image


@dataclass
class CLIPSelectionResult:
    frames: list[Image.Image]
    chunk_ids: list[int]
    frame_indices: list[int]
    scores: list[float]


class CLIPTopKFrameSelector:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str | torch.device = "auto",
        batch_size: int = 32,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        resolved_device = self._resolve_device(device)
        self.device = resolved_device
        self.batch_size = max(1, int(batch_size))
        self.model_name = model_name
        self.model_dtype = torch.float16 if resolved_device.type == "cuda" else torch.float32

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=self.model_dtype)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if str(device).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @torch.inference_mode()
    def score_frames(self, frames: Sequence[Image.Image], text: str) -> list[float]:
        if not frames:
            return []

        text_inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        scores: list[float] = []
        for start in range(0, len(frames), self.batch_size):
            batch_frames = list(frames[start : start + self.batch_size])
            image_inputs = self.processor(images=batch_frames, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(self.device, dtype=self.model_dtype)
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            batch_scores = torch.matmul(image_features, text_features.transpose(0, 1)).squeeze(-1)
            scores.extend(float(score) for score in batch_scores.detach().cpu().tolist())

        return scores

    @torch.inference_mode()
    def image_embeddings(self, frames: Sequence[Image.Image]) -> torch.Tensor:
        """返回归一化图像向量，供流式变化检测复用冻结 CLIP。"""
        if not frames:
            return torch.empty((0, 0), dtype=torch.float32)

        batches: list[torch.Tensor] = []
        for start in range(0, len(frames), self.batch_size):
            image_inputs = self.processor(images=list(frames[start : start + self.batch_size]), return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(self.device, dtype=self.model_dtype)
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            batches.append(torch.nn.functional.normalize(image_features, dim=-1).float().cpu())
        return torch.cat(batches, dim=0)

    @torch.inference_mode()
    def select_topk(
        self,
        frames: Sequence[Image.Image],
        chunk_ids: Sequence[int],
        text: str,
        top_k: int,
    ) -> CLIPSelectionResult:
        if len(frames) != len(chunk_ids):
            raise ValueError("frames and chunk_ids must have the same length")
        if not frames:
            raise ValueError("No frames available for CLIP top-k selection")

        scores = self.score_frames(frames, text)
        count = min(max(1, int(top_k)), len(frames))
        ranked_indices = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
        selected_indices = sorted(ranked_indices[:count])

        return CLIPSelectionResult(
            frames=[frames[index] for index in selected_indices],
            chunk_ids=[int(chunk_ids[index]) for index in selected_indices],
            frame_indices=selected_indices,
            scores=[float(scores[index]) for index in selected_indices],
        )
