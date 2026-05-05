from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from samviewer.metrics import BoundingBox, clamp_bbox


BackendName = Literal["bbox", "sam2"]


@dataclass
class SegmentationResult:
    mask: np.ndarray
    bbox: BoundingBox
    method: str
    score: float | None = None


@dataclass(frozen=True)
class Sam2Config:
    model_id: str | None = None
    checkpoint: str | None = None
    model_cfg: str | None = None
    device: str = "auto"
    multimask_output: bool = True


class OptionalSegmenter:
    """Optional segmentation backend with a lightweight bbox fallback.

    SAM2 is loaded lazily so the main viewer remains usable without PyTorch,
    model checkpoints, or the SAM2 package installed.
    """

    def __init__(
        self,
        backend: BackendName = "bbox",
        sam2_config: Sam2Config | None = None,
    ) -> None:
        self.backend = backend
        self.sam2_config = sam2_config or Sam2Config()
        self._predictor = None

    @property
    def available(self) -> bool:
        if self.backend == "bbox":
            return True
        if self.backend == "sam2":
            return self.sam2_status().available
        return False

    def sam2_status(self) -> "Sam2Status":
        return sam2_status()

    def segment_from_bbox(self, image: np.ndarray, bbox: BoundingBox) -> SegmentationResult:
        bbox = clamp_bbox(bbox, image.shape)
        if self.backend == "sam2":
            return self._segment_with_sam2(image, bbox)
        return _segment_bbox_mask(image, bbox)

    def _segment_with_sam2(self, image: np.ndarray, bbox: BoundingBox) -> SegmentationResult:
        status = self.sam2_status()
        if not status.available:
            raise RuntimeError(status.message)

        predictor = self._get_sam2_predictor()
        import torch

        device = _resolve_device(self.sam2_config.device)
        box_xyxy = np.array([bbox.x, bbox.y, bbox.right, bbox.bottom], dtype=np.float32)
        autocast = _autocast_context(device)

        with torch.inference_mode(), autocast:
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                box=box_xyxy,
                multimask_output=self.sam2_config.multimask_output,
            )

        if len(masks) == 0:
            raise RuntimeError("SAM2 returned no masks for the selected ROI.")

        best_index = int(np.argmax(scores)) if len(scores) else 0
        mask = np.asarray(masks[best_index]).astype(np.uint8) * 255
        tight_bbox = bbox_from_mask(mask) or bbox
        score = float(scores[best_index]) if len(scores) else None
        return SegmentationResult(mask=mask, bbox=tight_bbox, method="sam2", score=score)

    def _get_sam2_predictor(self):
        if self._predictor is not None:
            return self._predictor

        config = self.sam2_config
        device = _resolve_device(config.device)
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if config.model_id:
            self._predictor = SAM2ImagePredictor.from_pretrained(
                config.model_id,
                device=device,
            )
            return self._predictor

        if not config.checkpoint or not config.model_cfg:
            raise RuntimeError(
                "SAM2 backend needs either a Hugging Face model id or both "
                "a model config and checkpoint path."
            )

        checkpoint = str(Path(config.checkpoint).expanduser())
        from sam2.build_sam import build_sam2

        sam_model = build_sam2(config.model_cfg, checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(sam_model)
        return self._predictor


@dataclass(frozen=True)
class Sam2Status:
    available: bool
    message: str


def sam2_status() -> Sam2Status:
    missing: list[str] = []
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")
    try:
        import sam2  # noqa: F401
    except Exception:
        missing.append("sam2")

    if missing:
        return Sam2Status(
            available=False,
            message=(
                "SAM2 backend is not available. Missing package(s): "
                + ", ".join(missing)
                + ". Use the bbox fallback or install SAM2 separately."
            ),
        )
    return Sam2Status(available=True, message="SAM2 backend is available.")


def bbox_from_mask(mask: np.ndarray) -> BoundingBox | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return BoundingBox(x=x0, y=y0, width=x1 - x0 + 1, height=y1 - y0 + 1)


def _segment_bbox_mask(image: np.ndarray, bbox: BoundingBox) -> SegmentationResult:
    bbox = clamp_bbox(bbox, image.shape)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = 255
    return SegmentationResult(mask=mask, bbox=bbox, method="bbox")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast_context(device: str):
    import contextlib

    if device != "cuda":
        return contextlib.nullcontext()
    import torch

    return torch.autocast("cuda", dtype=torch.bfloat16)


def _sam2_available() -> bool:
    return sam2_status().available
