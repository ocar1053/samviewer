from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from samviewer.metrics import BoundingBox, clamp_bbox


@dataclass
class SegmentationResult:
    mask: np.ndarray
    bbox: BoundingBox
    method: str


class OptionalSegmenter:
    """Small extension point for SAM2 or another segmentation backend."""

    def __init__(self, backend: str = "bbox") -> None:
        self.backend = backend

    @property
    def available(self) -> bool:
        if self.backend == "bbox":
            return True
        if self.backend == "sam2":
            return _sam2_available()
        return False

    def segment_from_bbox(self, image: np.ndarray, bbox: BoundingBox) -> SegmentationResult:
        bbox = clamp_bbox(bbox, image.shape)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = 255
        return SegmentationResult(mask=mask, bbox=bbox, method=self.backend)


def _sam2_available() -> bool:
    try:
        import sam2  # noqa: F401
    except Exception:
        return False
    return True
