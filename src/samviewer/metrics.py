from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Mapping


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class AlignmentMetrics:
    ref_bbox: BoundingBox
    real_bbox: BoundingBox
    width_error_px: int
    height_error_px: int
    area_error_px: int
    width_error_pct: float
    height_error_pct: float
    area_error_pct: float
    center_dx_px: float
    center_dy_px: float
    center_distance_px: float
    max_size_error_pct: float

    @property
    def scale_status(self) -> str:
        if self.max_size_error_pct <= 5.0:
            return "within_5"
        if self.max_size_error_pct <= 10.0:
            return "within_10"
        return "outside_10"


def clamp_bbox(bbox: BoundingBox, image_shape: tuple[int, ...]) -> BoundingBox:
    image_h, image_w = image_shape[:2]
    x = min(max(0, int(bbox.x)), max(0, image_w - 1))
    y = min(max(0, int(bbox.y)), max(0, image_h - 1))
    width = min(max(1, int(bbox.width)), image_w - x)
    height = min(max(1, int(bbox.height)), image_h - y)
    return BoundingBox(x=x, y=y, width=width, height=height)


def scale_bbox(
    bbox: BoundingBox,
    from_shape: tuple[int, ...],
    to_shape: tuple[int, ...],
) -> BoundingBox:
    from_h, from_w = from_shape[:2]
    to_h, to_w = to_shape[:2]
    if from_w <= 0 or from_h <= 0:
        return bbox

    sx = to_w / from_w
    sy = to_h / from_h
    return clamp_bbox(
        BoundingBox(
            x=round(bbox.x * sx),
            y=round(bbox.y * sy),
            width=round(bbox.width * sx),
            height=round(bbox.height * sy),
        ),
        to_shape,
    )


def bbox_from_mapping(data: Mapping[str, int] | None) -> BoundingBox | None:
    if not data:
        return None
    return BoundingBox(
        x=int(data["x"]),
        y=int(data["y"]),
        width=int(data["width"]),
        height=int(data["height"]),
    )


def bbox_to_mapping(bbox: BoundingBox | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}


def _pct_error(real_value: float, ref_value: float) -> float:
    if ref_value == 0:
        return 0.0
    return abs(real_value - ref_value) / abs(ref_value) * 100.0


def compute_alignment(ref_bbox: BoundingBox, real_bbox: BoundingBox) -> AlignmentMetrics:
    ref_cx, ref_cy = ref_bbox.center
    real_cx, real_cy = real_bbox.center
    center_dx = real_cx - ref_cx
    center_dy = real_cy - ref_cy
    width_error = real_bbox.width - ref_bbox.width
    height_error = real_bbox.height - ref_bbox.height
    area_error = real_bbox.area - ref_bbox.area
    width_error_pct = _pct_error(real_bbox.width, ref_bbox.width)
    height_error_pct = _pct_error(real_bbox.height, ref_bbox.height)

    return AlignmentMetrics(
        ref_bbox=ref_bbox,
        real_bbox=real_bbox,
        width_error_px=width_error,
        height_error_px=height_error,
        area_error_px=area_error,
        width_error_pct=width_error_pct,
        height_error_pct=height_error_pct,
        area_error_pct=_pct_error(real_bbox.area, ref_bbox.area),
        center_dx_px=center_dx,
        center_dy_px=center_dy,
        center_distance_px=hypot(center_dx, center_dy),
        max_size_error_pct=max(width_error_pct, height_error_pct),
    )
