from __future__ import annotations

import cv2
import numpy as np

from samviewer.metrics import BoundingBox, Point, clamp_bbox, scale_bbox


REF_COLOR = (0, 200, 255)
REAL_COLOR = (80, 220, 80)
CENTER_COLOR = (255, 80, 80)


def draw_bbox(
    image: np.ndarray,
    bbox: BoundingBox | None,
    color: tuple[int, int, int] = REF_COLOR,
    label: str | None = None,
    thickness: int = 2,
) -> np.ndarray:
    out = image.copy()
    if bbox is None:
        return out

    bbox = clamp_bbox(bbox, out.shape)
    pt1 = (bbox.x, bbox.y)
    pt2 = (bbox.right, bbox.bottom)
    cv2.rectangle(out, pt1, pt2, color, thickness)
    cx, cy = bbox.center
    cv2.drawMarker(
        out,
        (round(cx), round(cy)),
        CENTER_COLOR,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )
    if label:
        cv2.putText(
            out,
            label,
            (bbox.x, max(18, bbox.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def overlay_bboxes(
    image: np.ndarray,
    ref_bbox: BoundingBox | None = None,
    real_bbox: BoundingBox | None = None,
) -> np.ndarray:
    out = image.copy()
    out = draw_bbox(out, ref_bbox, REF_COLOR, "reference")
    out = draw_bbox(out, real_bbox, REAL_COLOR, "real")
    return out


def draw_points_polygon(
    image: np.ndarray,
    points: list[Point] | None,
    color: tuple[int, int, int] = REF_COLOR,
    label: str | None = None,
    closed: bool = True,
) -> np.ndarray:
    out = image.copy()
    if not points:
        return out

    int_points = [(round(point[0]), round(point[1])) for point in points]
    if len(int_points) >= 2:
        cv2.polylines(
            out,
            [np.asarray(int_points, dtype=np.int32)],
            isClosed=closed and len(int_points) >= 3,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    for index, point in enumerate(int_points, start=1):
        cv2.circle(out, point, 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            str(index),
            (point[0] + 7, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if label:
        x = min(point[0] for point in int_points)
        y = min(point[1] for point in int_points)
        cv2.putText(
            out,
            label,
            (x, max(18, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def draw_mask_outline(
    image: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int] = REF_COLOR,
    label: str | None = None,
) -> np.ndarray:
    out = image.copy()
    if mask is None:
        return out
    outline = mask_outline(mask, color)
    outline_pixels = outline.any(axis=2)
    out[outline_pixels] = outline[outline_pixels]
    if label:
        ys, xs = np.where(mask > 0)
        if len(xs) and len(ys):
            cv2.putText(
                out,
                label,
                (int(xs.min()), max(18, int(ys.min()) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def overlay_reference_image(
    reference: np.ndarray,
    real: np.ndarray,
    alpha: float,
    ref_bbox: BoundingBox | None = None,
    real_bbox: BoundingBox | None = None,
) -> np.ndarray:
    resized_ref = cv2.resize(reference, (real.shape[1], real.shape[0]))
    blended = cv2.addWeighted(real, 1.0 - alpha, resized_ref, alpha, 0)
    scaled_ref_bbox = (
        scale_bbox(ref_bbox, reference.shape, real.shape) if ref_bbox is not None else None
    )
    return overlay_bboxes(blended, scaled_ref_bbox, real_bbox)


def mask_outline(mask: np.ndarray, color: tuple[int, int, int] = REF_COLOR) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    cv2.drawContours(outline, contours, -1, color, 2)
    return outline
