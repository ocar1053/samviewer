from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from samviewer.camera import (
    CameraConfig,
    load_rgb_image,
    open_camera,
    parse_camera_source,
    read_rgb_frame,
    scan_camera_indices,
)
from samviewer.metrics import (
    AlignmentMetrics,
    BoundingBox,
    CornerAlignmentMetrics,
    Point,
    bbox_from_points,
    bbox_to_mapping,
    clamp_bbox,
    compute_alignment,
    compute_bbox_iou,
    compute_corner_alignment,
    scale_points,
    scale_bbox,
)
from samviewer.segmentation import OptionalSegmenter, Sam2Config, sam2_status
from samviewer.visualization import (
    REAL_COLOR,
    REF_COLOR,
    draw_bbox,
    draw_mask_outline,
    draw_points_polygon,
    overlay_bboxes,
    overlay_reference_image,
)


MAX_ANNOTATION_WIDTH = 640


def main() -> None:
    st.set_page_config(
        page_title="Camera Alignment Viewer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    st.title("Camera Alignment Viewer")

    config = _render_sidebar()
    _render_reference_loader()
    _render_camera_controls(config)

    live_frame = st.session_state.get("live_frame")
    real_frame = st.session_state.get("real_frame")
    reference = st.session_state.get("reference_image")

    st.divider()
    _render_live_panel(live_frame, reference)

    st.divider()
    _render_roi_panels(reference, real_frame)

    st.divider()
    _render_alignment_panel(reference, real_frame)

    _maybe_rerun_live_preview()


def _init_state() -> None:
    defaults = {
        "cap": None,
        "camera_running": False,
        "camera_source_text": "0",
        "camera_indices": [0],
        "live_frame": None,
        "real_frame": None,
        "reference_image": None,
        "reference_name": None,
        "reference_upload_key": None,
        "ref_roi": None,
        "real_roi": None,
        "ref_roi_enabled": False,
        "real_roi_enabled": False,
        "ref_mask": None,
        "real_mask": None,
        "ref_points": [],
        "real_points": [],
        "ref_last_bbox_event": None,
        "real_last_bbox_event": None,
        "ref_last_point_event": None,
        "real_last_point_event": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_sidebar() -> CameraConfig:
    with st.sidebar:
        st.header("Camera")

        if st.button("Scan camera indices", use_container_width=True):
            try:
                found = scan_camera_indices(5)
                st.session_state.camera_indices = found or [0]
            except Exception as exc:
                st.warning(f"Camera scan failed: {exc}")

        index_options = [str(index) for index in st.session_state.camera_indices]
        options = index_options + ["Custom path / URL"]
        source_choice = st.selectbox("Camera source", options, index=0)
        if source_choice == "Custom path / URL":
            source_text = st.text_input(
                "Custom camera source",
                value=st.session_state.camera_source_text,
                placeholder="/dev/video0, rtsp://..., video.mp4",
            )
        else:
            source_text = source_choice
        st.session_state.camera_source_text = source_text

        width = st.number_input("Capture width", min_value=0, value=1920, step=160)
        height = st.number_input("Capture height", min_value=0, value=1080, step=90)
        st.checkbox("Live preview", key="live_preview", value=True)
        st.slider("Preview FPS", min_value=1, max_value=15, value=5, key="preview_fps")

        st.header("Alignment")
        st.radio(
            "Guide mode",
            ["Scale only", "Scale + center"],
            index=1,
            horizontal=False,
            key="guide_mode",
        )
        st.checkbox("Overlay reference bbox on live frame", key="overlay_ref_live", value=True)
        st.checkbox("Normalize reference bbox to real frame size", key="normalize_bbox", value=True)

        return CameraConfig(
            source=parse_camera_source(source_text),
            width=width or None,
            height=height or None,
        )


def _render_reference_loader() -> None:
    with st.expander("Reference image", expanded=True):
        col_upload, col_path = st.columns([1, 1])
        with col_upload:
            uploaded = st.file_uploader(
                "Upload reference image",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
            )
            if uploaded is not None:
                upload_key = (uploaded.name, uploaded.size)
                if st.session_state.reference_upload_key != upload_key:
                    image = Image.open(uploaded).convert("RGB")
                    st.session_state.reference_image = np.asarray(image)
                    st.session_state.reference_name = uploaded.name
                    st.session_state.reference_upload_key = upload_key
                    _reset_annotation("ref")
        with col_path:
            path_text = st.text_input("Or load image path", placeholder="/path/to/reference.png")
            if st.button("Load reference path", use_container_width=True) and path_text:
                try:
                    st.session_state.reference_image = load_rgb_image(Path(path_text))
                    st.session_state.reference_name = str(Path(path_text).expanduser())
                    st.session_state.reference_upload_key = None
                    _reset_annotation("ref")
                except Exception as exc:
                    st.error(str(exc))

        reference = st.session_state.get("reference_image")
        if reference is not None:
            name = st.session_state.get("reference_name") or "reference"
            st.caption(f"{name} | {reference.shape[1]} x {reference.shape[0]} px")


def _render_camera_controls(config: CameraConfig) -> None:
    with st.expander("Camera stream", expanded=True):
        col_start, col_stop, col_capture = st.columns(3)
        with col_start:
            if st.button("Start camera", type="primary", use_container_width=True):
                _start_camera(config)
        with col_stop:
            if st.button("Stop camera", use_container_width=True):
                _stop_camera()
        with col_capture:
            if st.button("Capture current frame", use_container_width=True):
                _capture_current_frame(config)

        if st.session_state.camera_running:
            try:
                frame = read_rgb_frame(st.session_state.cap)
                st.session_state.live_frame = frame
            except Exception as exc:
                st.error(str(exc))
                _stop_camera()

        real_frame = st.session_state.get("real_frame")
        if real_frame is not None:
            st.caption(f"Captured real frame | {real_frame.shape[1]} x {real_frame.shape[0]} px")


def _render_live_panel(live_frame: np.ndarray | None, reference: np.ndarray | None) -> None:
    st.subheader("Live camera")
    if live_frame is None:
        st.info("Start the camera to show the live feed.")
        return

    frame = live_frame
    ref_bbox = st.session_state.get("ref_roi")
    real_bbox = st.session_state.get("real_roi")
    ref_points = st.session_state.get("ref_points") or []

    if st.session_state.get("overlay_ref_live") and reference is not None and ref_bbox is not None:
        scaled_ref = scale_bbox(ref_bbox, reference.shape, frame.shape)
        frame = overlay_bboxes(frame, scaled_ref, None)

    if st.session_state.get("overlay_ref_live") and reference is not None and len(ref_points) == 4:
        scaled_points = scale_points(ref_points, reference.shape, frame.shape)
        frame = draw_points_polygon(frame, scaled_points, REF_COLOR, "reference corners")

    if real_bbox is not None:
        frame = draw_bbox(frame, real_bbox, REAL_COLOR, "real ROI")

    st.image(frame, channels="RGB", use_container_width=True)

    if reference is not None and ref_bbox is not None and real_bbox is not None:
        live_ref_bbox = scale_bbox(ref_bbox, reference.shape, frame.shape)
        intersection_area, union_area, iou = compute_bbox_iou(live_ref_bbox, real_bbox)
        st.metric("Live bbox IoU", f"{iou:.3f}")
        st.caption(f"Intersection={intersection_area}px | union={union_area}px")


def _render_roi_panels(reference: np.ndarray | None, real_frame: np.ndarray | None) -> None:
    st.subheader("ROI selection")
    col_ref, col_real = st.columns(2)
    with col_ref:
        _roi_editor("Reference object ROI", reference, "ref")
    with col_real:
        _roi_editor("Real captured object ROI", real_frame, "real")

    with st.expander("Optional segmentation interface", expanded=False):
        segmenter = _render_segmentation_settings()
        col_ref_seg, col_real_seg = st.columns(2)
        with col_ref_seg:
            if st.button("Create reference mask from ROI", use_container_width=True):
                _create_mask_from_roi("ref", reference, segmenter)
        with col_real_seg:
            if st.button("Create real mask from ROI", use_container_width=True):
                _create_mask_from_roi("real", real_frame, segmenter)


def _render_alignment_panel(reference: np.ndarray | None, real_frame: np.ndarray | None) -> None:
    st.subheader("Alignment metrics")
    ref_bbox = st.session_state.get("ref_roi")
    real_bbox = st.session_state.get("real_roi")
    ref_points = st.session_state.get("ref_points") or []
    real_points = st.session_state.get("real_points") or []

    if reference is None or real_frame is None:
        st.info("Load a reference image and capture a real frame.")
        return
    if ref_bbox is None or real_bbox is None:
        st.info("Select ROI on both the reference image and the captured real frame.")
        return

    metric_ref_bbox = ref_bbox
    if st.session_state.get("normalize_bbox") and reference.shape[:2] != real_frame.shape[:2]:
        metric_ref_bbox = scale_bbox(ref_bbox, reference.shape, real_frame.shape)

    metrics = compute_alignment(metric_ref_bbox, real_bbox)
    _render_metric_status(metrics)
    _render_metric_table(metrics)

    metric_ref_points = ref_points
    if ref_points and st.session_state.get("normalize_bbox") and reference.shape[:2] != real_frame.shape[:2]:
        metric_ref_points = scale_points(ref_points, reference.shape, real_frame.shape)
    if len(metric_ref_points) == 4 and len(real_points) == 4:
        _render_corner_metrics(compute_corner_alignment(metric_ref_points, real_points))

    _render_comparison_view(
        reference,
        real_frame,
        ref_bbox,
        metric_ref_bbox,
        real_bbox,
        ref_points,
        metric_ref_points,
        real_points,
    )


def _render_metric_status(metrics: AlignmentMetrics) -> None:
    status = metrics.scale_status
    if status == "within_5":
        st.success(f"Scale alignment: within 5% ({metrics.max_size_error_pct:.2f}%)")
    elif status == "within_10":
        st.warning(f"Scale alignment: within 10% ({metrics.max_size_error_pct:.2f}%)")
    else:
        st.error(f"Scale alignment: outside 10% ({metrics.max_size_error_pct:.2f}%)")

    if metrics.iou >= 0.75:
        st.success(f"BBox IoU: {metrics.iou:.3f}")
    elif metrics.iou >= 0.5:
        st.warning(f"BBox IoU: {metrics.iou:.3f}")
    else:
        st.error(f"BBox IoU: {metrics.iou:.3f}")

    if st.session_state.get("guide_mode") == "Scale + center":
        st.caption(
            "Center offset: "
            f"dx={metrics.center_dx_px:.1f}px, "
            f"dy={metrics.center_dy_px:.1f}px, "
            f"distance={metrics.center_distance_px:.1f}px"
        )


def _render_metric_table(metrics: AlignmentMetrics) -> None:
    rows = [
        {
            "metric": "width",
            "reference_px": metrics.ref_bbox.width,
            "real_px": metrics.real_bbox.width,
            "signed_error_px": metrics.width_error_px,
            "abs_error_pct": f"{metrics.width_error_pct:.2f}%",
        },
        {
            "metric": "height",
            "reference_px": metrics.ref_bbox.height,
            "real_px": metrics.real_bbox.height,
            "signed_error_px": metrics.height_error_px,
            "abs_error_pct": f"{metrics.height_error_pct:.2f}%",
        },
        {
            "metric": "area",
            "reference_px": metrics.ref_bbox.area,
            "real_px": metrics.real_bbox.area,
            "signed_error_px": metrics.area_error_px,
            "abs_error_pct": f"{metrics.area_error_pct:.2f}%",
        },
        {
            "metric": "bbox_iou",
            "reference_px": f"intersection {metrics.intersection_area_px}",
            "real_px": f"union {metrics.union_area_px}",
            "signed_error_px": f"{metrics.iou:.3f}",
            "abs_error_pct": f"{metrics.iou * 100.0:.1f}%",
        },
        {
            "metric": "center_x",
            "reference_px": f"{metrics.ref_bbox.center[0]:.1f}",
            "real_px": f"{metrics.real_bbox.center[0]:.1f}",
            "signed_error_px": f"{metrics.center_dx_px:.1f}",
            "abs_error_pct": "",
        },
        {
            "metric": "center_y",
            "reference_px": f"{metrics.ref_bbox.center[1]:.1f}",
            "real_px": f"{metrics.real_bbox.center[1]:.1f}",
            "signed_error_px": f"{metrics.center_dy_px:.1f}",
            "abs_error_pct": "",
        },
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_corner_metrics(metrics: CornerAlignmentMetrics) -> None:
    st.markdown("**Corner alignment**")
    st.caption(
        "Mean corner offset: "
        f"{metrics.mean_distance_px:.1f}px | "
        f"max: {metrics.max_distance_px:.1f}px | "
        f"area error: {metrics.area_error_pct:.2f}% | "
        f"center dx={metrics.center_dx_px:.1f}px, dy={metrics.center_dy_px:.1f}px"
    )
    rows = []
    for index, (ref_point, real_point, offset, distance) in enumerate(
        zip(metrics.ref_points, metrics.real_points, metrics.offsets, metrics.distances_px),
        start=1,
    ):
        rows.append(
            {
                "corner": index,
                "ref_x": f"{ref_point[0]:.1f}",
                "ref_y": f"{ref_point[1]:.1f}",
                "real_x": f"{real_point[0]:.1f}",
                "real_y": f"{real_point[1]:.1f}",
                "dx_px": f"{offset[0]:.1f}",
                "dy_px": f"{offset[1]:.1f}",
                "distance_px": f"{distance:.1f}",
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_comparison_view(
    reference: np.ndarray,
    real_frame: np.ndarray,
    ref_bbox: BoundingBox,
    metric_ref_bbox: BoundingBox,
    real_bbox: BoundingBox,
    ref_points: list[Point],
    metric_ref_points: list[Point],
    real_points: list[Point],
) -> None:
    mode = st.radio("View mode", ["Side-by-side", "Overlay"], horizontal=True)
    if mode == "Overlay":
        alpha = st.slider("Reference alpha", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        st.image(
            draw_points_polygon(
                draw_points_polygon(
                    overlay_reference_image(reference, real_frame, alpha, ref_bbox, real_bbox),
                    metric_ref_points,
                    REF_COLOR,
                    "reference corners",
                ),
                real_points,
                REAL_COLOR,
                "real corners",
            ),
            channels="RGB",
            use_container_width=True,
        )
        return

    col_ref, col_real = st.columns(2)
    with col_ref:
        st.image(
            draw_points_polygon(
                draw_bbox(reference, ref_bbox, REF_COLOR, "reference"),
                ref_points,
                REF_COLOR,
                "reference corners",
            ),
            channels="RGB",
            caption="Reference image",
            use_container_width=True,
        )
    with col_real:
        st.image(
            draw_points_polygon(
                draw_points_polygon(
                    overlay_bboxes(real_frame, metric_ref_bbox, real_bbox),
                    metric_ref_points,
                    REF_COLOR,
                    "reference corners",
                ),
                real_points,
                REAL_COLOR,
                "real corners",
            ),
            channels="RGB",
            caption="Captured real frame",
            use_container_width=True,
        )


def _roi_editor(title: str, image: np.ndarray | None, prefix: str) -> None:
    st.markdown(f"**{title}**")
    if image is None:
        st.info("No image available.")
        return

    h, w = image.shape[:2]
    roi_key = f"{prefix}_roi"
    enabled_key = f"{prefix}_roi_enabled"
    enabled = st.checkbox("Enable ROI", key=enabled_key)

    if not enabled:
        st.image(image, channels="RGB", use_container_width=True)
        st.session_state[roi_key] = None
        return

    current = st.session_state.get(roi_key)
    if current is None:
        current = _centered_bbox(image)
        _store_roi(prefix, current)
    else:
        current = clamp_bbox(current, image.shape)
        _store_roi(prefix, current)

    tool = st.radio(
        "Annotation tool",
        ["Drag bbox", "Click 4 corners", "Numeric / OpenCV"],
        horizontal=True,
        key=f"{prefix}_annotation_tool",
    )

    if tool == "Drag bbox":
        _mouse_bbox_editor(image, prefix)
        return
    if tool == "Click 4 corners":
        _corner_point_editor(image, prefix)
        return

    _numeric_bbox_editor(title, image, prefix)


def _mouse_bbox_editor(image: np.ndarray, prefix: str) -> None:
    bbox = st.session_state.get(f"{prefix}_roi")
    annotated = _annotation_image(image, prefix, bbox=bbox)
    display, scale = _resize_for_annotation(annotated)
    event = streamlit_image_coordinates(
        display,
        width=display.shape[1],
        height=display.shape[0],
        click_and_drag=True,
        key=f"{prefix}_bbox_drag",
        cursor="crosshair",
    )

    if _is_new_event(prefix, "bbox", event) and event is not None:
        selected = _bbox_from_drag_event(event, scale, image.shape)
        if selected is not None:
            _store_roi(prefix, selected)
            st.rerun()

    st.caption("Drag over the object to set the bbox.")
    if bbox is not None:
        st.caption(str(bbox_to_mapping(bbox)))


def _corner_point_editor(image: np.ndarray, prefix: str) -> None:
    points = list(st.session_state.get(f"{prefix}_points") or [])
    bbox = bbox_from_points(points, image.shape) if points else st.session_state.get(f"{prefix}_roi")
    annotated = _annotation_image(image, prefix, bbox=bbox, points=points)
    display, scale = _resize_for_annotation(annotated)

    col_undo, col_clear = st.columns(2)
    with col_undo:
        if st.button("Undo last point", key=f"{prefix}_undo_point", use_container_width=True):
            points = points[:-1]
            _store_points(prefix, points, image.shape)
            st.rerun()
    with col_clear:
        if st.button("Clear points", key=f"{prefix}_clear_points", use_container_width=True):
            _store_points(prefix, [], image.shape)
            st.rerun()

    event = streamlit_image_coordinates(
        display,
        width=display.shape[1],
        height=display.shape[0],
        click_and_drag=False,
        key=f"{prefix}_corner_click",
        cursor="crosshair",
    )

    if _is_new_event(prefix, "point", event) and event is not None:
        point = _point_from_click_event(event, scale, image.shape)
        if point is not None:
            next_points = points + [point]
            if len(next_points) > 4:
                next_points = [point]
            _store_points(prefix, next_points, image.shape)
            st.rerun()

    st.caption(f"Click corners in order: top-left, top-right, bottom-right, bottom-left. {len(points)}/4 set.")
    if points:
        st.dataframe(
            [
                {"index": index + 1, "x": round(point[0], 1), "y": round(point[1], 1)}
                for index, point in enumerate(points)
            ],
            hide_index=True,
            use_container_width=True,
        )


def _numeric_bbox_editor(title: str, image: np.ndarray, prefix: str) -> None:
    h, w = image.shape[:2]
    roi_key = f"{prefix}_roi"
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Centered ROI", key=f"{prefix}_centered", use_container_width=True):
            _store_roi(prefix, _centered_bbox(image))
            st.rerun()
    with col_b:
        if st.button("Full image ROI", key=f"{prefix}_full", use_container_width=True):
            _store_roi(prefix, BoundingBox(0, 0, w, h))
            st.rerun()
    with col_c:
        if st.button("OpenCV selectROI", key=f"{prefix}_opencv", use_container_width=True):
            try:
                selected = _select_roi_with_opencv(image, title)
                if selected is not None:
                    _store_roi(prefix, clamp_bbox(selected, image.shape))
                    st.rerun()
            except Exception as exc:
                st.error(f"OpenCV ROI selector failed: {exc}")

    col_x, col_y = st.columns(2)
    with col_x:
        x = st.number_input(
            "x",
            min_value=0,
            max_value=max(0, w - 1),
            step=1,
            key=f"{prefix}_roi_x",
        )
        width_key = f"{prefix}_roi_width"
        st.session_state[width_key] = min(
            int(st.session_state[width_key]),
            max(1, w - int(x)),
        )
        width = st.number_input(
            "width",
            min_value=1,
            max_value=max(1, w - int(x)),
            step=1,
            key=width_key,
        )
    with col_y:
        y = st.number_input(
            "y",
            min_value=0,
            max_value=max(0, h - 1),
            step=1,
            key=f"{prefix}_roi_y",
        )
        height_key = f"{prefix}_roi_height"
        st.session_state[height_key] = min(
            int(st.session_state[height_key]),
            max(1, h - int(y)),
        )
        height = st.number_input(
            "height",
            min_value=1,
            max_value=max(1, h - int(y)),
            step=1,
            key=height_key,
        )

    bbox = clamp_bbox(BoundingBox(int(x), int(y), int(width), int(height)), image.shape)
    st.session_state[roi_key] = bbox
    st.image(
        draw_bbox(
            draw_mask_outline(
                image,
                st.session_state.get(f"{prefix}_mask"),
                REF_COLOR if prefix == "ref" else REAL_COLOR,
                "mask",
            ),
            bbox,
            REF_COLOR if prefix == "ref" else REAL_COLOR,
            "ROI",
        ),
        channels="RGB",
        use_container_width=True,
    )
    st.caption(str(bbox_to_mapping(bbox)))


def _annotation_image(
    image: np.ndarray,
    prefix: str,
    bbox: BoundingBox | None = None,
    points: list[Point] | None = None,
) -> np.ndarray:
    color = REF_COLOR if prefix == "ref" else REAL_COLOR
    out = draw_mask_outline(image, st.session_state.get(f"{prefix}_mask"), color, "mask")
    if bbox is not None:
        out = draw_bbox(out, bbox, color, "ROI")
    existing_points = points if points is not None else st.session_state.get(f"{prefix}_points")
    if existing_points:
        out = draw_points_polygon(out, existing_points, color, "corners", closed=len(existing_points) >= 4)
    return out


def _resize_for_annotation(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, MAX_ANNOTATION_WIDTH / max(1, w))
    display_w = max(1, int(round(w * scale)))
    display_h = max(1, int(round(h * scale)))
    if scale == 1.0:
        return image, scale
    return cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA), scale


def _is_new_event(prefix: str, event_type: str, event: dict | None) -> bool:
    if not event:
        return False
    event_time = event.get("unix_time")
    if event_time is None:
        return True
    key = f"{prefix}_last_{event_type}_event"
    if st.session_state.get(key) == event_time:
        return False
    st.session_state[key] = event_time
    return True


def _bbox_from_drag_event(
    event: dict,
    scale: float,
    image_shape: tuple[int, ...],
) -> BoundingBox | None:
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(event):
        return None

    display_w = max(1, int(event.get("width", image_shape[1])))
    display_h = max(1, int(event.get("height", image_shape[0])))
    x1 = min(max(0.0, float(event["x1"])), display_w - 1)
    y1 = min(max(0.0, float(event["y1"])), display_h - 1)
    x2 = min(max(0.0, float(event["x2"])), display_w - 1)
    y2 = min(max(0.0, float(event["y2"])), display_h - 1)
    if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
        return None

    inv = 1.0 / max(scale, 1e-9)
    x = min(x1, x2) * inv
    y = min(y1, y2) * inv
    width = abs(x2 - x1) * inv
    height = abs(y2 - y1) * inv
    return clamp_bbox(
        BoundingBox(round(x), round(y), max(1, round(width)), max(1, round(height))),
        image_shape,
    )


def _point_from_click_event(
    event: dict,
    scale: float,
    image_shape: tuple[int, ...],
) -> Point | None:
    if "x" not in event or "y" not in event:
        return None
    display_w = max(1, int(event.get("width", image_shape[1])))
    display_h = max(1, int(event.get("height", image_shape[0])))
    x = min(max(0.0, float(event["x"])), display_w - 1)
    y = min(max(0.0, float(event["y"])), display_h - 1)
    inv = 1.0 / max(scale, 1e-9)
    image_h, image_w = image_shape[:2]
    return (
        min(max(0.0, x * inv), image_w - 1),
        min(max(0.0, y * inv), image_h - 1),
    )


def _render_segmentation_settings() -> OptionalSegmenter:
    backend = st.selectbox(
        "Segmentation backend",
        ["bbox", "sam2"],
        format_func=lambda value: "BBox mask fallback" if value == "bbox" else "SAM2",
    )

    if backend == "bbox":
        st.caption("Uses the selected ROI as a rectangular mask. No extra dependencies needed.")
        return _get_segmenter("bbox", None, None, None, "auto", True)

    status = sam2_status()
    if status.available:
        st.success(status.message)
    else:
        st.warning(status.message)

    model_source = st.radio(
        "SAM2 model source",
        ["Hugging Face model id", "Local config + checkpoint"],
        horizontal=True,
    )
    device = st.selectbox("Device", ["auto", "cuda", "cpu"])
    multimask_output = st.checkbox("Use multimask and choose best score", value=True)

    model_id = None
    model_cfg = None
    checkpoint = None
    if model_source == "Hugging Face model id":
        model_id = st.text_input("Model id", value="facebook/sam2.1-hiera-tiny")
        st.caption("Requires `huggingface_hub` in addition to the SAM2 package.")
    else:
        model_cfg = st.text_input(
            "Model config",
            value="configs/sam2.1/sam2.1_hiera_t.yaml",
        )
        checkpoint = st.text_input(
            "Checkpoint path",
            placeholder="/path/to/sam2.1_hiera_tiny.pt",
        )

    return _get_segmenter(
        backend,
        model_id or None,
        model_cfg or None,
        checkpoint or None,
        device,
        multimask_output,
    )


@st.cache_resource(show_spinner=False)
def _get_segmenter(
    backend: str,
    model_id: str | None,
    model_cfg: str | None,
    checkpoint: str | None,
    device: str,
    multimask_output: bool,
) -> OptionalSegmenter:
    config = Sam2Config(
        model_id=model_id,
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        device=device,
        multimask_output=multimask_output,
    )
    return OptionalSegmenter(backend=backend, sam2_config=config)


def _centered_bbox(image: np.ndarray) -> BoundingBox:
    h, w = image.shape[:2]
    width = max(1, round(w * 0.5))
    height = max(1, round(h * 0.5))
    return BoundingBox(
        x=max(0, round((w - width) / 2)),
        y=max(0, round((h - height) / 2)),
        width=width,
        height=height,
    )


def _store_roi(prefix: str, bbox: BoundingBox) -> None:
    st.session_state[f"{prefix}_roi"] = bbox
    st.session_state[f"{prefix}_roi_x"] = bbox.x
    st.session_state[f"{prefix}_roi_y"] = bbox.y
    st.session_state[f"{prefix}_roi_width"] = bbox.width
    st.session_state[f"{prefix}_roi_height"] = bbox.height


def _store_points(prefix: str, points: list[Point], image_shape: tuple[int, ...]) -> None:
    st.session_state[f"{prefix}_points"] = points
    bbox = bbox_from_points(points, image_shape)
    if bbox is not None:
        _store_roi(prefix, bbox)


def _reset_annotation(prefix: str) -> None:
    st.session_state[f"{prefix}_roi"] = None
    st.session_state[f"{prefix}_mask"] = None
    st.session_state[f"{prefix}_points"] = []
    st.session_state[f"{prefix}_last_bbox_event"] = None
    st.session_state[f"{prefix}_last_point_event"] = None


def _select_roi_with_opencv(image: np.ndarray, title: str) -> BoundingBox | None:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    selected = cv2.selectROI(title, image_bgr, showCrosshair=True, fromCenter=False)
    try:
        cv2.destroyWindow(title)
    except cv2.error:
        pass

    x, y, width, height = selected
    if width <= 0 or height <= 0:
        return None
    return BoundingBox(int(x), int(y), int(width), int(height))


def _create_mask_from_roi(
    prefix: str,
    image: np.ndarray | None,
    segmenter: OptionalSegmenter,
) -> None:
    if image is None:
        st.warning("No image available for segmentation.")
        return
    bbox = st.session_state.get(f"{prefix}_roi")
    if bbox is None:
        st.warning("Select an ROI first.")
        return
    try:
        with st.spinner(f"Running {segmenter.backend} segmentation..."):
            result = segmenter.segment_from_bbox(image, bbox)
    except Exception as exc:
        st.error(str(exc))
        return

    st.session_state[f"{prefix}_mask"] = result.mask
    _store_roi(prefix, result.bbox)
    score = f" score={result.score:.3f}" if result.score is not None else ""
    st.success(f"Created {prefix} mask with {result.method} backend.{score}")
    st.rerun()


def _start_camera(config: CameraConfig) -> None:
    _stop_camera()
    try:
        st.session_state.cap = open_camera(config)
        st.session_state.camera_running = True
        st.success(f"Camera started: {config.source!r}")
    except Exception as exc:
        st.session_state.camera_running = False
        st.session_state.cap = None
        st.error(str(exc))


def _stop_camera() -> None:
    cap = st.session_state.get("cap")
    if cap is not None:
        cap.release()
    st.session_state.cap = None
    st.session_state.camera_running = False


def _capture_current_frame(config: CameraConfig) -> None:
    live_frame = st.session_state.get("live_frame")
    if live_frame is not None:
        st.session_state.real_frame = live_frame.copy()
        _reset_annotation("real")
        st.success("Captured current live frame.")
        return

    try:
        cap = open_camera(config)
        try:
            st.session_state.real_frame = read_rgb_frame(cap)
            _reset_annotation("real")
            st.success("Captured one frame from camera.")
        finally:
            cap.release()
    except Exception as exc:
        st.error(str(exc))


def _maybe_rerun_live_preview() -> None:
    if not st.session_state.get("camera_running"):
        return
    if not st.session_state.get("live_preview"):
        return
    fps = max(1, int(st.session_state.get("preview_fps", 5)))
    time.sleep(1.0 / fps)
    st.rerun()
