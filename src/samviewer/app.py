from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

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
    bbox_to_mapping,
    clamp_bbox,
    compute_alignment,
    scale_bbox,
)
from samviewer.segmentation import OptionalSegmenter
from samviewer.visualization import (
    REAL_COLOR,
    REF_COLOR,
    draw_bbox,
    overlay_bboxes,
    overlay_reference_image,
)


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
        "ref_roi": None,
        "real_roi": None,
        "ref_roi_enabled": False,
        "real_roi_enabled": False,
        "ref_mask": None,
        "real_mask": None,
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

        width = st.number_input("Capture width", min_value=0, value=1280, step=160)
        height = st.number_input("Capture height", min_value=0, value=720, step=90)
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
                image = Image.open(uploaded).convert("RGB")
                st.session_state.reference_image = np.asarray(image)
                st.session_state.reference_name = uploaded.name
        with col_path:
            path_text = st.text_input("Or load image path", placeholder="/path/to/reference.png")
            if st.button("Load reference path", use_container_width=True) and path_text:
                try:
                    st.session_state.reference_image = load_rgb_image(Path(path_text))
                    st.session_state.reference_name = str(Path(path_text).expanduser())
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

    if st.session_state.get("overlay_ref_live") and reference is not None and ref_bbox is not None:
        scaled_ref = scale_bbox(ref_bbox, reference.shape, frame.shape)
        frame = overlay_bboxes(frame, scaled_ref, None)

    if real_bbox is not None:
        frame = draw_bbox(frame, real_bbox, REAL_COLOR, "real ROI")

    st.image(frame, channels="RGB", use_container_width=True)


def _render_roi_panels(reference: np.ndarray | None, real_frame: np.ndarray | None) -> None:
    st.subheader("ROI selection")
    col_ref, col_real = st.columns(2)
    with col_ref:
        _roi_editor("Reference object ROI", reference, "ref")
    with col_real:
        _roi_editor("Real captured object ROI", real_frame, "real")

    with st.expander("Optional segmentation interface", expanded=False):
        segmenter = OptionalSegmenter("bbox")
        st.caption(f"Fallback backend: {'available' if segmenter.available else 'unavailable'}")
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
    _render_comparison_view(reference, real_frame, ref_bbox, metric_ref_bbox, real_bbox)


def _render_metric_status(metrics: AlignmentMetrics) -> None:
    status = metrics.scale_status
    if status == "within_5":
        st.success(f"Scale alignment: within 5% ({metrics.max_size_error_pct:.2f}%)")
    elif status == "within_10":
        st.warning(f"Scale alignment: within 10% ({metrics.max_size_error_pct:.2f}%)")
    else:
        st.error(f"Scale alignment: outside 10% ({metrics.max_size_error_pct:.2f}%)")

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


def _render_comparison_view(
    reference: np.ndarray,
    real_frame: np.ndarray,
    ref_bbox: BoundingBox,
    metric_ref_bbox: BoundingBox,
    real_bbox: BoundingBox,
) -> None:
    mode = st.radio("View mode", ["Side-by-side", "Overlay"], horizontal=True)
    if mode == "Overlay":
        alpha = st.slider("Reference alpha", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        st.image(
            overlay_reference_image(reference, real_frame, alpha, ref_bbox, real_bbox),
            channels="RGB",
            use_container_width=True,
        )
        return

    col_ref, col_real = st.columns(2)
    with col_ref:
        st.image(
            draw_bbox(reference, ref_bbox, REF_COLOR, "reference"),
            channels="RGB",
            caption="Reference image",
            use_container_width=True,
        )
    with col_real:
        st.image(
            overlay_bboxes(real_frame, metric_ref_bbox, real_bbox),
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
        draw_bbox(image, bbox, REF_COLOR if prefix == "ref" else REAL_COLOR, "ROI"),
        channels="RGB",
        use_container_width=True,
    )
    st.caption(str(bbox_to_mapping(bbox)))


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
    result = segmenter.segment_from_bbox(image, bbox)
    st.session_state[f"{prefix}_mask"] = result.mask
    st.success(f"Created {prefix} mask with {result.method} backend.")


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
        st.success("Captured current live frame.")
        return

    try:
        cap = open_camera(config)
        try:
            st.session_state.real_frame = read_rgb_frame(cap)
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
