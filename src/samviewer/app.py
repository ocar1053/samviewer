from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from samviewer.camera import (
    CameraConfig,
    DEFAULT_CAPTURE_HEIGHT,
    DEFAULT_CAPTURE_WIDTH,
    load_rgb_image,
    open_camera,
    parse_camera_source,
    read_rgb_frame,
    scan_camera_indices,
)
from samviewer.metrics import (
    BoundingBox,
    Point,
    bbox_from_points,
    bbox_to_mapping,
    clamp_bbox,
    compute_bbox_iou,
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
)


MAX_ANNOTATION_WIDTH = 1280
MAX_LIVE_ANNOTATION_WIDTH = 1280
REFERENCE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_REFERENCE_FOLDER = "reference_images"
DEFAULT_COMPARISON_FOLDER = "comparison_saves"
DEFAULT_PREVIEW_FPS = 10
DEFAULT_CAMERA_BACKEND_LABEL = "Windows DirectShow" if sys.platform.startswith("win") else "Auto"
DEFAULT_ANNOTATION_TOOL = "Click 4 corners"
ANNOTATION_DEFAULTS_VERSION = 3


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


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
    reference = st.session_state.get("reference_image")

    st.divider()
    _render_reference_roi_panel(reference, live_frame)

    st.divider()
    _render_live_panel(live_frame, reference)

    _maybe_rerun_live_preview()


def _init_state() -> None:
    defaults = {
        "cap": None,
        "camera_running": False,
        "camera_source_text": "0",
        "camera_indices": [0],
        "camera_backend_label": DEFAULT_CAMERA_BACKEND_LABEL,
        "live_frame": None,
        "real_frame": None,
        "reference_image": None,
        "reference_name": None,
        "reference_source_key": None,
        "reference_folder_text": DEFAULT_REFERENCE_FOLDER,
        "reference_folder_recursive": False,
        "ref_roi": None,
        "real_roi": None,
        "ref_roi_shape": None,
        "real_roi_shape": None,
        "ref_roi_enabled": True,
        "real_roi_enabled": False,
        "ref_mask": None,
        "real_mask": None,
        "ref_points": [],
        "real_points": [],
        "ref_points_shape": None,
        "real_points_shape": None,
        "live_roi_enabled": False,
        "live_annotation_tool": DEFAULT_ANNOTATION_TOOL,
        "freeze_live_while_annotating": True,
        "live_last_bbox_event": None,
        "live_last_point_event": None,
        "ref_annotation_tool": DEFAULT_ANNOTATION_TOOL,
        "real_annotation_tool": DEFAULT_ANNOTATION_TOOL,
        "annotation_defaults_version": 0,
        "ref_last_bbox_event": None,
        "real_last_bbox_event": None,
        "ref_last_point_event": None,
        "real_last_point_event": None,
        "last_comparison_image_path": None,
        "last_comparison_data_path": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if st.session_state.annotation_defaults_version < ANNOTATION_DEFAULTS_VERSION:
        st.session_state.live_annotation_tool = DEFAULT_ANNOTATION_TOOL
        st.session_state.ref_annotation_tool = DEFAULT_ANNOTATION_TOOL
        st.session_state.real_annotation_tool = DEFAULT_ANNOTATION_TOOL
        st.session_state.ref_roi_enabled = True
        st.session_state.freeze_live_while_annotating = True
        st.session_state.annotation_defaults_version = ANNOTATION_DEFAULTS_VERSION


def _render_sidebar() -> CameraConfig:
    with st.sidebar:
        st.header("Camera")

        backend_options = _camera_backend_options()
        backend_labels = list(backend_options)
        backend_label = st.session_state.get("camera_backend_label", DEFAULT_CAMERA_BACKEND_LABEL)
        if backend_label not in backend_options:
            backend_label = "Auto"
        backend_choice = st.selectbox(
            "Camera backend",
            backend_labels,
            index=backend_labels.index(backend_label),
            help="On Windows, DirectShow often fails faster and avoids Media Foundation camera stalls.",
        )
        st.session_state.camera_backend_label = backend_choice
        backend = backend_options[backend_choice]

        if st.button("Scan camera indices", use_container_width=True):
            try:
                with st.spinner("Scanning cameras..."):
                    found = scan_camera_indices(5, backend=backend)
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

        width = st.number_input("Capture width", min_value=0, value=DEFAULT_CAPTURE_WIDTH, step=160)
        height = st.number_input("Capture height", min_value=0, value=DEFAULT_CAPTURE_HEIGHT, step=90)
        st.checkbox("Live preview", key="live_preview", value=True)
        st.slider("Preview FPS", min_value=1, max_value=15, value=DEFAULT_PREVIEW_FPS, key="preview_fps")

        st.header("Live overlay")
        st.checkbox("Overlay reference bbox on live frame", key="overlay_ref_live", value=True)

        return CameraConfig(
            source=parse_camera_source(source_text),
            width=width or None,
            height=height or None,
            backend=backend,
        )


def _camera_backend_options() -> dict[str, int | None]:
    options: dict[str, int | None] = {"Auto": None}
    if sys.platform.startswith("win"):
        options["Windows DirectShow"] = cv2.CAP_DSHOW
        options["Windows Media Foundation"] = cv2.CAP_MSMF
    return options


def _render_reference_loader() -> None:
    with st.expander("Reference image", expanded=True):
        col_folder, col_upload, col_path = st.columns([1.2, 1, 1])
        with col_folder:
            _render_reference_folder_picker()
        with col_upload:
            uploaded = st.file_uploader(
                "Upload reference image",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
            )
            if uploaded is not None:
                source_key = ("upload", uploaded.name, uploaded.size)
                if st.session_state.reference_source_key != source_key:
                    with Image.open(uploaded) as image:
                        _set_reference_image(np.asarray(image.convert("RGB")), uploaded.name, source_key)
        with col_path:
            path_text = st.text_input("Or load image path", placeholder="/path/to/reference.png")
            if st.button("Load reference path", use_container_width=True) and path_text:
                try:
                    _load_reference_from_path(Path(path_text))
                except Exception as exc:
                    st.error(str(exc))

        reference = st.session_state.get("reference_image")
        if reference is not None:
            name = st.session_state.get("reference_name") or "reference"
            st.caption(f"{name} | {reference.shape[1]} x {reference.shape[0]} px")


def _render_reference_folder_picker() -> None:
    folder_text = st.text_input(
        "Reference folder",
        key="reference_folder_text",
        help="Server-side folder to scan for reusable reference images.",
    )
    recursive = st.checkbox("Include subfolders", key="reference_folder_recursive")
    if not folder_text.strip():
        st.info("Enter a folder path to list reusable reference images.")
        return

    folder = _resolve_user_path(folder_text)
    if not folder.exists():
        st.info(f"Folder not found: {folder}")
        return
    if not folder.is_dir():
        st.warning(f"Not a folder: {folder}")
        return

    image_paths = _list_reference_images(folder, recursive)
    if not image_paths:
        st.info("No supported images found in this folder.")
        return

    options = [""] + [str(path) for path in image_paths]
    selection_key = "reference_folder_selection"
    if st.session_state.get(selection_key) not in options:
        st.session_state[selection_key] = ""

    selected_value = st.selectbox(
        "Choose reference image",
        options,
        format_func=lambda value: _format_reference_choice(value, folder),
        key=selection_key,
    )
    if not selected_value:
        return

    try:
        _load_reference_from_path(Path(selected_value))
    except Exception as exc:
        st.error(str(exc))


def _resolve_user_path(path_text: str) -> Path:
    path = Path(path_text.strip()).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _list_reference_images(folder: Path, recursive: bool) -> list[Path]:
    paths = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (
            path
            for path in paths
            if path.is_file() and path.suffix.lower() in REFERENCE_IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def _format_reference_choice(value: str, folder: Path) -> str:
    if not value:
        return "Select an image..."
    path = Path(value)
    try:
        return str(path.relative_to(folder))
    except ValueError:
        return path.name


def _load_reference_from_path(path: Path) -> None:
    resolved = _resolve_user_path(str(path))
    source_key = _reference_path_source_key(resolved)
    if st.session_state.get("reference_source_key") == source_key:
        return
    _set_reference_image(load_rgb_image(resolved), str(resolved), source_key)


def _reference_path_source_key(path: Path) -> tuple[str, str, int, int]:
    resolved = path.resolve(strict=False)
    stat = resolved.stat()
    return ("path", str(resolved), stat.st_mtime_ns, stat.st_size)


def _set_reference_image(
    image: np.ndarray,
    name: str,
    source_key: tuple[str, str, int] | tuple[str, str, int, int],
) -> None:
    st.session_state.reference_image = image
    st.session_state.reference_name = name
    st.session_state.reference_source_key = source_key
    _reset_annotation("ref")


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

        if st.session_state.camera_running and not _should_pause_live_refresh_for_annotation():
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

    ref_bbox = _roi_for_shape("ref", live_frame.shape) if reference is not None else None
    real_bbox = _roi_for_shape("real", live_frame.shape)
    ref_points = _points_for_shape("ref", live_frame.shape) if reference is not None else []
    real_points = _points_for_shape("real", live_frame.shape)
    st.caption(f"Reference corners: {len(ref_points)}/4 | Live real corners: {len(real_points)}/4")
    enable_live_roi = st.checkbox(
        "Enable live real ROI",
        key="live_roi_enabled",
        help="Draw a bbox or click four real-object corners directly on the live camera preview.",
    )
    live_tool = None
    if enable_live_roi:
        live_tool = st.radio(
            "Live annotation tool",
            ["Drag bbox", "Click 4 corners"],
            index=1,
            horizontal=True,
            key="live_annotation_tool",
        )
        st.checkbox(
            "Pause live until 4 corners are set",
            key="freeze_live_while_annotating",
            help="Keeps the image still while selecting corners, then live preview resumes after all 4 points are set.",
        )

    _render_live_match_status(reference, live_frame, ref_bbox, real_bbox, enable_live_roi)
    frame = _live_overlay_image(live_frame, reference, ref_bbox, real_bbox, ref_points, real_points)

    if enable_live_roi and live_tool == "Click 4 corners":
        _live_corner_point_editor(frame, live_frame.shape)
    elif enable_live_roi:
        _live_bbox_editor(frame, live_frame.shape)
    else:
        st.image(frame, channels="RGB", use_container_width=True)

    save_frame = _live_overlay_image(
        live_frame,
        reference,
        ref_bbox,
        real_bbox,
        ref_points,
        real_points,
        force_reference_overlay=True,
    )
    _render_comparison_save_controls(save_frame, ref_bbox, real_bbox, ref_points, real_points)


def _render_live_match_status(
    reference: np.ndarray | None,
    live_frame: np.ndarray,
    ref_bbox: BoundingBox | None,
    real_bbox: BoundingBox | None,
    enable_live_roi: bool,
) -> None:
    col_score, col_intersection, col_union = st.columns(3)

    if reference is None:
        col_score.metric("Live match percentage", "N/A")
        col_intersection.metric("Intersection", "N/A")
        col_union.metric("Union", "N/A")
        st.caption("Missing: reference image.")
        return
    if ref_bbox is None:
        col_score.metric("Live match percentage", "N/A")
        col_intersection.metric("Intersection", "N/A")
        col_union.metric("Union", "N/A")
        st.caption("Missing: reference ROI. Enable ROI in Reference object ROI and click 4 corners.")
        return
    if real_bbox is None:
        col_score.metric("Live match percentage", "N/A")
        col_intersection.metric("Intersection", "N/A")
        col_union.metric("Union", "N/A")
        if enable_live_roi:
            st.caption("Missing: live real ROI. Click 4 live corners to calculate the percentage.")
        else:
            st.caption("Missing: live real ROI. Enable live real ROI first.")
        return

    intersection_area, union_area, iou = compute_bbox_iou(ref_bbox, real_bbox)
    overlap_percent = _format_percent(iou)
    col_score.metric("Live match percentage", overlap_percent)
    col_intersection.metric("Intersection", f"{intersection_area}px")
    col_union.metric("Union", f"{union_area}px")


def _live_overlay_image(
    live_frame: np.ndarray,
    reference: np.ndarray | None,
    ref_bbox: BoundingBox | None,
    real_bbox: BoundingBox | None,
    ref_points: list[Point],
    real_points: list[Point],
    force_reference_overlay: bool = False,
) -> np.ndarray:
    frame = live_frame

    show_reference_overlay = (
        (force_reference_overlay or st.session_state.get("overlay_ref_live"))
        and ref_bbox is not None
        and (len(ref_points) == 4 or (force_reference_overlay and not ref_points))
    )

    if show_reference_overlay:
        frame = overlay_bboxes(frame, ref_bbox, None)

    if show_reference_overlay:
        frame = draw_points_polygon(frame, ref_points, REF_COLOR, "reference corners")

    if real_bbox is not None:
        frame = draw_bbox(frame, real_bbox, REAL_COLOR, "real ROI")

    if real_points:
        frame = draw_points_polygon(
            frame,
            real_points,
            REAL_COLOR,
            "real corners",
            closed=len(real_points) >= 4,
        )

    return frame


def _render_comparison_save_controls(
    frame: np.ndarray,
    ref_bbox: BoundingBox | None,
    real_bbox: BoundingBox | None,
    ref_points: list[Point],
    real_points: list[Point],
) -> None:
    if not _comparison_is_complete(ref_bbox, real_bbox, ref_points, real_points):
        return

    if st.button("Save comparison image + lines", use_container_width=True):
        try:
            image_path, data_path = _save_comparison(frame, ref_bbox, real_bbox, ref_points, real_points)
            st.session_state.last_comparison_image_path = str(image_path)
            st.session_state.last_comparison_data_path = str(data_path)
            st.success(f"Saved comparison image: {image_path}")
            st.caption(f"Line data: {data_path}")
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    last_image = st.session_state.get("last_comparison_image_path")
    last_data = st.session_state.get("last_comparison_data_path")
    if last_image and last_data:
        st.caption(f"Last saved: {last_image}")


def _save_comparison(
    frame: np.ndarray,
    ref_bbox: BoundingBox,
    real_bbox: BoundingBox,
    ref_points: list[Point],
    real_points: list[Point],
) -> tuple[Path, Path]:
    output_dir = Path(DEFAULT_COMPARISON_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{time.time_ns() % 1_000_000_000:09d}"
    stem = f"comparison_{timestamp}_{suffix}"
    image_path = output_dir / f"{stem}.png"
    data_path = output_dir / f"{stem}.json"

    ok = cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not save comparison image: {image_path}")

    intersection_area, union_area, iou = compute_bbox_iou(ref_bbox, real_bbox)
    data = {
        "image": str(image_path),
        "reference_bbox": bbox_to_mapping(ref_bbox),
        "live_bbox": bbox_to_mapping(real_bbox),
        "reference_points": _points_to_mappings(ref_points),
        "live_points": _points_to_mappings(real_points),
        "intersection_area_px": intersection_area,
        "union_area_px": union_area,
        "match_iou": iou,
        "match_percent": iou * 100.0,
    }
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return image_path, data_path


def _comparison_is_complete(
    ref_bbox: BoundingBox | None,
    real_bbox: BoundingBox | None,
    ref_points: list[Point],
    real_points: list[Point],
) -> bool:
    ref_ready = ref_bbox is not None and (not ref_points or len(ref_points) == 4)
    real_ready = real_bbox is not None and (not real_points or len(real_points) == 4)
    return ref_ready and real_ready


def _points_to_mappings(points: list[Point]) -> list[dict[str, float]]:
    return [{"x": point[0], "y": point[1]} for point in points]


def _live_bbox_editor(frame: np.ndarray, live_shape: tuple[int, ...]) -> None:
    display, scale = _resize_for_annotation(frame, MAX_LIVE_ANNOTATION_WIDTH)
    event = streamlit_image_coordinates(
        display,
        width=display.shape[1],
        height=display.shape[0],
        click_and_drag=True,
        key="live_bbox_drag",
        cursor="crosshair",
    )

    if _is_new_event("live", "bbox", event) and event is not None:
        selected = _bbox_from_drag_event(event, scale, live_shape)
        if selected is not None:
            _store_roi("real", selected, live_shape)
            _store_points_only("real", [], live_shape)
            st.session_state.real_roi_enabled = True
            st.rerun()

    col_hint, col_clear = st.columns([3, 1])
    with col_hint:
        st.caption("Drag over the live object to set/update the green real ROI.")
    with col_clear:
        if st.button("Clear live ROI", use_container_width=True):
            _clear_live_real_annotation()
            st.rerun()


def _live_corner_point_editor(frame: np.ndarray, live_shape: tuple[int, ...]) -> None:
    points = _points_for_shape("real", live_shape)
    display, scale = _resize_for_annotation(frame, MAX_LIVE_ANNOTATION_WIDTH)

    col_undo, col_clear = st.columns(2)
    with col_undo:
        if st.button("Undo last live point", use_container_width=True):
            _store_live_real_points(points[:-1], live_shape)
            st.rerun()
    with col_clear:
        if st.button("Clear live points", use_container_width=True):
            _clear_live_real_annotation()
            st.rerun()

    event = streamlit_image_coordinates(
        display,
        width=display.shape[1],
        height=display.shape[0],
        click_and_drag=False,
        key="live_corner_click",
        cursor="crosshair",
    )

    if _is_new_event("live", "point", event) and event is not None:
        point = _point_from_click_event(event, scale, live_shape)
        if point is not None:
            next_points = points + [point]
            if len(next_points) > 4:
                next_points = [point]
            _store_live_real_points(next_points, live_shape)
            st.session_state.real_roi_enabled = True
            st.rerun()

    st.caption(
        "Click live corners in order: top-left, top-right, bottom-right, bottom-left. "
        f"{len(points)}/4 set."
    )
    if points:
        st.dataframe(
            [
                {"index": index + 1, "x": round(point[0], 1), "y": round(point[1], 1)}
                for index, point in enumerate(points)
            ],
            hide_index=True,
            use_container_width=True,
        )


def _render_reference_roi_panel(
    reference: np.ndarray | None,
    live_frame: np.ndarray | None,
) -> None:
    st.subheader("Reference ROI")
    reference_for_roi = _reference_image_for_roi(reference, live_frame)
    if reference is not None and live_frame is None:
        st.info("Start the camera first. The reference image will be resized to the live camera size before ROI selection.")
        st.image(reference, channels="RGB", use_container_width=True)
        return

    if reference_for_roi is not None and live_frame is not None:
        _reset_ref_annotation_if_shape_changed(reference_for_roi.shape)
        st.caption(
            "Reference image resized for ROI: "
            f"{reference_for_roi.shape[1]} x {reference_for_roi.shape[0]} px "
            "(same size as live camera)."
        )

    _roi_editor("Reference object ROI", reference_for_roi, "ref")

    with st.expander("Optional segmentation interface", expanded=False):
        segmenter = _render_segmentation_settings()
        if st.button("Create reference mask from ROI", use_container_width=True):
            _create_mask_from_roi("ref", reference_for_roi, segmenter)


def _reference_image_for_roi(
    reference: np.ndarray | None,
    live_frame: np.ndarray | None,
) -> np.ndarray | None:
    if reference is None:
        return None
    if live_frame is None:
        return reference
    target_h, target_w = live_frame.shape[:2]
    if reference.shape[:2] == (target_h, target_w):
        return reference
    interpolation = cv2.INTER_AREA if reference.shape[0] > target_h or reference.shape[1] > target_w else cv2.INTER_LINEAR
    return cv2.resize(reference, (target_w, target_h), interpolation=interpolation)


def _reset_ref_annotation_if_shape_changed(image_shape: tuple[int, ...]) -> None:
    for key in ("ref_roi_shape", "ref_points_shape"):
        source_shape = st.session_state.get(key)
        if source_shape is not None and source_shape[:2] != image_shape[:2]:
            _reset_annotation("ref")
            return


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
        st.session_state[f"{prefix}_roi_shape"] = None
        return

    current = _roi_for_shape(prefix, image.shape)
    if current is None:
        current = _centered_bbox(image)
        _store_roi(prefix, current, image.shape)
    else:
        current = clamp_bbox(current, image.shape)
        _store_roi(prefix, current, image.shape)

    tool = st.radio(
        "Annotation tool",
        ["Drag bbox", "Click 4 corners", "Numeric / OpenCV"],
        index=1,
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
            _store_roi(prefix, selected, image.shape)
            st.rerun()

    st.caption("Drag over the object to set the bbox.")
    if bbox is not None:
        st.caption(str(bbox_to_mapping(bbox)))


def _corner_point_editor(image: np.ndarray, prefix: str) -> None:
    points = _points_for_shape(prefix, image.shape)
    bbox = bbox_from_points(points, image.shape) if points else _roi_for_shape(prefix, image.shape)
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
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Centered ROI", key=f"{prefix}_centered", use_container_width=True):
            _store_roi(prefix, _centered_bbox(image), image.shape)
            st.rerun()
    with col_b:
        if st.button("Full image ROI", key=f"{prefix}_full", use_container_width=True):
            _store_roi(prefix, BoundingBox(0, 0, w, h), image.shape)
            st.rerun()
    with col_c:
        if st.button("OpenCV selectROI", key=f"{prefix}_opencv", use_container_width=True):
            try:
                selected = _select_roi_with_opencv(image, title)
                if selected is not None:
                    _store_roi(prefix, clamp_bbox(selected, image.shape), image.shape)
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
    _store_roi(prefix, bbox, image.shape)
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


def _resize_for_annotation(
    image: np.ndarray,
    max_width: int = MAX_ANNOTATION_WIDTH,
) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_width / max(1, w))
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


def _roi_for_shape(prefix: str, image_shape: tuple[int, ...]) -> BoundingBox | None:
    bbox = st.session_state.get(f"{prefix}_roi")
    if bbox is None:
        return None

    source_shape = st.session_state.get(f"{prefix}_roi_shape")
    if source_shape is not None and source_shape[:2] != image_shape[:2]:
        return scale_bbox(bbox, source_shape, image_shape)
    return clamp_bbox(bbox, image_shape)


def _points_for_shape(prefix: str, image_shape: tuple[int, ...]) -> list[Point]:
    points = list(st.session_state.get(f"{prefix}_points") or [])
    if not points:
        return []

    source_shape = st.session_state.get(f"{prefix}_points_shape")
    if source_shape is not None and source_shape[:2] != image_shape[:2]:
        return scale_points(points, source_shape, image_shape)
    return points


def _store_roi(prefix: str, bbox: BoundingBox, image_shape: tuple[int, ...] | None = None) -> None:
    st.session_state[f"{prefix}_roi"] = bbox
    if image_shape is not None:
        st.session_state[f"{prefix}_roi_shape"] = image_shape
    st.session_state[f"{prefix}_roi_x"] = bbox.x
    st.session_state[f"{prefix}_roi_y"] = bbox.y
    st.session_state[f"{prefix}_roi_width"] = bbox.width
    st.session_state[f"{prefix}_roi_height"] = bbox.height


def _store_points_only(
    prefix: str,
    points: list[Point],
    image_shape: tuple[int, ...] | None,
) -> None:
    st.session_state[f"{prefix}_points"] = points
    st.session_state[f"{prefix}_points_shape"] = image_shape if points else None


def _store_points(prefix: str, points: list[Point], image_shape: tuple[int, ...]) -> None:
    _store_points_only(prefix, points, image_shape)
    bbox = bbox_from_points(points, image_shape)
    if bbox is not None:
        _store_roi(prefix, bbox, image_shape)


def _store_live_real_points(points: list[Point], live_shape: tuple[int, ...]) -> None:
    _store_points_only("real", points, live_shape)
    if len(points) == 4:
        bbox = bbox_from_points(points, live_shape)
        if bbox is not None:
            _store_roi("real", bbox, live_shape)
        return

    st.session_state.real_roi = None
    st.session_state.real_roi_shape = None
    st.session_state.real_roi_enabled = bool(points)


def _clear_live_real_annotation() -> None:
    st.session_state.real_roi = None
    st.session_state.real_roi_shape = None
    st.session_state.real_roi_enabled = False
    _store_points_only("real", [], None)


def _reset_annotation(prefix: str) -> None:
    st.session_state[f"{prefix}_roi"] = None
    st.session_state[f"{prefix}_roi_shape"] = None
    st.session_state[f"{prefix}_mask"] = None
    st.session_state[f"{prefix}_points"] = []
    st.session_state[f"{prefix}_points_shape"] = None
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
    _store_roi(prefix, result.bbox, image.shape)
    score = f" score={result.score:.3f}" if result.score is not None else ""
    st.success(f"Created {prefix} mask with {result.method} backend.{score}")
    st.rerun()


def _start_camera(config: CameraConfig) -> None:
    _stop_camera()
    cap = None
    try:
        with st.spinner(f"Opening camera {config.source!r}..."):
            cap = open_camera(config)
            try:
                st.session_state.live_frame = read_rgb_frame(cap)
            except Exception as exc:
                cap.release()
                cap = None
                raise RuntimeError(
                    "Camera opened, but no frame was returned. Close other camera apps "
                    "or try a different camera backend."
                ) from exc
        st.session_state.cap = cap
        st.session_state.camera_running = True
        st.success(f"Camera started: {config.source!r}")
    except Exception as exc:
        if cap is not None:
            cap.release()
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
        with st.spinner(f"Capturing from camera {config.source!r}..."):
            cap = open_camera(config)
            try:
                st.session_state.real_frame = read_rgb_frame(cap)
            finally:
                cap.release()
        _reset_annotation("real")
        st.success("Captured one frame from camera.")
    except Exception as exc:
        st.error(str(exc))


def _maybe_rerun_live_preview() -> None:
    if not st.session_state.get("camera_running"):
        return
    if not st.session_state.get("live_preview"):
        return
    if _should_pause_live_refresh_for_annotation():
        return
    fps = max(1, int(st.session_state.get("preview_fps", DEFAULT_PREVIEW_FPS)))
    time.sleep(1.0 / fps)
    st.rerun()


def _should_pause_live_refresh_for_annotation() -> bool:
    if not st.session_state.get("live_roi_enabled"):
        return False
    if not st.session_state.get("freeze_live_while_annotating"):
        return False
    if st.session_state.get("live_annotation_tool") != "Click 4 corners":
        return False
    return len(st.session_state.get("real_points") or []) < 4
