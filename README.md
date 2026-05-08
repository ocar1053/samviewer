# Camera Alignment Viewer

Streamlit + OpenCV MVP for comparing a live or captured physical camera image
against a virtual/reference image. The first research goal is approximate
viewpoint alignment: make the target object, such as a table, appear close in
pixel size and center position across the two images.

## Features

- OpenCV camera input from webcam, USB camera, video file, or URL.
- Live camera preview with optional reference bbox overlay.
- Live real-object ROI drawing on the camera preview with overlap percentage.
- Reference image upload or path loading.
- Captured real frame snapshot.
- Manual ROI selection for the reference object and the real object.
- Browser-based mouse annotation:
  - drag a bbox over the object
  - click four ordered table corners
- OpenCV `selectROI` window when a desktop GUI is available.
- Numeric ROI fallback for remote or headless environments.
- Pixel metrics for bbox width, height, area, center, size error, and center
  offset.
- Corner metrics for four corresponding points, including per-corner offset,
  mean/max offset, polygon area error, and center offset.
- Side-by-side view and blended overlay view.
- Optional segmentation backend. The default bbox-mask fallback has no extra
  dependencies; SAM2 can be enabled separately and used with a box prompt.

## Setup

```bash
cd /home/ocarpan/samviewer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable package installation:

```bash
pip install -e .
```

## Run

```bash
cd /home/ocarpan/samviewer
streamlit run main.py
```

Then open the URL printed by Streamlit, usually `http://localhost:8501`.

## Basic Workflow

1. Choose camera source `0`, `1`, `2`, or a custom path/URL.
2. Click **Start camera**.
3. Load a reference image.
4. Click **Capture current frame**.
5. Enable ROI for the reference image and select the table.
6. Enable ROI for the captured real frame and click the four real table corners
   in order: top-left, top-right, bottom-right, bottom-left.
7. Use the live overlap percentage and reference bbox overlay while moving the physical
   camera.

## Mouse Annotation

Each reference/real panel has three annotation tools:

- **Drag bbox**: drag over the target object to set the bbox.
- **Click 4 corners**: click the object corners in order: top-left, top-right,
  bottom-right, bottom-left. The app draws a quadrilateral and uses its tight
  bbox for the basic pixel-size metrics.
- **Numeric / OpenCV**: edit bbox numbers directly or use OpenCV `selectROI`.

When both reference and real images have four corners, the app also reports
corner offsets and polygon area error. This is useful for table-top alignment
because it compares the visible table plane more directly than a plain bbox.
The real ROI panel defaults to **Click 4 corners** so the live preview can use
the tight bbox from those four points for overlap percentage.

## Notes

- For true pixel-size comparison, use the same resolution for the rendered
  reference image and the camera frame whenever possible.
- If the image resolutions differ, the UI can normalize the reference bbox into
  the real frame coordinate system for metric calculation and overlay.
- `OpenCV selectROI` opens a native desktop window. If the app runs on a remote
  server without GUI forwarding, use the numeric ROI controls instead.

## Optional SAM2 Backend

This project keeps SAM2 optional because it pulls in PyTorch, model weights, and
possibly CUDA compilation. The app will keep working with the bbox fallback if
SAM2 is not installed.

Official repo: <https://github.com/facebookresearch/sam2>

Recommended setup:

```bash
cd /home/ocarpan/samviewer
source .venv/bin/activate

# Install PyTorch/TorchVision for your CUDA or CPU environment first.
# See https://pytorch.org/get-started/locally/

git clone https://github.com/facebookresearch/sam2.git /home/ocarpan/sam2
cd /home/ocarpan/sam2
pip install -e .

# Needed only when loading models by Hugging Face model id in the viewer.
pip install huggingface_hub
```

In the viewer:

1. Expand **Optional segmentation interface**.
2. Choose **SAM2**.
3. Use `facebook/sam2.1-hiera-tiny` for the lightest Hugging Face option, or
   choose a local config + checkpoint.
4. Select an ROI first, then click **Create reference mask from ROI** or
   **Create real mask from ROI**.

The app uses your selected ROI as SAM2's box prompt, chooses the highest-scoring
mask, computes a tight bbox from that mask, and updates the ROI used by the
alignment metrics.

## Future Extensions

- Add point prompts for positive/negative corrections after the box prompt.
- Add four-point correspondence selection and homography estimation.
- Add ArUco/AprilTag marker detection for scale or camera pose hints.
