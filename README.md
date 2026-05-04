# Camera Alignment Viewer

Streamlit + OpenCV MVP for comparing a live or captured physical camera image
against a virtual/reference image. The first research goal is approximate
viewpoint alignment: make the target object, such as a table, appear close in
pixel size and center position across the two images.

## Features

- OpenCV camera input from webcam, USB camera, video file, or URL.
- Live camera preview with optional reference bbox overlay.
- Reference image upload or path loading.
- Captured real frame snapshot.
- Manual ROI selection for the reference object and the real object.
- OpenCV `selectROI` window when a desktop GUI is available.
- Numeric ROI fallback for remote or headless environments.
- Pixel metrics for bbox width, height, area, center, size error, and center
  offset.
- Side-by-side view and blended overlay view.
- Optional segmentation extension point with a bbox-mask fallback. SAM2 can be
  wired into `src/samviewer/segmentation.py` later without changing the UI flow.

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
6. Enable ROI for the captured real frame and select the same table.
7. Use the metrics and live reference bbox overlay while moving the physical
   camera.

## Notes

- For true pixel-size comparison, use the same resolution for the rendered
  reference image and the camera frame whenever possible.
- If the image resolutions differ, the UI can normalize the reference bbox into
  the real frame coordinate system for metric calculation and overlay.
- `OpenCV selectROI` opens a native desktop window. If the app runs on a remote
  server without GUI forwarding, use the numeric ROI controls instead.

## Future Extensions

- Add a SAM2 backend in `segmentation.py` for mask generation from points or
  boxes.
- Add four-point correspondence selection and homography estimation.
- Add ArUco/AprilTag marker detection for scale or camera pose hints.
