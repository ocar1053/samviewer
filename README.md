# Camera Alignment Viewer

Streamlit + OpenCV MVP for comparing a live or captured physical camera image
against a virtual/reference image. The first research goal is approximate
viewpoint alignment: make the target object, such as a table, appear close in
pixel size and center position across the two images.

## Features

- OpenCV camera input from webcam, USB camera, video file, or URL.
- Live camera preview with optional reference bbox overlay.
- Live real-object bbox or four-corner ROI drawing on the camera preview with
  overlap percentage.
- Reference image folder picker, upload, or path loading.
- Captured real frame snapshot.
- Manual ROI selection for the reference object.
- Browser-based mouse annotation:
  - drag a bbox over the object
  - click four ordered table corners
- OpenCV `selectROI` window when a desktop GUI is available.
- Numeric ROI fallback for remote or headless environments.
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

On Windows, prefer running Streamlit through the project's virtual environment:

```powershell
.\.venv\Scripts\python -m streamlit run main.py
```

Then open the URL printed by Streamlit, usually `http://localhost:8501`.

## Basic Workflow

1. Choose camera source `0`, `1`, `2`, or a custom path/URL.
2. Click **Start camera**.
3. Load a reference image. Put reusable images in `reference_images/` and choose
   one from the folder picker, or upload / load a path manually.
4. In **Reference ROI**, click the four reference corners. Once the camera is
   running, the reference image is resized to the same pixel size as the live
   camera frame before annotation.
5. In **Live camera**, enable **Live real ROI** and choose either **Drag bbox**
   or **Click 4 corners**.
6. Use the live matching percentage and reference bbox overlay while moving the physical
   camera.

## Windows Camera Troubleshooting

- Use camera source `0` first. If needed, click **Scan camera indices**.
- Use **Windows DirectShow** as the camera backend. It is the default on Windows.
- Close apps that may already hold the camera, such as Windows Camera, Teams,
  Zoom, OBS, or browser webcam tabs.
- If **Start camera** appears stuck, lower the capture resolution, then try again.
- Make sure Windows allows desktop apps to use the camera in
  **Settings > Privacy & security > Camera**.

## Mouse Annotation

The reference ROI panel has three annotation tools:

- **Drag bbox**: drag over the target object to set the bbox.
- **Click 4 corners**: click the object corners in order: top-left, top-right,
  bottom-right, bottom-left. The app draws a quadrilateral and uses its tight
  bbox for live matching.
- **Numeric / OpenCV**: edit bbox numbers directly or use OpenCV `selectROI`.

The live camera panel also defaults to **Click 4 corners** for the real object.
When both reference and live real ROIs are set, the app shows the live match
percentage above the live image.

## Notes

- The app resizes the reference image to the live camera frame size before ROI
  selection, so reference and live ROIs use the same pixel coordinate system.
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
4. Select a reference ROI first, then click **Create reference mask from ROI**.

The app uses your selected ROI as SAM2's box prompt, chooses the highest-scoring
mask, computes a tight bbox from that mask, and updates the ROI used by the
live matching.

## Future Extensions

- Add point prompts for positive/negative corrections after the box prompt.
- Add four-point correspondence selection and homography estimation.
- Add ArUco/AprilTag marker detection for scale or camera pose hints.
