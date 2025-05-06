
# MotionEnergyAnalyzer

`MotionEnergyAnalyzer` is a Python class that processes grayscale video data to compute **motion energy (ME)**—a pixel-wise frame-to-frame difference—and saves the results in both video and numerical formats. This tool is designed for behavioral or neuroscience experiments where motion energy is used to quantify movement over time.

---

## Features

* Computes **motion energy frames** using absolute pixel differences between consecutive frames
* Saves ME as:

  * A grayscale `.mp4` video
  * A `.csv` file with per-frame ME sum values (motion trace)
* Optionally extracts short clips from the video and ME output
* Organizes and saves results in structured directories using metadata
* Loads and saves metadata to/from `postprocess_metadata.json`

---

## Installation

Install required packages:

```bash
pip install numpy opencv-python pandas
```

You also need a `utils.py` module that includes:

* `get_metadata_json(self)`
* `construct_results_folder(metadata: dict)`
* `object_to_dict(obj)`
* `save_video(frame_list, video_path: Path, fps: float)`

---

## Input Requirements

1. **Video file** (e.g., `.mp4`) as input
2. A corresponding `postprocess_metadata.json` file located next to the video, with the following structure:

```json
{
  "video_metadata": {
    "mouse_id": "123456",
    "camera_label": "Face",
    "video_name": "Face_20230101T101010"
  }
}
```

---

## Usage

```python
from motion_energy_analyzer import MotionEnergyAnalyzer  # Update with actual filename
from pathlib import Path

analyzer = MotionEnergyAnalyzer(Path("/path/to/video.mp4"))
analyzer.analyze()
```

This will:

* Load metadata
* Compute motion energy video and trace
* Save ME `.mp4` and `.csv` files
* Extract and save short clips from both the original and ME videos
* Save updated `postprocess_metadata.json` in the results folder

---

## Output

All results are saved under `/results/{mouse_id}_{camera_label}_{video_name}/`, including:

* `Face_20230101T101010_motion_energy.mp4`: ME video
* `Face_20230101T101010_motion_energy_sums.csv`: CSV of motion energy trace
* `gray_video_clip.mp4`: 30s clip from original grayscale video
* `motion_energy_clip.mp4`: 30s clip from ME video
* `postprocess_metadata.json`: Metadata for downstream tools

---

## Configuration

You can configure clip extraction using the class attributes:

```python
analyzer.start_sec = 10      # start of clip in seconds
analyzer.duration_sec = 20   # duration of clip in seconds
```

---

## Notes

* All frames are converted to grayscale if needed.
* The first frame is skipped for ME calculation, but included in clips.
* Motion energy is calculated using OpenCV's `absdiff()` followed by normalization to \[0, 255].

---


