import numpy as np
import zarr
import dask.array as da
import json
import os
import cv2
import pandas as pd
from pathlib import Path
import utils

RESULTS = Path("/results")

class MotionEnergyAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_metadata = None
        self.start_sec = 15
        self.duration_sec = 30

    def _load_metadata(self):
        """Load video metadata from a JSON file."""
        json_path = utils.get_metadata_json(self)
        with json_path.open('r') as f:
            metadata = json.load(f)
        self.video_metadata = metadata

    def _validate_frame(self, frame):
        """Ensure frames are grayscale."""
        if frame.ndim == 2:
            return frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

    def _get_motion_energy_frame(self, prev_gray, gray):
        """Compute motion energy frame (absdiff, normalize, uint8)."""
        motion_energy_frame = cv2.absdiff(gray, prev_gray)
        motion_energy_frame = cv2.normalize(motion_energy_frame, None, 0, 255, cv2.NORM_MINMAX)
        return motion_energy_frame.astype(np.uint8)

    def _get_full_results_path(self):
        """Construct and return full results path."""
        folder = utils.construct_results_folder(self.video_metadata)
        results_path = RESULTS / folder
        results_path.mkdir(parents=True, exist_ok=True)
        return results_path

    def _save(self):
        """Save object metadata as JSON."""
        meta_dict = utils.object_to_dict(self)
        me_metadata_path = self._get_full_results_path() / "motion_energy_metadata.json"
        with me_metadata_path.open('w') as f:
            json.dump(meta_dict, f, indent=4)

    def _compute_ME_from_video(self):
        """Compute and save motion energy video and sums."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {self.video_path}")

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video for motion energy
        output_video_path = self._get_full_results_path() / "motion_energy_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height), isColor=False)

        # Motion energy sums CSV path
        me_sums_output_path = self._get_full_results_path() / "motion_energy_sums.csv"

        # Frame indices for short clip
        start_frame = int(self.start_sec * fps)
        end_frame = int((self.start_sec + self.duration_sec) * fps)

        # Initialize variables
        ret, prev_frame = cap.read()
        if not ret:
            raise IOError("Error reading the first frame.")
        prev_gray = self._validate_frame(prev_frame)

        motion_energy_sums = []
        behavior_video_clip = []
        motion_energy_clip = []

        frame_idx = 1  # Start from second frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = self._validate_frame(frame)
            me_frame = self._get_motion_energy_frame(prev_gray, gray)
            me_sum = int(np.sum(me_frame))

            # Write ME frame and sum
            out.write(me_frame)
            motion_energy_sums.append(me_sum)

            # Save clips if in selected time window
            if start_frame <= frame_idx < end_frame:
                behavior_video_clip.append(gray)
                motion_energy_clip.append(me_frame)

            prev_gray = gray
            frame_idx += 1

        cap.release()
        out.release()

        # Save ME sums
        df = pd.DataFrame({'motion_energy_sum': motion_energy_sums})
        df.to_csv(me_sums_output_path, index=True)

        # Save video clips
        utils.save_video(behavior_video_clip, video_path=self._get_full_results_path() / "gray_video_clip.avi", fps=fps)
        utils.save_video(motion_energy_clip, video_path=self._get_full_results_path() / "motion_energy_clip.avi", fps=fps)

        print(f"Motion energy frames saved to {output_video_path}")
        print(f"Motion energy sums saved to {me_sums_output_path}")

    def analyze(self):
        """Main method to compute motion energy and save results."""
        self._load_metadata()
        self._compute_ME_from_video()
        self._save()

