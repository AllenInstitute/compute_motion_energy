import numpy as np
import json
import os
import cv2
import pandas as pd
from pathlib import Path
import utils

RESULTS = Path("/results")

class MotionEnergyAnalyzer:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.video_metadata = None
        self.start_sec = 15
        self.duration_sec = 30

    def _load_metadata(self):
        """Load video metadata from a JSON file."""
        json_path = utils.get_metadata_json(self.video_path)
        with Path(json_path).open('r') as f:
            metadata = json.load(f)
        self.video_metadata = metadata
        print(f'Metadata from {json_path} loaded successfully.')
        self._get_full_results_path()

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
        self.full_results_path = results_path
        return self


    def _compute_ME_from_video(self):
        """Compute and save motion energy video and sums."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {self.video_path}")

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.gray_video_fps = fps
        self.gray_video_size = (frame_height, frame_width)

        # Output video for motion energy
        output_video_path = self.full_results_path / f"{self.video_metadata['video_name']}_motion_energy.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height), isColor=False)

        # Motion energy sums CSV path
        me_sums_output_path = self.full_results_path / f"{self.video_metadata['video_name']}_motion_energy_sums.csv"

        # Frame indices for short clip
        start_frame = int(self.start_sec * fps)
        end_frame = int((self.start_sec + self.duration_sec) * fps)

        # Initialize variables
        ret, prev_frame = cap.read()
        if not ret:
            raise IOError("Error reading the first frame.")
        prev_gray = self._validate_frame(prev_frame)

        motion_energy_sums = [] # collects ME trace values over time
        behavior_video_clip = [] # collects frames for short example video
        motion_energy_clip = [] # collects frames for short example video

        frame_idx = 0
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
        self.motion_energy_trace = motion_energy_sums

        # Save video clips
        utils.save_video(behavior_video_clip, video_path=self.full_results_path / "gray_video_clip.mp4", fps=fps)
        utils.save_video(motion_energy_clip, video_path=self.full_results_path / "motion_energy_clip.mp4", fps=fps)

        print(f"Motion energy frames saved to {output_video_path}")
        print(f"Motion energy sums saved to {me_sums_output_path}")

    def _save(self):
        """Save object metadata as JSON."""
        metadata = {} # create new dictionary to organize video and ME metadata
        # this is overcomplicated, but metadata from loading and compute motion energy capsules is saved separately
        metadata['video_metadata'] = utils.object_to_dict(self.video_metadata) 
        me_dict = utils.object_to_dict(self.__dict__.pop('video_metadata'))
        metadata['me_metadata'] = me_dict
        me_metadata_path = self.full_results_path / "postprocess_metadata.json"
        with me_metadata_path.open('w') as f:
            json.dump(metadata, f, indent=4)

    def analyze(self):
        """Main method to compute motion energy and save results."""
        self._load_metadata()
        self._compute_ME_from_video()
        self._save()

