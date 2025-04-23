
import numpy as np
import zarr
import dask
import dask.array as da
import json
import os
import utils
import pickle

load_from_zarr_files = False
RESULTS = Path("/results")

class MotionEnergyAnalyzer:
    def __init__(self, video_path: str):
        self.video_oath = video_path
        self.video_metadata = None
        self.start_sec = 15
        self.duration_sec = 30
            
    def _load_metadata(self):
        """Load metadata from json file."""
        json_path = utils.get_metadata_json(self)
        with json_path.open('r') as f:
            metadata = json.load(f)
        self.video_metadata = metadata
    
    def _validate_frame(frame):
        if frame.ndim == 2:  # Already grayscale
            return frame
        elif frame.ndim == 3 and frame.shape[2] == 3:  # Color image (BGR)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

    def _get_full_results_path(self):
        folder = utils.construct_results_folder(self.metadata)
        results_path = Path(RESULTS, folder)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        return results_path

    def _get_motion_energy_frame(prev_gray, gray):
        motion_energy_frame = cv2.absdiff(gray, prev_gray)
        motion_energy_frame = cv2.normalize(motion_energy_frame, None, 0, 255, cv2.NORM_MINMAX)
        motion_energy_frame = motion_energy_frame.astype(np.uint8)  # Ensure 8-bit for writing
        return motion_energy_frame

    def _save(self):
        meta_dict = utils.object_to_dict(self)
        me_metadata_path = Path(self._get_full_results_path(), "motion_energy_metadata.json")
        with me_metadata_path.open('w') as f:
            json.dump(, f, indent=4)

    def _compute_ME_from_video(self):

        ## LOAD AND PROCESS VIDEO
        # Open the input video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {input_video_path}")

        # Get video properties
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object for ME frames
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video_path = Path(self._get_full_results_path(), "motion_energy_video.avi")
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height), isColor=False)
        
        # Define CSV file for ME trace
        self.me_sums_output_path = Path(self._get_full_results_path(), "motion_energy_sums.csv")
        frame_idx = 1  # Start from frame 1 (since first frame is skipped)
        motion_energy_sums = []  # Store ME sums

        # Save short videos
        behavior_video_clip = []
        motion_energy_clip = []
        start_frame = int(self.start_sec * self.fps)
        end_frame = int((self.start_sec + self.duration_sec) * self.fps)

        # Read the first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise IOError("Error reading the first frame from the video.")
        prev_gray = self._validate_frame(prev_frame)


        ## COMPUTE MOTION ENERGY
        # Iterate from second frame onward
        while True:
            ret, frame = cap.read()
            if not ret:
                break      

            gray = self._validate_frame(frame)
            motion_energy_frame = self._get_motion_energy_frame(gray, prev_gray)

            me_sum = int(np.sum(motion_energy_frame))
            motion_energy_sums.append(me_sum)

            out.write(motion_energy_frame)

            #save short clips for examples
            if frame_index > start_frame and frame_index < end_frame:
                behavior_video_clip.append(gray)
                motion_energy_clip.append(motion_energy_frame)

            prev_gray = gray
            frame_idx += 1

        cap.release()
        out.release()

         # Save ME sums to CSV
        df = pd.DataFrame({'motion_energy_sum': motion_energy_sums})
        df.to_csv(self.me_sums_output_path, index=True)

        # Save short clips
        utils.save_video(behavior_video_clip, video_path = Path(self._get_full_results_path(), "gray_video_clip.avi"), fps = self.fps)
        utils.save_video(motion_energy_clip, video_path = Path(self._get_full_results_path(), "motion_energy_clip.avi"), fps = self.fps)
        print(f"Motion energy video saved to {self.output_video_path}")
        print(f"Motion energy sums saved to {self.me_sums_output_path}")

        return self


    def analyze(self):
        """
        Analyze motion energy based on the frames.
        Applies cropping if the crop attribute is True and saves results.
        """
        
        # Load metadata
        self._load_metadata()

        # Compute ME frame by frame, save results
        self._compute_ME_from_video()

        # Save object as a dictionary to motion_energy_metadata.json
        self._save()

      

# class NumpyEncoder(json.JSONEncoder):
#     """ Special json encoder for numpy types """
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return str(obj)
#         return json.JSONEncoder.default(self, obj)


        

