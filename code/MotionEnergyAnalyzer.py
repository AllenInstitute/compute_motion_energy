Here's the cleaned-up version of the code with corrected typos, missing variables, and formatting improvements:

```python
import numpy as np
import cv2
import zarr
import dask
import dask.array as da
import json
import os
import utils


class MotionEnergyAnalyzer:
    def __init__(self, frame_zarr_path: str):
        self.frame_zarr_path = frame_zarr_path
        self.zarr_store_frames = zarr.DirectoryStore(frame_zarr_path)
        self.loaded_metadata = None

    def _load_metadata(self):
        """Load metadata from the Zarr store."""
        root_group = zarr.open_group(self.zarr_store_frames, mode='r')
        self.loaded_metadata = json.loads(root_group.attrs['metadata'])

    def _save_video(frames, output_video_path='output_movie.avi', fps=60, num_frames = 1000):
        """
        Save the provided frames to a video file using OpenCV.
        """
        # Get frame shape and number of frames
        _, frame_height, frame_width = frames.shape

        # Specify the codec and create the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

        # Process and write each frame to the video file
        for i in range(num_frames):
            frame = frames[i].compute()  # Compute the frame to load it into memory
            frame = frame.astype('uint8')  # Ensure the frame is of type uint8 for video
            out.write(frame)  # Write the frame to the video file

        # Release the video writer
        out.release()
        print(f"Video saved to '{output_video_path}'")

    def compute_motion_energy(self, frames):
        """
        Compute motion energy from a set of frames.
        Motion energy is computed as the sum of absolute differences between consecutive frames.
        """
        if len(frames) < 2:
            raise ValueError("At least two frames are required to compute motion energy.")
        
        motion_energy = da.abs(frames[1:] - frames[:-1])
        return motion_energy


    def analyze(self):
        """
        Analyze motion energy based on the frames.
        Applies cropping if the crop attribute is True and saves results.
        """
        # Load the frames from Zarr
        grayscale_frames = da.from_zarr(self.zarr_store_frames)

        # Load metadata
        self._load_metadata()

        # Check for cropping option
        crop = self.loaded_metadata.get('crop', False)

        if crop:
            crop_region = utils.get_crop_region()
            crop_y_start, crop_x_start, crop_y_end, crop_x_end = crop_region
            motion_energy = self.compute_motion_energy(grayscale_frames)
            cropped_motion_energy = motion_energy[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        else:
            motion_energy = self.compute_motion_energy(grayscale_frames)

        # Save motion energy frames as a video
        self._save_video(motion_energy, fps=self.loaded_metadata['fps'], num_frames=10000)

        # Save motion energy frames to Zarr
        zarr_path = utils.get_zarr_path(self)
        zarr_store = zarr.DirectoryStore(zarr_path)
        motion_energy.to_zarr(zarr_store, component='data', overwrite=True)

        # Add metadata to the Zarr store
        root_group = zarr.group(store=zarr_store, overwrite=True)
        root_group.attrs['metadata'] = json.dumps(self.loaded_metadata)
        print(f'Saved motion energy frames to {zarr_path}')

        # Compute trace and save it to the object
        sum_trace = motion_energy.sum(axis=(1, 2)).compute()
        self.motion_energy_sum = sum_trace.reshape(-1, 1)

        return self
```
