
import os
import json
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path


def get_metadata_json(self):
    json_path = Path(self.video_path).with_suffix('.json')
    return json_path


def construct_results_folder(metadata: dict) -> str:
    """
    Construct the folder name for Zarr storage based on metadata.

    Args:
        metadata (dict): A dictionary containing 'mouse_id', 'camera_label', and 'data_asset_name'.

    Returns:
        str: Constructed folder name.
    """
    try:
        return f"{metadata['data_asset_name']}_{metadata['camera_label']}_motion_energy"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")


def object_to_dict(obj):
    """
    Recursively converts an object to a dictionary.

    Args:
        obj: The object to convert.

    Returns:
        dict: The dictionary representation of the object.
    """
    if hasattr(obj, "__dict__"):
        return {key: object_to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: object_to_dict(value) for key, value in obj.items()}
    return obj


def save_video(frames, video_path='', video_name='', fps=60):
    """
    Save the provided frames to a video file using OpenCV.
    """

    output_video_path = Path(video_path, video_name)

    # Ensure frames is a NumPy array or Dask array
    print(f"Frames type: {type(frames)}")

    # Get frame shape and check dimensions
    frame_height, frame_width = frames.shape[1:3]  # Assume (num_frames, H, W)
    num_frames = frames.shape[0]
    
    # Specify the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Process and write each frame to the video file
    for i in range(1,num_frames):  # Start from 1 to num_frames-1
        if i >= frames.shape[0]:  # Prevent out-of-bounds access
            print(f"Warning: Requested frame {i} exceeds available frames.")
            break
            
        frame = frame.astype(np.uint8)  # Convert to uint8
        out.write(frame)  # Write the frame to the video file

    # Release the video writer
    out.release()
    print(f"Video clip saved to '{output_video_path}'")


