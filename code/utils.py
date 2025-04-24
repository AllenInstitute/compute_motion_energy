import os
import json
import numpy as np
import cv2
from pathlib import Path


def get_metadata_json(obj):
    """
    Get the corresponding JSON metadata path for the video.

    Args:
        obj: An object with a 'video_path' attribute.

    Returns:
        Path: Path to the metadata JSON file.
    """
    return Path(obj.video_path).with_suffix('.json')


def construct_results_folder(metadata: dict) -> str:
    """
    Construct a results folder name based on metadata fields.

    Args:
        metadata (dict): Must contain 'mouse_id', 'camera_label', and 'data_asset_name'.

    Returns:
        str: Folder name for results.
    """
    try:
        return f"{metadata['data_asset_name']}_{metadata['camera_label']}_motion_energy"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")


def object_to_dict(obj):
    if hasattr(obj, "__dict__"):
        meta_dict = {key: object_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        meta_dict = [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        meta_dict = {key: object_to_dict(value) for key, value in obj.items()}
    else:
        meta_dict = obj
    
    # Convert Path to str for json file
    metadata_fixed = {k: str(v) if isinstance(v, Path) else v for k, v in meta_dict.items()}
    
    return metadata_fixed


def save_video(frames, video_path='', video_name='', fps=60):
    """
    Save a sequence of grayscale frames as a video file.

    Args:
        frames (list or np.ndarray): Sequence of 2D frames (grayscale).
        video_path (str or Path): Directory to save the video or full path to the video.
        video_name (str): Name of the video file (used if video_path is a directory).
        fps (int): Frames per second for the output video.
    """
    # Handle input frames (list or array)
    if isinstance(frames, list):
        frames = np.array(frames)
    if frames.ndim != 3:
        raise ValueError(f"Frames must be a 3D array (num_frames, height, width), got {frames.shape}")

    # Determine output path
    output_video_path = Path(video_path)
    if output_video_path.is_dir() and video_name:
        output_video_path = output_video_path / video_name
    elif output_video_path.suffix == '':
        raise ValueError("If 'video_name' is not provided, 'video_path' must include the filename.")

    # Frame dimensions
    num_frames, frame_height, frame_width = frames.shape

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height), isColor=False)

    # Write frames
    for i in range(num_frames):
        frame = frames[i].astype(np.uint8)
        out.write(frame)

    out.release()
    print(f"Video clip saved to '{output_video_path}'")
