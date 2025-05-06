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
    json_file = video_path.parent.rglob("*metadata.json")
    #video_path = str(obj.video_path).replace("_processed", "metadata")
    #return Path(video_path).with_suffix('.json')
    return json_file


def construct_results_folder(metadata: dict) -> str:
    """
    Construct a results folder name based on metadata fields.

    Args:
        metadata (dict): Must contain 'camera_label', and 'data_asset_name'.

    Returns:
        str: Folder name for results.
    """
    try:
        return f"{metadata['data_asset_name']}_{metadata['camera_label']}_motion_energy"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")


def object_to_dict(obj):
    """
    Recursively convert an object (with __dict__) to a dictionary,
    converting any non-JSON-serializable elements to serializable types.
    
    Args:
        obj: The object or structure to convert.
    
    Returns:
        A JSON-serializable dictionary or list representation.
    """
    if hasattr(obj, "__dict__"):
        data = {key: object_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, dict):
        data = {key: object_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        data = [object_to_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        data = tuple(object_to_dict(item) for item in obj)
    else:
        data = obj

    return _obj_to_dict(data)


def _obj_to_dict(data):
    """
    Recursively convert non-JSON-serializable items to JSON-compatible types.
    
    Args:
        data: The data structure (dict, list, etc.) to convert.
    
    Returns:
        A structure with all values converted to JSON-serializable types.
    """
    if isinstance(data, dict):
        return {k: _obj_to_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_obj_to_dict(item) for item in data]
    elif isinstance(data, tuple):
        return [_obj_to_dict(item) for item in data]  # Convert tuple to list
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    else:
        return data


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
