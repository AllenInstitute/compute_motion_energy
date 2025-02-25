
import os
import json
import numpy as np
from tqdm import tqdm
import cv2

def find_zarr_paths(directory: str = '/root/capsule/data', subselect: str = '', tag: str = '') -> list:
    """
    Retrieve paths to Zarr directories within the specified directory, optionally filtered by a subdirectory.

    Args:
        directory (str): The base directory to search for Zarr files.
        subselect (str): Optional subdirectory name to filter the search.
        tag (str): str tag in video filename to include. (not being used)

    Returns:
        list: A list of paths to Zarr directories.
    """
    zarr_paths = []
    for root, dirs, _ in os.walk(directory):
        if subselect not in root:
            continue  # Skip directories that don't match the subselect filter
        
        
        for d in tqdm(dirs, desc=f"Searching for Zarr directories in {root}"):
            if 'zarr' in d:
                full_path = os.path.join(root, d)
                print(f"\nFound Zarr directory: {full_path}")
                zarr_paths.append(full_path)

    return zarr_paths

def get_crop_region() -> tuple:
    """
    Define the crop region for processing video frames.

    Returns:
        tuple: A tuple (y_start, x_start, y_end, x_end) representing the crop coordinates.
    """
    # return (100, 100, 300, 400)
    return (200, 290, 280, 360) # for Thyme

def get_results_path() -> str:
    """
    Retrieve the path to the results folder. Modify this function as needed to fit your project structure.

    Returns:
        str: Path to the results folder.
    """
    # Placeholder implementation, update with actual results folder logic if needed
    return '/root/capsule/results'

def get_zarr_filename(path_to: str = 'motion_energy') -> str:
    """
    Construct the path for saving Zarr storage based on metadata.

    Args:
        metadata (dict): A dictionary containing metadata such as 'mouse_id', 'camera_label', and 'data_asset_name'.
        path_to (str): Specifies the type of frames to be saved ('gray_frames' or 'motion_energy_frames').

    Returns:
        str: Full path to the Zarr storage file.
    """

    filename = 'processed_frames_zarr' if path_to == 'gray_frames' else 'motion_energy_frames.zarr'
    return filename

def get_data_path(metadata: dict) -> str:
    """
    Construct the path for data storage based on metadata.

    Args:
        metadata (dict): A dictionary containing metadata such as 'mouse_id', 'camera_label', and 'data_asset_name'.

    Returns:
        str: Full path to the data storage folder.
    """
    data_folder = construct_zarr_folder(metadata)
    data_path = os.path.join(get_results_path(), data_folder)

    # Create the directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    return data_path


def construct_zarr_folder(metadata: dict) -> str:
    """
    Construct the folder name for Zarr storage based on metadata.

    Args:
        metadata (dict): A dictionary containing 'mouse_id', 'camera_label', and 'data_asset_name'.

    Returns:
        str: Constructed folder name.
    """
    try:
        return f"{metadata['mouse_id']}_{metadata['data_asset_name']}_{metadata['camera_label']}_motion_energy"
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

import os
import cv2
import numpy as np

def save_video(frames, video_path='', video_name='motion_energy_clip.avi', fps=60, num_frames=1000):
    """
    Save the provided frames to a video file using OpenCV.
    """

    # Ensure the output directory exists
    if video_path and not os.path.exists(video_path):
        os.makedirs(video_path)

    output_video_path = os.path.join(video_path, video_name)

    # Ensure frames is a NumPy array or Dask array
    print(f"Frames type: {type(frames)}")

    # Get frame shape and check dimensions
    frame_height, frame_width = frames.shape[1:3]  # Assume (num_frames, H, W)
    
    # Specify the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Process and write each frame to the video file
    for i in range(1,num_frames):  # Start from 1 to num_frames-1
        if i >= frames.shape[0]:  # Prevent out-of-bounds access
            print(f"Warning: Requested frame {i} exceeds available frames.")
            break

        frame = frames[i]
        if hasattr(frame, 'compute'):  # If it's a Dask array
            frame = frame.compute()
            
        frame = frame.astype(np.uint8)  # Convert to uint8
        out.write(frame)  # Write the frame to the video file

    # Release the video writer
    out.release()
    print(f"Video saved to '{output_video_path}'")


