
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def find_zarr_paths(directory: str = '/root/capsule/data', subselect: str = None) -> list:
    """
    Retrieve paths to Zarr directories within the specified directory, optionally filtered by a subdirectory.

    Args:
        directory (str): The base directory to search for Zarr files.
        subselect (str): Optional subdirectory name to filter the search.

    Returns:
        list: A list of paths to Zarr directories.
    """
    zarr_paths = []
    for root, dirs, _ in os.walk(directory):
        if subselect and subselect not in root:
            continue  # Skip directories that don't match the subselect filter

        for d in tqdm(dirs, desc=f"Searching for Zarr directories in {root}"):
            if 'zarr' in d:
                full_path = os.path.join(root, d)
                print(f"Found Zarr directory: {full_path}")
                zarr_paths.append(full_path)

    return zarr_paths

def get_crop_region() -> tuple:
    """
    Define the crop region for processing video frames.

    Returns:
        tuple: A tuple (y_start, x_start, y_end, x_end) representing the crop coordinates.
    """
    return (100, 100, 300, 400)

def get_results_folder() -> str:
    """
    Retrieve the path to the results folder. Modify this function as needed to fit your project structure.

    Returns:
        str: Path to the results folder.
    """
    # Placeholder implementation, update with actual results folder logic if needed
    return '/path/to/results'

def get_zarr_path(metadata: dict, path_to: str = 'motion_energy') -> str:
    """
    Construct the path for saving Zarr storage based on metadata.

    Args:
        metadata (dict): A dictionary containing metadata such as 'mouse_id', 'camera_label', and 'data_asset_id'.
        path_to (str): Specifies the type of frames to be saved ('gray_frames' or 'motion_energy_frames').

    Returns:
        str: Full path to the Zarr storage file.
    """
    zarr_folder = construct_zarr_folder(metadata)
    zarr_path = os.path.join(get_results_folder(), zarr_folder)

    # Create the directory if it doesn't exist
    os.makedirs(zarr_path, exist_ok=True)

    filename = 'processed_frames_zarr' if path_to == 'gray_frames' else 'motion_energy_frames.zarr'
    return os.path.join(zarr_path, filename)

def get_data_path(metadata: dict) -> str:
    """
    Construct the path for data storage based on metadata.

    Args:
        metadata (dict): A dictionary containing metadata such as 'mouse_id', 'camera_label', and 'data_asset_id'.

    Returns:
        str: Full path to the data storage folder.
    """
    data_folder = construct_zarr_folder(metadata)
    data_path = os.path.join(get_results_folder(), data_folder)

    # Create the directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    return data_path


def construct_zarr_folder(metadata: dict) -> str:
    """
    Construct the folder name for Zarr storage based on metadata.

    Args:
        metadata (dict): A dictionary containing 'mouse_id', 'camera_label', and 'data_asset_id'.

    Returns:
        str: Constructed folder name.
    """
    try:
        return f"{metadata['mouse_id']}_{metadata['camera_label']}_{metadata['data_asset_id']}"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")


def save_video(frames, video_path = '', video_name='motion_energy_clip.avi', fps=60, num_frames = 1000):
    """
    Save the provided frames to a video file using OpenCV.
    """
    
    output_video_path = os.path.join(video_path, video_name)
    # Get frame shape and number of frames
    print(type(frames))
    _, frame_height, frame_width = frames.shape

    # Specify the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Process and write each frame to the video file
    for i in range(5000, num_frames+5000):
        frame = frames[i].compute()  # Compute the frame to load it into memory
        frame = frame.astype('uint8')  # Ensure the frame is of type uint8 for video
        out.write(frame)  # Write the frame to the video file

    # Release the video writer
    out.release()
    print(f"Video saved to '{tput_video_pathou}'")

