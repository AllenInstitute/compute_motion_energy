""" top level run script """
import os
from tqdm import tqdm
import utils
import time  # Added for timing
from MotionEnergyAnalyzer import MotionEnergyAnalyzer
import numpy as np

DATA_PATH = utils.get_data_folder(pipeline=True)
#zarr_paths = np.unique(utils.find_zarr_paths(DATA_PATH))
zarr_paths = list(DATA_PATH.glob("*/processed_frames.zarr"))
print(f'Found {len(zarr_paths)}.')
def run():
    for zarr_path in zarr_paths:
        start_time = time.time()  # Start the timer

        me_analyser = MotionEnergyAnalyzer(zarr_path)
        me_analyser._analyze()

        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

if __name__ == "__main__": 
    run()