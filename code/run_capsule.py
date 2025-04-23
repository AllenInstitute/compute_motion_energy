""" top level run script """
import os
# from tqdm import tqdm
import utils
import time
from pathlib import Path

from MotionEnergyAnalyzer import MotionEnergyAnalyzer
import numpy as np

DATA_PATH = Path("/data")
video_extensions = ('*.mp4', '*.avi')

# Use glob for each extension
video_files = []
for ext in video_extensions:
    video_files.extend(data_path.rglob(ext))  # Recursive search


print(f'Found {len(video_files)}.')
def run():
    for video_file in video_files:
        start_time = time.time()  # Start the timer

        me_analyser = MotionEnergyAnalyzer(video_file)
        me_analyser.analyze()

        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

if __name__ == "__main__": 
    run()