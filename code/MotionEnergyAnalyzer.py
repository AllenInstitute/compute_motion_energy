
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
    def __init__(self, video_path: str, crop: bool = True):
        self.video_oath = video_path
        self.video_metadata = None
        self.crop = crop
        if crop:
            self.crop_region = utils.get_crop_region()
            
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
        return results_path

    def _get_motion_energy_frame(prev_gray, gray):
        motion_energy_frame = cv2.absdiff(gray, prev_gray)
        motion_energy_frame = cv2.normalize(motion_energy_frame, None, 0, 255, cv2.NORM_MINMAX)
        motion_energy_frame = motion_energy_frame.astype(np.uint8)  # Ensure 8-bit for writing
        return motion_energy_frame


    def _compute_ME_from_video(self):

        ## LOAD AND PROCESS VIDEO
        # Open the input video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {input_video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object for ME frames
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video_path = Path(self._get_full_results_path, "motion_energy_video.avi")
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)
        
        # Define CSV file for ME trace
        self.me_sums_output_path = Path(self._get_full_results_path, "motion_energy_sums.csv")
        frame_idx = 1  # Start from frame 1 (since first frame is skipped)
        motion_energy_sums = []  # Store ME sums

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
            prev_gray = gray
            frame_idx += 1

        cap.release()
        out.release()

         # Save ME sums to CSV
        df = pd.DataFrame({'motion_energy_sum': motion_energy_sums})
        df.to_csv(self.me_sums_output_path, index=True)

        print(f"Motion energy video saved to {self.output_video_path}")
        print(f"Motion energy sums saved to {self.me_sums_output_path}")


    def analyze(self):
        """
        Analyze motion energy based on the frames.
        Applies cropping if the crop attribute is True and saves results.
        """
        
        # Load metadata
        self._load_metadata()

        # Compute ME frame by frame, save results
        self._compute_ME_from_video(self)



        try:
            # save motion energy trace for redundancy as np array
            np.savez(f'{top_zarr_path}/motion_energy_trace.npz', 
                full_frame_motion_energy = self.full_frame_motion_energy_sum, cropped_frame_motion_energy = self.cropped_frame_motion_energy_sum)
            print('saved motion energy trace to npz file for redundancy.')
        except:
            print('Could not save npz file {top_zarr_path}')

        ## save object as dictionary
        try:
            obj_dict = utils.object_to_dict(self)  
            with open(f'{top_zarr_path}/motion_energy_dictionary.pkl', 'wb') as file:
                pickle.dump(obj_dict, file)
            print('saved motion energy object as dicitonary, for redundancy.')
        except:
            print('Could not save pkl file {top_zarr_path}')

        # Save motion energy frames to zarr
        me_zarr_filename = utils.get_zarr_filename(path_to='motion_energy')
        me_zarr_path = os.path.join(top_zarr_path, me_zarr_filename)

        me_zarr_store = zarr.DirectoryStore(me_zarr_path)
        root_group = zarr.group(me_zarr_store, overwrite=True)
        if self.crop:
            cropped_motion_energy_frames.to_zarr(me_zarr_store, component='cropped_frames', overwrite=True)
            print('Saved cropped frames too.')
        motion_energy_frames.to_zarr(me_zarr_store, component='full_frames', overwrite=True)
        print(f'Saved motion energy frames to {me_zarr_path}')

        ### Add metadata to the Zarr store ###
        # Turn object attributed to dicitonary
       
        meta_dict = utils.object_to_dict(self)
        root_group.attrs['metadata'] = json.dumps(meta_dict, cls = NumpyEncoder)
        print('added metadata to zarr files.')

        ### Save motion energy frames as a video ###
        # path in results to where data from this video will be saved
        utils.save_video(frames = motion_energy_frames, video_path = top_zarr_path,
        video_name='motion_energy_clip.avi', fps=self.video_metadata.get('fps'), num_frames=1000)
        utils.save_video(frames = grayscale_frames, video_path = top_zarr_path,
        video_name='example_video_clip.avi', fps=self.video_metadata.get('fps'), num_frames=1000)

        

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
        return json.JSONEncoder.default(self, obj)

    ## TypeError: _compute_motion_energy() takes 1 positional argument but 2 were given

    # def _compute_motion_energy(frames):
    #     """
    #     Compute motion energy from a set of frames.
    #     Motion energy is computed as the sum of absolute differences between consecutive frames.
    #     """
    #     if len(frames) < 2:
    #         raise ValueError("At least two frames are required to compute motion energy.")
        
    #     motion_energy = da.abs(frames[1:] - frames[:-1])
    #     return motion_energy

        

