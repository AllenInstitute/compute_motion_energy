
import numpy as np
import zarr
import dask
import dask.array as da
import json
import os
import utils
import pickle

class MotionEnergyAnalyzer:
    def __init__(self, frame_zarr_path: str, crop: bool = True):
        self.frame_zarr_path = frame_zarr_path
        self.zarr_store_frames = zarr.DirectoryStore(frame_zarr_path)
        self.video_metadata = None
        self.crop = crop
        if crop:
            self.crop_region = utils.get_crop_region()
            
    def _load_metadata(self):
        """Load metadata from the Zarr store."""
        root_group = zarr.open_group(self.zarr_store_frames, mode='r')
        metadata = json.loads(root_group.attrs['metadata'])
        self.video_metadata = metadata


    def _analyze(self):
        """
        Analyze motion energy based on the frames.
        Applies cropping if the crop attribute is True and saves results.
        """
        ### Load the frames from Zarr ###
        grayscale_frames = da.from_zarr(self.zarr_store_frames, component='data')

        ### Load metadata ###
        self._load_metadata()

        ### Compute motion energy frames ###
        H, W = self.video_metadata.get('height'), self.video_metadata.get('width')
        motion_energy_frames = da.abs(grayscale_frames[1:] - grayscale_frames[:-1])
        print('Dropped first frame of the video since its metadata')
        motion_energy_frames = motion_energy_frames.rechunk((100, H, W))  # Adjust based on available memory

        if self.crop: 
            crop_y_start, crop_x_start, crop_y_end, crop_x_end = self.crop_region
            cropped_motion_energy_frames = motion_energy_frames[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            H, W = crop_y_end - crop_y_start, crop_x_end - crop_x_start
            cropped_motion_energy_frames = cropped_motion_energy_frames.rechunk((100, H, W))
            print('cropping is done.') 

        print('Motion Energy frames are done.')

        ### Construct path where to save data ###
        self.video_metadata['data_asset_name']='test'
        top_zarr_folder = utils.construct_zarr_folder(self.video_metadata)
        #top_zarr_path = os.path.join(utils.get_results_folder(pipeline=True), top_zarr_folder)
        top_zarr_path = os.path.join("/results/", top_zarr_folder)
        if os.path.exists(top_zarr_path) is False:
            os.makedirs(top_zarr_path)

        ### Compute trace and save it to the object ###
        sum_trace = motion_energy_frames.sum(axis=(1, 2)).compute().reshape(-1, 1)
        self.full_frame_motion_energy_sum = sum_trace
        if self.crop:
            self.cropped_frame_motion_energy_sum = \
            cropped_motion_energy_frames.sum(axis=(1, 2)).compute().reshape(-1, 1)
        else:
            self.cropped_frame_motion_energy_sum = np.full(len(sum_trace), np.nan).reshape(-1, 1)

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
        elif isinstance(obj, PosixPath):
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

        

