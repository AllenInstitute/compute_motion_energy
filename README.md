# Motion Energy Analysis Capsule

This repository a second step in Behavior Video QC a pipeline to analyze motion energy from grayscale video frames stored in Zarr format. The script loads frames, computes motion energy, applies cropping if specified, and saves the results as a sample video and Zarr files.

## Features  
- Loads grayscale video frames from Zarr format.  
- Computes motion energy as the absolute difference between consecutive frames.  
- Supports optional cropping for region-based motion analysis. Crop region is specified in `utils.get_crop_region()`
- Saves motion energy frames in zarr format and computed sum motion energy trace in .npz file.  
- Saves metadata and object attributes as a dictionary.  
- Tracks processing time.  

## Prerequisites  

Ensure you have the following dependencies installed:  

```bash
pip install tqdm numpy zarr dask
```
Additionally, make sure `utils.py` is available in your project.

## Usage  

### Running the script  
Execute the script using:  
```bash
python run_capsule.py
```

### Parameters  

| Parameter      | Description                                                | Default
|--------------|------------------------------------------------------------|------------------------
| `zarr_paths`  | List of paths to Zarr directories containing video frames. | utils.get_zarr_paths()
| `crop`        | Boolean flag to enable cropping of motion energy analysis. | True

### Modifying Parameters  

To analyze different Zarr datasets, modify `zarr_paths` in `run_capsule.py`:  

```python
zarr_paths = ['/custom/path/video1.zarr', '/custom/path/video2.zarr']
```

To enable or disable cropping, set the crop parameter in the MotionEnergyAnalyzer class. Crop region can be specified in utils:

```
me_analyser = MotionEnergyAnalyzer(zarr_path, crop=True)
```
