import numpy as np

class MotionEnergyAnalyzer:
    def __init__(self):
        self.frame_source = frame_source  # The external object containing 'loaded_frames' and optionally 'cropped_frames'
        self.crop = crop
        self.motion_energy = None
        self.cropped_motion_energy = None
    
    def compute_motion_energy(self, frames):
        """
        Compute motion energy from a set of frames.
        Motion energy is computed as the sum of absolute differences between consecutive frames.
        """
        if len(frames) < 2:
            raise ValueError("At least two frames are required to compute motion energy.")
        
        motion_energy = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i] - frames[i-1])
            motion_energy.append(np.sum(diff))
        
        return np.array(motion_energy)
    
    def analyze(self):
        """
        Analyze motion energy based on the frames in the frame_source object.
        Uses cropped_frames if crop attribute is True.
        """
        if self.crop:
            if not hasattr(self.frame_source, 'cropped_frames'):
                raise AttributeError("The frame_source object must have 'cropped_frames' attribute when crop is True.")
            frames = self.frame_source.cropped_frames
            self.cropped_motion_energy = self.compute_motion_energy(frames)
        
            if not hasattr(self.frame_source, 'loaded_frames'):
                raise AttributeError("The frame_source object must have 'loaded_frames' attribute.")
            frames = self.frame_source.loaded_frames
            self.motion_energy = self.compute_motion_energy(frames)
    
    def get_results(self):
        """
        Return the computed motion energy results.
        """
        return {
            'motion_energy': self.motion_energy,
            'cropped_motion_energy': self.cropped_motion_energy
        }

# Example usage:
# frame_source = SomeExternalObject()
# analyzer = MotionEnergyAnalyzer(frame_source, crop=True)
# analyzer.analyze()
# results = analyzer.get_results()
