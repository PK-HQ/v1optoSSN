import numpy as np


class VisualStimulus:
    def __init__(self, orientation, spatial_frequency, contrast, size, onset=0):
        """
        Initialize the visual stimulus.

        Parameters:
        - orientation: float. Orientation of the Gabor stimulus (degrees).
        - spatial_frequency: float. Cycles per degree.
        - contrast: float. Contrast of the stimulus (0 to 1).
        - size: float. Size of the stimulus in degrees.
        - onset: float. Time at which the stimulus is presented.
        """
        self.orientation = orientation
        self.spatial_frequency = spatial_frequency
        self.contrast = contrast
        self.size = size
        self.onset = onset

    def generate_input(self, grid_size):
        """
        Generate a Gabor-like visual input on a 2D grid.

        Parameters:
        - grid_size: tuple. Size of the 2D grid (rows, cols).

        Returns:
        - 2D numpy array of the visual input.
        """
        rows, cols = grid_size
        x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))

        # Rotate coordinates to match orientation
        theta = np.deg2rad(self.orientation)
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        # Generate Gabor
        gabor = np.exp(-(x ** 2 + y ** 2) / (2 * (self.size ** 2))) * np.cos(
            2 * np.pi * self.spatial_frequency * x_theta)

        # Scale by contrast
        return self.contrast * gabor
