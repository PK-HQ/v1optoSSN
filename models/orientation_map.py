# models/orientation_map.py

import os
import numpy as np
from scipy.io import loadmat

class OrientationMap:
    """
    Class to handle loading or generating orientation maps for V1 neurons.
    """

    def __init__(self, filepath=None, directory="data/orientation_maps", size=(512, 512)):
        """
        Initialize the OrientationMap.

        Parameters:
        - filepath: str, optional. Path to a specific .mat file. If None, searches for files containing 'Orientation'.
        - directory: str, default "data/orientation_maps". Directory to search for files if filepath is not provided.
        - size: tuple, default (512, 512). The size of the orientation map.
        """
        self.size = size
        self.filepath = filepath
        self.directory = directory
        self.map = self.load_orientation_map()

    def find_orientation_file(self):
        """
        Search for a file containing 'Orientation' in the directory.

        Returns:
        - Full path to the matching file.
        """
        try:
            file_list = os.listdir(self.directory)
            orientation_files = [f for f in file_list if "Orientation" in f and f.endswith(".mat")]
            if not orientation_files:
                raise FileNotFoundError(f"No files containing 'Orientation' found in {self.directory}.")
            if len(orientation_files) > 1:
                print(f"Warning: Multiple files found. Using the first: {orientation_files[0]}")
            return os.path.join(self.directory, orientation_files[0])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error accessing directory: {self.directory}") from e

    def load_orientation_map(self):
        """
        Load the orientation map from a file or generate a placeholder if no file is found.

        Returns:
        - numpy array of orientation preferences in degrees (0-180°).
        """
        # Dynamically find the file if no filepath is provided
        if not self.filepath:
            self.filepath = self.find_orientation_file()

        # Load the .mat file
        try:
            mat_data = loadmat(self.filepath)
            if "MapOrt" not in mat_data:
                raise ValueError(f"The .mat file {self.filepath} must contain a variable named 'MapOrt'.")
            orientation_map = mat_data["MapOrt"]

            # Check dimensions
            if orientation_map.shape != self.size:
                raise ValueError(f"Orientation map size mismatch: {orientation_map.shape} vs expected {self.size}.")

            return orientation_map.astype(float)
        except Exception as e:
            raise RuntimeError(f"Error loading orientation map from {self.filepath}: {e}")
