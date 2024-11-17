import numpy as np

class OptogeneticStimulus:
    def __init__(self, column_tuning, num_columns, column_area, power):
        """
        Initialize the optogenetic stimulus.

        Parameters:
        - column_tuning: float. Orientation preference of stimulated columns (degrees).
        - num_columns: int. Number of columns stimulated.
        - column_area: float. Area of each stimulated column.
        - power: float. Light power in mW.
        """
        self.column_tuning = column_tuning
        self.num_columns = num_columns
        self.column_area = column_area
        self.power = power

    def generate_input(self, grid_size, orientation_map):
        """
        Generate optogenetic stimulation input targeting specific orientation columns.

        Parameters:
        - grid_size: tuple. Size of the 2D grid (rows, cols).
        - orientation_map: 2D numpy array of orientation preferences.

        Returns:
        - 2D numpy array of the optogenetic input.
        """
        rows, cols = grid_size

        # Create a binary mask for the target orientation
        mask = (orientation_map == self.column_tuning)

        # Normalize the mask and scale by the optogenetic power
        opto_input = mask.astype(float) * self.power
        return opto_input / np.max(opto_input) if np.max(opto_input) > 0 else opto_input

