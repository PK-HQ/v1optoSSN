import os
import numpy as np
from scipy.io import loadmat
from PIL import Image


def crop_to_central_region(map, central_size):
    """
    Crop a 2D map to its central square region.

    Parameters:
    - map: 2D numpy array. Input map to crop.
    - central_size: int. Size of the central square region to extract.

    Returns:
    - Cropped 2D numpy array of size (central_size, central_size).
    """
    rows, cols = map.shape
    center_row, center_col = rows // 2, cols // 2
    half_size = central_size // 2

    row_start = center_row - half_size
    row_end = center_row + half_size
    col_start = center_col - half_size
    col_end = center_col + half_size

    cropped_map = map[row_start:row_end, col_start:col_end]

    # Verify the cropped region has the correct size
    if cropped_map.shape != (central_size, central_size):
        raise ValueError(f"Cropping failed: expected shape ({central_size}, {central_size}), got {cropped_map.shape}")

    return cropped_map


def load_orientation_map(filepath, central_region_size=512):
    """
    Load the orientation map from a .mat file and extract the central region.

    Parameters:
    - filepath: str. Path to the .mat file containing the 'MapOrt' variable.
    - central_region_size: int, default 512. Size of the central square region to extract.

    Returns:
    - 2D numpy array of the central orientation map region.
    """
    try:
        # Load the .mat file
        mat_data = loadmat(filepath)
        if "MapOrt" not in mat_data:
            raise ValueError("The .mat file must contain the 'MapOrt' variable.")

        # Extract the full orientation map
        orientation_map = mat_data["MapOrt"].astype(float)

        # Crop to the central region
        orientation_map_central = crop_to_central_region(orientation_map, central_region_size)

        return orientation_map_central

    except Exception as e:
        raise ValueError(f"Error loading orientation map from {filepath}: {e}")


def search_and_load_tif(directory, target_substring, central_region_size=None):
    """
    Search for and load a TIF file from a directory containing the target substring in its filename.
    Optionally crop the loaded map to a central square region.

    Parameters:
    - directory: str. Path to the directory containing TIF files.
    - target_substring: str. Substring to search for in the filenames.
    - central_region_size: int, optional. Size of the central square region to crop.

    Returns:
    - 2D numpy array of the loaded (and optionally cropped) TIF image.
    """
    # List all files in the directory
    try:
        file_list = os.listdir(directory)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Directory not found: {directory}") from e

    # Find the file with the target substring
    target_files = [f for f in file_list if target_substring in f and f.endswith(".tif")]
    if not target_files:
        raise FileNotFoundError(f"No TIF files with substring '{target_substring}' found in {directory}.")
    if len(target_files) > 1:
        print(f"Warning: Multiple TIF files with substring '{target_substring}' found. Using the first match.")

    # Load the first matching file
    target_file = os.path.join(directory, target_files[0])
    try:
        with Image.open(target_file) as img:
            tif_array = np.array(img)

        # Optionally crop the central region
        if central_region_size is not None:
            tif_array = crop_to_central_region(tif_array, central_region_size)

        return tif_array

    except Exception as e:
        raise ValueError(f"Error loading TIF file {target_file}: {e}")
