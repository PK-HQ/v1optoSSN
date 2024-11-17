# utils/gcamp_conversion.py

import numpy as np

def convert_to_gcamp(spiking_data, expression_map):
    """
    Converts spiking activity to GCaMP fluorescence using an expression map.

    Parameters:
    - spiking_data: 3D array of spiking activity over time (e.g., 512x512x time_bins)
    - expression_map: 2D array indicating GCaMP expression level per neuron

    Returns:
    - 3D array of GCaMP fluorescence data scaled by expression_map
    """
    fluorescence = spiking_data * expression_map[..., np.newaxis]
    return fluorescence
