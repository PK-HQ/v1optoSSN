# config.py
"""
Configuration file for model2: Includes settings for SSN and ConvLSTM models, input stimuli,
paths to datasets, and evaluation metrics.
"""

import os

# General Settings
NETWORK_SIZE = (512, 512)  # Size of the simulated neuron grid
TRIAL_DURATION = 1.2  # Trial duration in seconds
BIN_SIZE = 0.05  # Time bin size in seconds

# SSN Model Parameters
SSN_PARAMS = {
    "e_ratio": 0.8,  # Proportion of excitatory neurons
    "connectivity_sigma": 8.0,  # Standard deviation of Gaussian connectivity (mm)
    "alpha": 2.0,  # Supralinear power-law exponent
}

# ConvLSTM Model Parameters
CONVLSTM_PARAMS = {
    "input_size": (2048, 2048),  # Input dimensions for V1 activity (pixels)
    "hidden_units": [32, 64],  # ConvLSTM hidden units for each layer
    "kernel_size": (3, 3),  # Convolution kernel size
    "num_frames": 24,  # Number of time frames per trial
    "output_size": (2048, 2048),  # Output dimensions for predicted activity (pixels)
}

# Stimulus Parameters
VISUAL_STIMULUS_DEFAULTS = {
    "orientation": 0,  # Orientation of the Gabor stimulus (degrees)
    "contrast_levels": [0.25, 0.5, 0.75, 1.0],  # Gabor contrast levels
    "spatial_frequency": 2,  # Cycles per degree
    "size": 2,  # Size of the Gabor stimulus (degrees)
    "eccentricity": 4.3,  # Eccentricity from the fixation point (degrees)
    "onset": 0.2,  # Onset time of the visual stimulus (seconds)
}

OPTOGENETIC_STIMULUS_DEFAULTS = {
    "column_tuning": 0,  # Orientation tuning of stimulated columns
    "num_columns": [5, 10, 15, 20, 35],  # Number of optogenetically stimulated columns
    "column_area": 50,  # Area of stimulated columns (pixels)
    "power": 1.0,  # Power of the optogenetic stimulation
    "onset_offset": 0.041,  # Time offset from visual onset (seconds)
    "pulses": 6,  # Number of stimulation pulses
}

# Dataset Paths
DATA_PATHS = {
    "orientation_map": os.path.join("data", "orientation_maps", "default_orientation_map.npy"),
    "opsin_map": os.path.join("data", "expression_maps", "opsin_map.npy"),
    "gcamp_map": os.path.join("data", "expression_maps", "gcamp_map.npy"),
    "output_dir": os.path.join("results", "trial_data"),
}

# Evaluation Metrics
EVALUATION_METRICS = {
    "spatial_similarity": "Pearson",  # Options: "Pearson", "SSIM"
    "target_similarity": 0.8,  # Minimum similarity threshold for success
}

# Debugging and Logging
DEBUG_MODE = False  # Enable/Disable debugging mode for detailed output
LOGGING_LEVEL = "INFO"  # Logging levels: DEBUG, INFO, WARNING, ERROR

# Derived Settings (Computed dynamically)
TIME_BINS = int(TRIAL_DURATION / BIN_SIZE)  # Total number of time bins
