import os
import numpy as np
from models.ssn_model import SSNModel
from models.visual_stimulus import VisualStimulus
from models.optogenetic_stimulus import OptogeneticStimulus
from utils.file_loader import load_orientation_map, search_and_load_tif
from utils.plotting import plot_orientation_map, plot_stimulus, plot_condition_grid

# Configuration
orientation_map_path = "data/orientation_maps/M28D20240118R0OrientationP2.mat"  # Path to .mat file for orientation map
opsin_map_dir = "data/expression_maps/"  # Directory containing opsin expression .tif files
gcamp_map_dir = "data/expression_maps/"  # Directory containing GCaMP expression .tif files
output_dir = "results/trial_data/"  # Directory to save trial outputs
stimuli_dir = "results/stimuli/"  # Directory to save stimuli plots
os.makedirs(output_dir, exist_ok=True)
os.makedirs("results/orientation_maps", exist_ok=True)
os.makedirs(stimuli_dir, exist_ok=True)
crop_size=64
# Load orientation map
try:
    orientation_map = load_orientation_map(orientation_map_path, central_region_size=crop_size)
    print("Orientation map successfully loaded.")
except Exception as e:
    print(f"Error loading orientation map: {e}")
    exit(1)

# Load and crop expression maps
try:
    opsin_map = search_and_load_tif(opsin_map_dir, "EX570", central_region_size=crop_size)
    gcamp_map = search_and_load_tif(gcamp_map_dir, "EX480", central_region_size=crop_size)
    print("Expression maps successfully loaded and cropped.")
except Exception as e:
    print(f"Error loading expression maps: {e}")
    exit(1)


# Initialize the SSN model
ssn_model = SSNModel(
    orientation_map=orientation_map,
    size=(64, 64)
)

# Define conditions for visual and optogenetic stimuli
contrasts = [0.2, 0.5, 1.0]  # Example contrast levels
orientations = [0, 90]  # Visual orientations
column_tunings = [0, 90]  # Optogenetic column tunings

# Preallocate storage for trial data and metadata
trial_data_all_conditions = []
metadata = []

# Adjust trial duration to 300 ms and bin size to 100 ms
trial_duration = 0.3  # seconds (300 ms)
bin_size = 0.1  # seconds (100 ms bins instead of 50 ms)

# Run trials for all conditions
for contrast in contrasts:
    for orientation in orientations:
        for tuning in column_tunings:
            # Initialize stimuli (parameters passed to SSNModel)
            visual_stimulus = VisualStimulus(
                orientation=orientation,
                spatial_frequency=2,
                contrast=contrast,
                size=2
            )
            opto_stimulus = OptogeneticStimulus(
                column_tuning=tuning,
                num_columns=10,
                column_area=50,
                power=1.0
            )

            # Generate inputs for visualization
            visual_input = visual_stimulus.generate_input((64, 64))
            opto_input = opto_stimulus.generate_input((64, 64), orientation_map)

            # Plot stimuli for debugging
            plot_stimulus(visual_input,
                          title=f"Visual Stimulus: Contrast={contrast}, Ori={orientation}°",
                          save_path=f"{stimuli_dir}visual_c{contrast}_o{orientation}.png")
            plot_stimulus(opto_input,
                          title=f"Optogenetic Stimulus: Tuning={tuning}°",
                          save_path=f"{stimuli_dir}opto_t{tuning}.png")

            # Run trial with opsin_map during optogenetic stimulation
            trial_data = ssn_model.run_trial(
                visual_stim=visual_stimulus,
                opto_stim=opto_stimulus,
                trial_duration=trial_duration,
                bin_size=bin_size,
                opsin_map=opsin_map  # Pass the opsin map here
            )

            # Append data
            trial_data_all_conditions.append(trial_data)
            metadata.append({
                "contrast": contrast,
                "orientation": orientation,
                "column_tuning": tuning
            })

            print(f"Trial completed: Contrast={contrast}, Orientation={orientation}, Opto={tuning}")

# Save trial data
output_path = os.path.join(output_dir, "output_activity.npy")
np.save(output_path, trial_data_all_conditions)
print(f"Trial data saved successfully to {output_path}")

# Save metadata
metadata_path = os.path.join(output_dir, "metadata.npy")
np.save(metadata_path, metadata)

# Plot orientation map
plot_orientation_map(orientation_map, save_path="results/orientation_maps/orientation_map.png")

# Generate condition grid plot
plot_condition_grid(
    trial_data_all_conditions,
    metadata,
    contrasts,
    orientations,
    column_tunings,
    save_path="results/condition_grid.png"
)

print("All processing complete.")
