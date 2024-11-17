import numpy as np
import matplotlib.pyplot as plt


def plot_orientation_map(orientation_map, save_path):
    """
    Plot the orientation map with HSV colormap.

    Parameters:
    - orientation_map: 2D numpy array of orientation preferences.
    - save_path: str. Path to save the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(orientation_map, cmap='hsv', vmin=0, vmax=180)
    plt.colorbar(label="Orientation Preference (°)")
    plt.title("Orientation Map")
    plt.axis("off")
    plt.savefig(save_path)
    plt.show()  # Add this line to display the plot
    plt.close()


def plot_stimulus(inputs, title, save_path):
    """
    Plot a 2D stimulus input pattern.

    Parameters:
    - inputs: 2D numpy array of stimulus input.
    - title: str. Title of the plot.
    - save_path: str. Path to save the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(inputs, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Input Strength")
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path)
    plt.show()  # Add this line to display the plot
    plt.close()


def plot_condition_grid(trial_data_all_conditions, metadata, contrasts, orientations, column_tunings, save_path):
    """
    Plot a grid of trial activities for all conditions.

    Parameters:
    - trial_data_all_conditions: list of 3D numpy arrays (trial data).
    - metadata: list of dicts. Metadata for each trial.
    - contrasts: list of floats. Contrast levels.
    - orientations: list of floats. Visual stimulus orientations.
    - column_tunings: list of floats. Optogenetic stimulus orientations.
    - save_path: str. Path to save the plot.
    """
    num_rows = len(contrasts)
    num_cols = len(orientations) * len(column_tunings)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    vmin, vmax = 0, np.max([np.mean(trial_data, axis=2) for trial_data in trial_data_all_conditions])

    for i, contrast in enumerate(contrasts):
        for j, orientation in enumerate(orientations):
            for k, tuning in enumerate(column_tunings):
                # Find the trial matching the current condition
                condition_idx = [
                    idx for idx, meta in enumerate(metadata)
                    if meta['contrast'] == contrast and
                       meta['orientation'] == orientation and
                       meta['column_tuning'] == tuning
                ][0]  # Get the first match

                avg_activity = np.mean(trial_data_all_conditions[condition_idx], axis=2)

                ax = axs[i, j * len(column_tunings) + k]
                im = ax.imshow(avg_activity, cmap="hot", vmin=vmin, vmax=vmax)
                ax.set_title(f"C={contrast}, O={orientation}°, T={tuning}°")
                ax.axis("off")

    # Add a shared colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label("Neural Activity")

    plt.savefig(save_path)
    plt.show()  # Add this line to display the plot
    plt.close()
