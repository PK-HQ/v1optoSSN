import numpy as np
from scipy.signal import convolve2d

class SSNModel:
    def __init__(self, orientation_map, size=(64, 64), sigma_e=8.0, sigma_i=4.0, alpha=1.0, n=2):
        """
        Initialize the SSN model.

        Parameters:
        - orientation_map: 2D numpy array of orientation preferences.
        - size: tuple, default (64, 64). Size of the neuron grid.
        - sigma_e: float. Spread (std dev) of excitatory connections.
        - sigma_i: float. Spread (std dev) of inhibitory connections.
        - alpha: float. Scaling factor for orientation similarity.
        - n: float. Exponent for the supralinear transfer function.
        """
        self.orientation_map = orientation_map
        self.size = size
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.alpha = alpha
        self.n = n

        # Initialize connectivity matrices
        self.exc_conn, self.inh_conn = self.create_ssn_connectivity()

    def create_ssn_connectivity(self):
        """
        Create the SSN connectivity matrix with orientation similarity weighting.

        Returns:
        - excitatory_connectivity: 2D numpy array.
        - inhibitory_connectivity: 2D numpy array.
        """
        size_x, size_y = self.orientation_map.shape
        x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))
        center = (size_x // 2, size_y // 2)

        # Compute distances from the center
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)

        # Compute orientation similarity using a sinusoidal function
        orientation_diff = np.abs(self.orientation_map - self.orientation_map[center[0], center[1]])
        orientation_similarity = np.cos(np.deg2rad(orientation_diff))**2  # Squared cosine

        # Excitatory and inhibitory kernels
        excitatory_kernel = np.exp(-distances**2 / (2 * self.sigma_e**2)) * orientation_similarity
        inhibitory_kernel = np.exp(-distances**2 / (2 * self.sigma_i**2)) * orientation_similarity

        # Normalize kernels
        excitatory_kernel /= np.sum(excitatory_kernel)
        inhibitory_kernel /= np.sum(inhibitory_kernel)

        return excitatory_kernel, inhibitory_kernel

    def supralinear_transfer_function(self, input_current):
        """
        Supralinear transfer function for neuronal response.

        Parameters:
        - input_current: 2D numpy array of input currents.

        Returns:
        - firing_rate: 2D numpy array of firing rates.
        """
        return np.maximum(0, input_current)**self.n

    def apply_opsin_map(self, opto_input, opsin_map):
        """
        Scale optogenetic input using the opsin map.

        Parameters:
        - opto_input: 2D numpy array of optogenetic input.
        - opsin_map: 2D numpy array of opsin expression levels.

        Returns:
        - Scaled optogenetic input.
        """
        return opto_input * opsin_map

    def run_trial(self, visual_stim, opto_stim, trial_duration=1.2, bin_size=0.05, opsin_map=None):
        """
        Run a single trial of the SSN model.

        Parameters:
        - visual_stim: VisualStimulus object containing stimulus parameters.
        - opto_stim: OptogeneticStimulus object containing optostim parameters.
        - trial_duration: float, default 1.2. Duration of the trial in seconds.
        - bin_size: float, default 0.05. Temporal resolution of the trial (in seconds).
        - opsin_map: 2D numpy array. Map of opsin expression.

        Returns:
        - 3D numpy array of firing rates (grid x grid x time_bins).
        """
        time_bins = int(trial_duration / bin_size)
        activity = np.zeros((self.size[0], self.size[1], time_bins))

        # Generate inputs
        visual_input = visual_stim.generate_input(self.size)
        opto_input = opto_stim.generate_input(self.size, self.orientation_map)

        # Apply opsin map to optogenetic input
        if opsin_map is not None:
            opto_input = self.apply_opsin_map(opto_input, opsin_map)

        # Initialize activity
        current_activity = np.zeros((self.size[0], self.size[1]))

        for t in range(time_bins):
            # Total input = external input + recurrent input
            recurrent_exc = self.convolve_with_kernel(current_activity, self.exc_conn)
            recurrent_inh = self.convolve_with_kernel(current_activity, self.inh_conn)
            total_input = visual_input + opto_input + recurrent_exc - recurrent_inh

            # Update activity using the supralinear transfer function
            current_activity = self.supralinear_transfer_function(total_input)

            # Store activity for this time bin
            activity[:, :, t] = current_activity

        return activity

    def convolve_with_kernel(self, activity, kernel):
        """
        Efficiently convolve the activity map with the given kernel.

        Parameters:
        - activity: 2D numpy array of current activity.
        - kernel: 2D numpy array representing the connectivity kernel.

        Returns:
        - convolved_activity: 2D numpy array of the convolved activity.
        """
        return convolve2d(activity, kernel, mode='same', boundary='wrap')
