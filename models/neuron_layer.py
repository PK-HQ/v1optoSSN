# models/neuron_layer.py

import numpy as np


class NeuronLayer:
    """
    Defines the neuron layer structure, population assignments, and management.

    Attributes:
    - E_population: Boolean mask for excitatory neurons
    - I_population: Boolean mask for inhibitory neurons
    """

    def __init__(self, network_size=(512, 512), e_ratio=0.8):
        self.size = network_size
        self.e_ratio = e_ratio
        self.E_population, self.I_population = self.initialize_neuron_population()

    def initialize_neuron_population(self):
        e_neurons = np.random.rand(*self.size) < self.e_ratio
        i_neurons = ~e_neurons
        return e_neurons, i_neurons
