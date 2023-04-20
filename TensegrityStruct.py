import importlib
import CableTensegrityStruct as CTS
importlib.reload(CTS)
import numpy as np


class TensegrityStruct(CTS.CableTensegrityStruct):
    """
    A class that inherits from CableTensegrityStruct, but also has implementation for bars in the structure.

    ...

    Attributes
    ----------
    bars : ndarray
        Matrix where each row represents a connection between two nodes, with indices for the nodes that are
        connected and the resting length of the bar
    c : float
        Material parameter for the bars
    bar_density : float
        Density of the bars scaled by the acceleration of gravity
    The object also has the properties which other objects of the class CableTensegrityStruct.

    Methods
    -------
    gradient()
        Makes use of CableTensegrityStruct.gradient() and includes the contribution of the bars to the gradient
    E_bar_elast()
        Calculates the sum of the elastic energies stored in the bars
    E_bar_grav()
        Calculates the sum of the gravitational potential energy of the bars
    E()
        Makes use of CableTensegrityStruct.E() and includes the contributions to the energy from the bars

    """
    def __init__(self, num_of_fixed_nodes, nodes, masses, cables, bars, k, c, bar_density):
        """

        bars : ndarray
            Matrix where each row represents a connection between two nodes, with indices for the nodes that are
            connected and the resting length of the bar
        c : float
            Material parameter for the bars
        bar_density : float
            The density of the bars, scaled by the acceleration of gravity
        """
        CTS.CableTensegrityStruct.__init__(self, num_of_fixed_nodes, nodes, masses, cables, k)

        self.bars = bars
        self.bar_density = bar_density
        self.c = c

    def gradient(self):
        """
        Makes use of CableTensegrityStruct.gradient() and includes the contribution of the bars to the gradient.
        """
        #grad = np.zeros((self.num_of_nodes, 3))
        grad = np.zeros((self.num_of_free_nodes,3))
        # The for loop is analogous to the for loop in the gradient for CableTensegrityStruct.gradient()
        # but here we instead consider cables
        for node_index in range(self.num_of_fixed, self.num_of_nodes):
            grad_node = np.zeros(3)
            bars_ij = self.bars[self.bars[:, 0] == node_index]
            bars_ji = self.bars[self.bars[:, 1] == node_index]

            for bar in bars_ij:
                rest_length = bar[2]

                node_i = self.nodes[int(bar[0])]
                node_j = self.nodes[int(bar[1])]
                dist = np.linalg.norm(node_i - node_j)

                grad_node += self.c / rest_length ** 2 * (node_i - node_j) * (1 - rest_length / dist)
                grad_node[-1] += self.bar_density * rest_length / 2
                

            for bar in bars_ji:
                rest_length = bar[2]

                node_i = self.nodes[int(bar[1])]
                node_j = self.nodes[int(bar[0])]
                dist = np.linalg.norm(node_i - node_j)

                grad_node += self.c / rest_length ** 2 * (node_i - node_j) * (1 - rest_length / dist)
                grad_node[-1] += self.bar_density * rest_length / 2

            grad[node_index-self.num_of_fixed, :] = grad_node
        grad = grad.ravel()
        grad += super().gradient()
        return grad

    def E_cable_elast(self):
        """
        Function to calculate the sum of the elastic energies
        in the cables if there are any present.

        """
        if self.cables.size: # Ensures that we have cables present
            return super().E_cable_elast()
        else:
            return 0.

    def E_bar_elast(self):
        """
        Calculates the sum of the elastic energies stored in the bars
        """
        energy = 0.
        for bar in self.bars:
            node1 = self.nodes[int(bar[0])]
            node2 = self.nodes[int(bar[1])]
            dist = np.linalg.norm(node1 - node2)
            rest_length = bar[2]
            

            energy += self.c / (2 * rest_length ** 2) * (dist - rest_length) ** 2
        return energy

    def E_bar_grav(self):
        """
        Calculates the sum of the gravitational potential energy of the bars
        """
        energy = 0.
        for bar in self.bars:
            node1 = self.nodes[int(bar[0])]
            node2 = self.nodes[int(bar[1])]

            rest_length = bar[2]

            energy += rest_length * (node1[2] + node2[2]) * self.bar_density / 2
        return energy

    def E(self):
        """
        Makes use of CableTensegrityStruct.E() and includes 
        the contributions to the energy from the bars
        """
        return super().E() + self.E_bar_elast() + self.E_bar_grav()
