import importlib
import CableTensegrityStruct as CTS
importlib.reload(CTS)
import numpy as np


class TensegrityStruct(CTS.CableTensegrityStruct):
    def __init__(self, num_of_nodes, num_of_fixed_nodes, nodes, masses, cables, bars, k, c, bar_density):
        CTS.CableTensegrityStruct.__init__(self, num_of_nodes, num_of_fixed_nodes, nodes, masses, cables, k)

        self.bars = bars
        self.bar_density = bar_density
        self.c = c

    def gradient(self):
        """
        Calculates the gradient of our objective function
        and returns a 1D ndarray of length 3*the number of nodes
        """
        grad = np.zeros((self.num_of_nodes, 3))
        for node_index in range(self.num_of_fixed, self.num_of_nodes):  # only iterate over each free node
            grad_node = np.zeros(3)  # gradient with respect to a single node
            bars_ij = self.bars[self.bars[:, 0] == node_index]  # cables e_ij
            bars_ji = self.bars[self.bars[:, 1] == node_index]  # cables e_ji

            for bar in bars_ij:
                rest_length = bar[2]

                node_i = self.nodes[bar[0]]
                node_j = self.nodes[bar[1]]
                dist = np.linalg.norm(node_i - node_j)

                grad_node += self.c / rest_length ** 2 * (node_i - node_j) * (1 - rest_length / dist)
                grad_node[-1] += self.bar_density * rest_length / 2

            for bar in bars_ji:
                rest_length = bar[2]

                node_i = self.nodes[bar[1]]
                node_j = self.nodes[bar[0]]
                dist = np.linalg.norm(node_i - node_j)

                grad_node += self.c / rest_length ** 2 * (node_i - node_j) * (1 - rest_length / dist)
                grad_node[-1] += self.bar_density * rest_length / 2

            grad[node_index, :] = grad_node
        grad = grad.ravel()
        grad += super().gradient()
        return grad

    def E_bar_elast(self):
        """
        Calculates the sum of the elastic energies of the
        bars in the structure.
        """
        energy = 0
        for bar in self.bars:
            node1 = self.nodes[bar[0]]
            node2 = self.nodes[bar[1]]
            dist = np.linalg.norm(node1 - node2)

            rest_length = bar[2]

            energy += self.c / (2 * rest_length ** 2) * (dist - rest_length) ** 2
        return energy

    def E_bar_grav(self):
        """
        Calculates the sum of the gravitational potential energies of 
        the bars in the structure
        """
        energy = 0
        for bar in self.bars:
            node1 = self.nodes[bar[0]]
            node2 = self.nodes[bar[1]]
            dist = np.linalg.norm(node1 - node2)

            rest_length = bar[2]

            energy += rest_length * (node1[2] + node2[2]) * self.bar_density / 2
        return energy

    def E(self):
        return super().E() + self.E_bar_elast() + self.E_bar_grav()

    def plot(self):
        fig, ax = super().plot()
        points = self.nodes
        bar_indices = self.bars[:, (0, 1)]
        for point in bar_indices:
            ax.plot(points[point, 0], points[point, 1], points[point, 2], color="blue")
        return fig, ax
