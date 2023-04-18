import importlib
import TensegrityStruct as TS
importlib.reload(TS)
import numpy as np


class FreeStandingStruct(TS.TensegrityStruct):
    def __init__(self, num_of_nodes, nodes, masses, cables, bars, k, c, bar_density, penalty, num_of_fixed_nodes=0):
        TS.TensegrityStruct.__init__(self, num_of_nodes, num_of_fixed_nodes, nodes, masses, cables, bars, k, c, bar_density)

        self.penalty = penalty
    def E(self):
        """
        Energy function with a quadratic penalty term
        """
        x3 = self.nodes[:,-1]
        c_min = np.minimum(x3, np.zeros(self.num_of_nodes))
        return super().E()+self.penalty/2*np.dot(c_min, c_min)
    """
    def update_nodes(self, new_X):
        self.nodes[0, 2] = new_X[2]
        self.nodes[1:, ] = np.reshape(new_X[3:], (self.num_of_nodes-1, 3))
        self.X[2:] = new_X[2:]

    """

    def gradient(self):
        print(super().gradient())
        penalty_grad = np.zeros(self.X.size) #additional gradient term
        x3 = self.nodes[:,-1]
        c_min = np.minimum(x3, np.zeros(self.num_of_nodes))
        penalty_grad[2::3] = self.penalty*c_min
        #print(penalty_grad)
        grad = super().gradient() + penalty_grad
        grad[:2]=0
        return grad
