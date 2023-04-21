import importlib
import TensegrityStruct as TS
importlib.reload(TS)
import numpy as np


class FreeStandingStruct(TS.TensegrityStruct):
    """
    A class used to represent free standing structures.

    ...

    Attributes
    ----------
        penalty : Float
            Represents the penalty in the quadratic penalty function
        X : ndarray
            This is now a vector in 3N-2, as described in the project text. Therefore,
            self.X!=np.ravel(self.nodes)

        All other parameters are as in the TensegrityStruct, but the number of fixed nodes is 0 by default.

    Methods
    ----------
        E_cable_e()
            Function to calculate the elastic energy in the cables if there are any present.

        E()
            The object's objective function, i.e. the quadratic penalty function. This was done this way as this is
            really the function we want to minimize for these type of objects, not only its energy. The function
            calculates the structure's energy if the penalty is first set to 0.

        gradient()
            Uses previous implementations of the gradient in order to calculate the gradient of our new objective
            function and also returns a vector with compatible dimensions to X.

        The class also inherits all other methods from the TensegrityStruct class.

    """
    def __init__(self, nodes, masses, cables, bars, k, c, bar_density, penalty):
        """
        :param penalty: Float to represent the penalty in the quadratic penalty function
        """
        TS.TensegrityStruct.__init__(self, 0, nodes, masses, cables, bars, k, c, bar_density)

        self.penalty = penalty # penalty which will be used in the quadratic penalty function
        self.X = self.X[2:] # omitting the two first coordinates, as they are fixed


    def E(self):
        """
        The object's objective function, i.e. the quadratic penalty function. This was done this way as this is
        really the function we want to minimize for these type of objects, not only its energy. The function
        calculates the function's energy if the penalty is first set to 0.
        """
        x3 = self.nodes[:, -1] #third coordinate of every node
        c_min = np.minimum(x3, np.zeros(self.num_of_nodes)) #minimum of 0 and constraint (x3)
        return super().E()+self.penalty/2*np.dot(c_min, c_min)

    def update_nodes(self, new_X):
        """
        Function to update the nodes matrix given a vector new_X of length 3N-2
        """
        self.nodes[0, 2] = new_X[0] #updates the third coordinate of the first node
        self.nodes[1:, :] = np.reshape(new_X[1:], (self.num_of_nodes-1, 3)) #updates the rest of the nodes
        self.X = new_X #updates X


    def gradient(self):
        '''
        Uses previous implementations of the gradient in order to calculate the gradient of our new objective
        function, with dimension 3N-2 because of the two fixed coordinates in the first node
        '''
        penalty_grad = np.zeros(self.X.size) # additional gradient term
        x3 = self.nodes[:, -1] #third coordinates
        c_min = np.minimum(x3, np.zeros(self.num_of_nodes)) #constraint
        penalty_grad[::3] = self.penalty*c_min #gradient of penalty term
        
        #calculate the gradient of E, and add the gradient of the penalt term
        grad = super().gradient()[2:] + penalty_grad

        return grad