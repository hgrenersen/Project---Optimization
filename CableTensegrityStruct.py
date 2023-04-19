import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
class CableTensegrityStruct:
    """
    A class used to represent our tensegrity structures, 
    containing only cables, and a certain number of fixed nodes.
    ...

    Attributes
    ---------
    num_of_nodes : int
        Number of nodes in the structure
    num_of_fixed : int
        Number of fixed nodes in the structure
    nodes : ndarray
        Array containing the positions of the nodes in 
        the structure
    masses : ndarray 
        2D array containing in the first column the index 
        of the node to which the load belongs, and in the 
        other column the corresponding weight
    cables : ndarray
        ##### kanskje droppe 'in which for row i'?
        3D array in which for row i, the first column is 
        the index of a certain node, and the second column 
        is the index of the node which it is connected to
        by cable i. The last column contains the resting
        length of the cable. 
    k : int
        Material parameter which affects the elastic energy
    X : ndarray
        Vector form of our nodes
    num_of_free_nodes : int
        Number of free nodes in the structure

    Methods
    -------
    E_ext()
        Calculates the gravitational potential energy of
        the external loads in the structure
    E_cable_elast()
        Calculates the sum of the elastic energies of the
        cables in the structure.
    E()
        Calculates the total energy of the structure
    gradient()
        Calculates the gradient of our objective function
        and returns a vector of length 3*the number of nodes
    plot()
        Returns a plot of the structure
    animate()
        Uses the plot() function in order to animate the structure
        providing a full 3d-overview of the structure
    """

    def __init__(self, num_of_fixed_nodes, nodes, masses, cables, k):
       """
        num_of_nodes : int
            total number of nodes
        num_of_fixed_nodes : int
            number of fixed nodes
        nodes : 2D ndarray
            array containing the positions of the nodes
            the columns contain the x, y and z coordinates of the nodes
        masses : 2D ndarray
            first column contains the index of the node,
            second column contains the corresponding weight
        cables : 2D ndarray 
            first column is the index, i, of the current node
            second column is the index, j, connected to node i
            third column is the resting length of the cable
        k : int
            material parameter which affects the elastic energy
       """
       # Føler vi må ha dette:
       self.num_of_fixed = num_of_fixed_nodes
       self.nodes = nodes
       self.masses = masses
       self.cables = cables
       self.k = k
    
       self.num_of_nodes = len(nodes)
       self.X = np.ravel(nodes[num_of_fixed_nodes:]) # Vector form
       self.num_of_free_nodes = self.num_of_nodes-num_of_fixed_nodes


       self.masses = np.zeros(self.num_of_nodes)
       if masses.size:
           for i in range(len(masses)):
               mass = masses[i]
               self.masses[int(mass[0])] = mass[1]

    def E_ext(self):
        """
        Calculates the gravitational potential energy of
        the external loads in the structure

        :return: Gravitational potential energy of all external loads
        """
        #mass_indices = self.masses[:, 0].astype(np.int64)
        #return np.dot(self.masses[:, 1],self.nodes[mass_indices, 2])
        return np.dot(self.masses, self.nodes[:,2])


    def E_cable_elast(self):
        """
        Calculates the sum of the elastic energies of the
        cables in the structure.

        :return: Sum of elastic energies of the cables
        """
        energy=0
        for cable in self.cables:
            node1 = self.nodes[cable[0]]
            node2 = self.nodes[cable[1]]
            dist = np.linalg.norm(node1-node2)

            rest_length = cable[2]

            energy+=self.k/(2*rest_length**2)*(dist-rest_length)**2
        return energy

    def E(self):
        """
        Calculates the total energy of the structure

        :return: Total energy of the structure
        """
        return self.E_ext()+self.E_cable_elast()

    def gradient(self):
        """
        Calculates the gradient of our objective function
        and returns a 1D ndarray of length 3*the number of nodes

        :return: The gradient of the energy function
        """
        #grad = np.zeros((self.num_of_nodes, 3))
        grad = np.zeros((self.num_of_free_nodes, 3))
        for node_index in range(self.num_of_fixed, self.num_of_nodes): #only iterate over each free node
            grad_node = np.zeros(3) #gradient with respect to a single node
            cables_ij = self.cables[self.cables[:, 0]==node_index] #cables e_ij
            cables_ji = self.cables[self.cables[:, 1]==node_index] #cables e_ji
    
            for cable in cables_ij:
                rest_length = cable[2]
    
                node_i = self.nodes[cable[0]]
                node_j = self.nodes[cable[1]]
                dist = np.linalg.norm(node_i - node_j)
    
                if dist > rest_length:
                    grad_node+=self.k/rest_length**2*(node_i-node_j)*(1-rest_length/dist)
    
            for cable in cables_ji:
                rest_length = cable[2]
    
                node_j = self.nodes[cable[0]]
                node_i = self.nodes[cable[1]]
                dist = np.linalg.norm(node_i - node_j)
                
                if dist > rest_length:
                    grad_node+=self.k/rest_length**2*(node_i-node_j)*(1-rest_length/dist)
    
            #grad_node[-1] += self.masses[self.masses[:,0] == node_index, 1]
            grad_node[-1] += self.masses[node_index]

            grad[node_index-self.num_of_fixed, :] = grad_node
        grad = grad.ravel()
        return grad
    
    def update_nodes(self, new_X):
        """
        Function to ensure that the properties of our object are updated properly when we find a new configuration
        :param new_X: Vector of length 3 times the number of nodes
        """
        self.nodes[self.num_of_fixed:,:] = np.reshape(new_X, (self.num_of_free_nodes, 3))
        self.X = new_X
