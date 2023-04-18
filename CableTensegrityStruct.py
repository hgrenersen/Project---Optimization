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

    def __init__(self, num_of_nodes, num_of_fixed_nodes, nodes, masses, cables, k):
       ### skal den docstringen være med her også?
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
       #Føler vi må ha dette:
       self.num_of_fixed = num_of_fixed_nodes
       self.nodes = nodes
       self.masses = masses 
       self.cables = cables
       self.k = k
    
       self.num_of_nodes = num_of_nodes
       self.X = np.ravel(nodes) #Vector form
       self.num_of_free_nodes = num_of_nodes-num_of_fixed_nodes

    def E_ext(self):
        """
        Calculates the gravitational potential energy of
        the external loads in the structure
        """
        mass_indices = self.masses[:, 0].astype(np.int64)
        return np.dot(self.masses[:, 1],self.nodes[mass_indices, 2])


    def E_cable_elast(self):
        """
        Calculates the sum of the elastic energies of the
        cables in the structure.
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
        """
        return self.E_ext()+self.E_cable_elast()

    def gradient(self):
        """
        Calculates the gradient of our objective function
        and returns a 1D ndarray of length 3*the number of nodes
        """
        grad = np.zeros((self.num_of_nodes, 3)) 
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
                dist = np.linalg.norm(node_j - node_i)
                
                if dist > rest_length:
                    grad_node-=self.k/rest_length**2*(node_j-node_i)*(1-rest_length/dist)
    
            grad_node[-1] += self.masses[self.masses[:,0] == node_index, 1]   
    
            grad[node_index,:] = grad_node
        grad = grad.ravel()
        return grad
    
    def update_nodes(self, new_X):
        self.nodes = np.reshape(new_X, (self.num_of_nodes, 3))
        self.X = new_X
    
    def plot(self):
        """
        Returns a plot of the structure
        """
        fig= plt.figure()
        ax = fig.add_subplot(projection='3d')
        # Set the axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        points = self.nodes
        for point in points:
            ax.scatter(point[0], point[1], point[2])
        title="The first " +str( self.num_of_fixed) + " nodes are fixed" 
        plt.legend(points,bbox_to_anchor = (1 , 1), title=title)
        cable_indices = self.cables[:, (0, 1)]
        for point in cable_indices:
            ax.plot(points[point, 0], points[point, 1], points[point, 2],"--", color="green")
        return fig, ax

    def animate(self):
        """
        Uses the plot() function in order to animate the structure
        providing a full 3d-overview of the structure
        """
        fig, ax = self.plot()
        
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.005)
            
    def textbox(self):
        """
        Same as the plot function, but with text for our 
        parameters
        """
        fig, ax = self.plot()
        textstr = "Our structure has the following connections and parameters\n"
        i=0
        for cable in self.cables:
            #textstr+=r"Node " + str(cable[0]+1) +" is connected to node " + str(cable[1]+1) +\
            #      " by a cable with resting length " + str(cable[2]) + "\n"
            textstr += r"$l_{%s %s}$: %s "%(cable[0]+1, cable[1]+1, cable[2])+"   "
        textstr+= "k = " +str(self.k)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, 0.5, s=textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
        return fig, ax
