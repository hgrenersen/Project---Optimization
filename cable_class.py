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
            Number of nodes in the structure
        num_of_fixed_nodes : int
            Number of fixed nodes in the structure
        nodes : ndarray
            Array containing the positions of the nodes in 
            the structure
        masses : ndarray 
            2D array containing in the first column the index 
            of the node to which the load belongs, and in the 
            other column the corresponding weight
        cables : ndarray
            3D array in which for row i, the first column is 
            the index of a certain node, and the second column 
            is the index of the node which it is connected to
            by cable i. The last column contains the resting
            length of the cable. 
        k : int
            Material parameter which affects the elastic energy
       """
       #Føler vi må ha dette:
       self.num_of_fixed = num_of_fixed_nodes
       self.nodes = nodes
       self.masses = masses 
       self.cables = cables
       self.k = k
        #Litt usikker på hvordan dette bør gjøres
       self.num_of_nodes = num_of_nodes
       self.X = np.ravel(nodes)
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
        and returns a vector of length 3*the number of nodes
        """
        grad = np.zeros(3*self.num_of_nodes) #Vector in 3N
        #Below we only iterate over each free node
        for index in range(3*self.num_of_fixed, 3*self.num_of_nodes, 3):

            elasticity = np.zeros(3)
            node_index = index/3
            #Find the nodes with higher indices which the node is connected to
            case1 = np.where(self.cables[:, 0]==node_index) 
            #Find the nodes with lower indices which the node is connected to
            case2 = np.where(self.cables[:, 1]==node_index)

            if np.size(case1):
                for case in np.nditer(case1):
                    connection = self.cables[case]
                    rest_length = connection[2]

                    node1=self.nodes[connection[0]]
                    node2=self.nodes[connection[1]]
                    dist = np.linalg.norm(node1-node2)
                    
                    if dist < rest_length:
                        continue
                    else:
                        elasticity+=self.k*\
                        (node1-node2)/\
                        rest_length**2*(1-rest_length/\
                        (dist))
            if np.size(case2):
                for case in np.nditer(case2):
                    connection = self.cables[case]
                    rest_length = connection[2]

                    node1=self.nodes[connection[0]]
                    node2=self.nodes[connection[1]]
                    dist = np.linalg.norm(node1-node2)

                    if dist<rest_length:
                        continue
                    else:
                        elasticity-=self.k*\
                        (node1-node2)/\
                        rest_length**2*(1-rest_length/\
                        (dist))

            mass_index = np.where(self.masses[:, 0]==node_index)
            if np.size(mass_index):
                elasticity[2]+=self.masses[mass_index, 1]   
                   
            grad[index:index+3]+=elasticity
        return grad
    
    def update_nodes(self, new_X):
        self.nodes = np.reshape(new_X, (self.num_of_nodes, 3))

    def strongWolfe(self,p,
                initial_value, #f(xk)
                initial_descent, #np.inner(grad(xk), pk)
                initial_step_length = 1.0,
                c1 = 1e-2,
                c2 = 0.99,
                max_extrapolation_iterations = 50,
                max_interpolation_iterations = 20,
                rho = 2.0):
        """
        Makes a step for the structure in the BFGS algorithm
        p : int
            The search direction
        initial_value : float
            The value of the objective function at the current point
        initial_descent : float
            The dot product between the gradient and the search direction
        initial_step_length : float
            The step length we use as a starting point, which is later increased or decreased. 1 by default
        c1 : float
        c2 : float
        rho : float
            Expansion factor, 2 by default
        """

        def make_step():
            next_x = self.X+alphaR*p
            self.update_nodes(next_x)
            next_value = self.E()
            next_grad = self.gradient()
            return next_x, next_value, next_grad

        alphaR = initial_step_length
        alphaL = 0.0

        next_x, next_value, next_grad = make_step()
        

        Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
    
        descentR = np.inner(p,next_grad)
        curvatureLow = (descentR >= c2*initial_descent) #We have assumed that pk is indeed a descent
        #direction, so initial_descent is in fact negative
        curvatureHigh = (descentR <= -c2*initial_descent)

        itnr=0
        

        while (itnr < max_extrapolation_iterations and (Armijo and (not curvatureLow))):
            
            # alphaR is a new lower bound for the step length
            # the old upper bound alphaR needs to be replaced with a larger step length
            alphaL = alphaR #Setting the previous upper bound as new lower bound
            alphaR *= rho #Increasing the upper limit by the expansion factor 
            
            # update function value and gradient
            next_x, next_value, next_grad = make_step()
            
            # update the Armijo and Wolfe conditions
            Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
            descentR = np.inner(p,next_grad)
            curvatureLow = (descentR >= c2*initial_descent)
            curvatureHigh = (descentR <= -c2*initial_descent)
        
        # at that point we should have a situation where alphaL is too small
        # and alphaR is either satisfactory or too large
        # (Unless we have stopped because we used too many iterations. There
        # are at the moment no exceptions raised if this is the case.)
        alpha = alphaR
        grad_evals = itnr+1
        itnr=0
        
        # Use bisection in order to find a step length alpha that satisfies
        # all conditions.
        while (itnr < max_interpolation_iterations and (not (Armijo and curvatureLow and curvatureHigh))):
            #While the three conditions don't hold, continue:
           
            itnr+=1
            if (Armijo and (not curvatureLow)): #Same condition as in the above for-loop
                # the step length alpha was still too small
                # replace the former lower bound with alpha
                alphaL = alpha
                
            else:
                # the step length alpha was too large
                # replace the upper bound with alpha
                alphaR = alpha
                
            # choose a new step length as the mean of the new bounds
            alpha = (alphaL+alphaR)/2
            
            # update function value and gradient
            next_x, next_value, next_grad = make_step()
            
            # update the Armijo and Wolfe conditions
            Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
            descentR = np.inner(p,next_grad)
            curvatureLow = (descentR >= c2*initial_descent)
            curvatureHigh = (descentR <= -c2*initial_descent)
            # return the next iterate as well as the function value and gradient there
            # (in order to save time in the outer iteration; we have had to do these
            # computations anyway)
        
        return next_x,next_value,next_grad
    
    def BFGS(self, tol=1e-6):
        """
        Calculates a stable configuration for the structure, using BFGS
        and strong Wolfe conditions
        """
        Hk = np.eye(3*self.num_of_nodes) 
        k=0
        grad_k = self.gradient()
        Xk=self.X
        fk = self.E()
        imgs = []
        while np.linalg.norm(grad_k)>tol and k<10000:
            imgs.append(self.plot()[0])
            
            pk = -Hk@grad_k 

            Xold = Xk 
            old_grad=grad_k 

            Xk, fk, grad_k = self.strongWolfe(pk, fk, np.inner(grad_k, pk))

            self.update_nodes(Xk)
            
            sk = Xk-Xold

            yk = grad_k-old_grad 
            
            rho_k = 1/(np.inner(yk, sk))

            if k==0:
                Hk = Hk*(1/(rho_k*np.inner(yk,yk)))
            z = Hk.dot(yk)
            Hk += -rho_k*(np.outer(sk,z) + np.outer(z,sk)) + rho_k*(rho_k*np.inner(yk,z)+1)*np.outer(sk,sk)
            
            k+=1
        print(k)
        return Xk, imgs



            
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
        plt.legend(points,bbox_to_anchor = (1 , 1))
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
        


nodes = np.array([[5, 5, 0], [-5, 5, 0], [-5, -5, 0], [5, -5, 0], [2, 2, -1.5], [-2, 2, -1.5], [-2, -2, -1.5], [2, -2, -1.5]])
num_of_nodes = 8
num_of_fixed_nodes = 4
cables = np.array([[0, 4,3], [1, 5, 3],
 [2, 6, 3], [3, 7, 3], [4, 5, 3],[4, 7, 3], 
  [5, 6, 3], [6, 7, 3]])
masses = np.array([[4, 1/6], [5, 1/6], [6, 1/6], [7, 1/6]])

struct1 = CableTensegrityStruct(num_of_nodes, num_of_fixed_nodes,
nodes, masses,
cables, 3)
#struct1.animate()
#print(struct1.E_ext())
#print(struct1.E_cable_elast())
#struct1.plot()
#print(masses)
#print(struct1.gradient())
#fig, ax= struct1.textbox()
#plt.legend()
#plt.show()

new_X = np.array([5, 5, 0, -5, 5, 0, -5, -5, 0, 5, -5, 0, -5, 8, 0, 3, -4, 3, 1, 3, -2, 3, -3, 0])

struct1.update_nodes(new_X)
#struct1.animate()
#struct1.textbox()
#plt.show()

final_X, imgs = struct1.BFGS()
struct1.update_nodes(final_X)
#struct1.animate()
#plt.show()

### TESTING


from PIL import Image
import imageio
import io

# create a list of Matplotlib figures

# create a sequence of images
images = []
for fig_i in imgs:
    # convert the Matplotlib figure to an image
    buf = io.BytesIO()
    fig_i.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    images.append(img)

# save the sequence of images as a GIF
imageio.mimsave('figures.gif', images, duration=2)
"""

nodes = np.array([[0, np.sqrt(3)/4, 0], [-1/2, -np.sqrt(3)/4, 0], [1/2, -np.sqrt(3)/4, 0], [2, -4, 5]])
num_of_nodes = 4
num_of_fixed_nodes = 3
cables = np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [1, 2, 1], [1, 3, 1], [2, 3, 1]] )
masses = np.array([[3, 1/6]])

struct1 = CableTensegrityStruct(num_of_nodes, num_of_fixed_nodes,
nodes, masses,
cables, 3)
struct1.plot()
plt.show()
new_X, imgs = struct1.BFGS()
struct1.update_nodes(new_X)
struct1.plot()
plt.show()
print(struct1.gradient())


"""
""" 
Plotting of bar-structures:

bar_indices=self.bars[:, (0, 1)]
        if type(bar_indices) !=None:
            for point in bar_indices:
                ax.plot(points[point, 0], points[point, 1], points[point, 2], color="blue")
"""