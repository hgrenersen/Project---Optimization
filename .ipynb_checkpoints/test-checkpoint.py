#importing libraries
import numpy as np
import copy
import matplotlib.pyplot as plt
import importlib
from PIL import Image
#import imageio
import io

#importing .py files and reloading
import CableTensegrityStruct as CTS
importlib.reload(CTS)
import TensegrityStruct as TS
importlib.reload(TS)
import optimization as opt
importlib.reload(opt)
import plotting
importlib.reload(plotting)
import FreeStandingStruct as FSS
importlib.reload(FSS)

L=2

R = L/np.sqrt(3)

h=10

def make_equilateral_triangle(R, theta, z):
    P1 = np.array([R*np.cos(theta), R*np.sin(theta), z])
    P2 = np.array([R*np.cos(theta+2*np.pi/3), R*np.sin(theta+2*np.pi/3), z])
    P3 = np.array([R*np.cos(theta+4*np.pi/3), R*np.sin(theta+4*np.pi/3), z])
    return P1, P2, P3

P1, P2, P3 = make_equilateral_triangle(R, 0, 0)

P7, P8, P9 = make_equilateral_triangle(R, np.pi, h)
print(np.append(P1[:2],h))
nodes = np.array([P1, P2, P3, 
                  [0, 0, 5],
                   [0, 0, 1],
                   P7, P8, P9,
                   np.append(P1[:2],h),
                   np.append(P2[:2],h),
                   np.append(P3[:2],h),
                   ])

cables=np.array([[0, 1, 3],
                 [0,2,3],
                 [1,2,3],
                [3, 4, 3],

                [6, 8, 3], 
                [6, 10, 3],

                [5, 10, 3],
                [5, 9, 3],

                [7, 8,3],
                [7, 9, 3]
                
                ])

bars = np.array([[0, 3, 2],
                 [4, 5, 2],
                 [0, 8, 2],
                 [1, 9, 2],
                 [2, 10, 2]
                ])

k=0.5
c=1
bar_density=0.1
masses=np.array([[]])


new_struct2 = FSS.FreeStandingStruct(nodes, masses, cables, bars, k, c, bar_density, 100)
#plotting.plot_structure(new_struct2)
#plt.show()
tolerances=np.array([1e-8, 1e-10, 1e-12])
#opt.quadratic_penalty_method(new_struct2, 1000, tolerances, maxiter_BFGS=2000)
#plotting.plot_structure(new_struct2)
#plt.show()
print(new_struct2.gradient())
print(new_struct2.nodes)








s = 0.70970
t = 9.54287

num_of_fixed_nodes = 4

nodes = np.array([[ 1,  1, 0],
                  [-1,  1, 0],
                  [-1, -1, 0],
                  [ 1, -1, 0],
                  [-s+0.1,  0+0.1, t+0.1],
                  [ 0-0.1, -s-0.1, t-0.1],
                  [ s+0.1,  0+0.1, t+0.1],
                  [ 0-0.1,  s-0.1, t-0.1]])

nodes = np.array([[ 1,  1, 0],
                  [-1,  1, 0],
                  [-1, -1, 0],
                  [ 1, -1, 0],
                  [ 5,  5, 10],
                  [-5,  5, 10],
                  [-5, -5, 10],
                  [ 5, -5, 10],
                  [0, 0, 15],
                  [0, 0, 2]
                  ]).astype(np.float64)

cables = np.array([[0, 7, 8],
                   [1, 4, 8],
                   [2, 5, 8],
                   [3, 6, 8],
                   [4, 5, 1],
                   [4, 7, 1],
                   [5, 6, 1],
                   [6, 7, 1],
                   [0, 1, 1.5],
                   [0, 3, 1.5],
                   [1, 2, 1.5],
                   [2, 3, 1.5], 
                   [4, 9, 2],
                   [5, 9, 2],
                   [6, 9, 2],
                   [7, 9, 2]
                   ])

bars = np.array([[0, 4, 10],
                 [1, 5, 10],
                 [2, 6, 10],
                 [3, 7, 10],
                 [4, 8, 5],
                 [5, 8, 5],
                 [6, 8, 5],
                 [7, 8, 5]])

masses = np.array([])

c=1
k=1
bar_density = 1e-4

new_struct2 = FSS.FreeStandingStruct(nodes, masses, cables, bars, k, c, bar_density, 100)
plotting.plot_structure(new_struct2)
plt.show()

tolerances=np.array([1e-7, 1e-9, 1e-11])
opt.quadratic_penalty_method(new_struct2, 1000, tolerances, maxiter_BFGS=500)


plotting.plot_structure(new_struct2)
plt.show()
#new_struct2.penalty=0
print((new_struct2.gradient()))

print("løk")