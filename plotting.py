import importlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import TensegrityStruct as TS
importlib.reload(TS)
import FreeStandingStruct as FSS
importlib.reload(FSS)

def plot_structure(struct, ax, title="Tensegrity structure"):
    """
    Plots the structure in a given axes
    """
    # Set the axis labels
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    points = struct.nodes
    for i in range(struct.num_of_fixed):
        point = points[i]
        ax.scatter(point[0], point[1], point[2], color = "red", label = "fixed")
        ax.text(point[0], point[1], point[2]+0.3, i+1, color = "red")

    for i in range(struct.num_of_fixed, struct.num_of_nodes):
        point = points[i]
        ax.scatter(point[0], point[1], point[2], color = "blue", label="free")
        ax.text(point[0], point[1], point[2]+0.3, i+1, color = "blue")

    if struct.cables.size:
        cable_indices = struct.cables[:, :-1].astype(dtype=np.int16)
        for point in cable_indices:
            ax.plot(points[point, 0], points[point, 1], points[point, 2],"--", color="green", label="cable")

    if type(struct) == TS.TensegrityStruct or type(struct) == FSS.FreeStandingStruct:
        if struct.bars.size:
            bar_indices = struct.bars[:, :-1].astype(dtype=np.int16)
            for point in bar_indices:
                ax.plot(points[point, 0], points[point, 1], points[point, 2],"-", color="hotpink", label="bar")

    # Creating legends
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))

    node_dict = {}
    if "fixed" in legend_dict:
        node_dict["fixed"] = legend_dict["fixed"]
    if "free" in legend_dict:
        node_dict["free"] = legend_dict["free"]

    edge_dict = {}
    if "cable" in legend_dict:
        edge_dict["cable"] = legend_dict["cable"]
    if "bar" in legend_dict:
        edge_dict["bar"] = legend_dict["bar"]

    #Adding legends
    node_legend = ax.legend(node_dict.values(), node_dict.keys(), loc = 'upper left', title = "Nodes")
    ax.add_artist(node_legend)

    edge_legend = ax.legend(edge_dict.values(), edge_dict.keys(), loc = 'upper right', title = "Edges")
    ax.add_artist(edge_legend)

def convergence_plot(norms, ax):
    """
    Plots structure in given axes
    """
    ax.set_title("Convergence plot")
    ax.set_yscale('log')
    ax.plot(np.arange(norms.size), norms)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Gradient norm")
    ax.grid()

def nodes(struct, ax, title="Nodes:"):
    """
    Prints nodes in given axes
    """
    nodestr = title
    i = 1
    for node in struct.nodes:
        nodestr += f"\n{i}: ["
        for j in range(3):
            coord = node[j]
            if np.abs(coord) < 1e-4 and coord != 0:
                if j == 2:
                    nodestr += f"{coord:.3e}"
                else:
                    nodestr += f"{coord:.3e},".ljust(10)
            else:
                if j == 2:
                    nodestr += f"{round(coord,4)}"
                else:
                    nodestr += f"{round(coord,4)},".ljust(10)
        nodestr += "]"
        i+=1
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, nodestr, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)
    ax.axis('off')
    
def edges(struct, ax, many_params=False, title="Edges:"):
    """
    Prints edges in given axes
    """
    if many_params:
        linebreak = lambda x : x%2==0
    else:
        linebreak = lambda x : x%3==0
    edgestr = title
    i=0
    if struct.cables.size:
        for cable in struct.cables:
            if linebreak(i):
                edgestr += "\n"
            edgestr += r"  $\ell_{%s %s}$: %s "%(cable[0]+1, cable[1]+1, cable[2])
            i+=1
    if type(struct) == TS.TensegrityStruct or type(struct) == FSS.FreeStandingStruct:
        if struct.bars.size:
            for bar in struct.bars:
                if linebreak(i):
                    edgestr += "\n"
                edgestr += r"  $\ell_{%s %s}$: %s "%(bar[0]+1, bar[1]+1, bar[2])
                i+=1
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, edgestr, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)
    ax.axis('off')
    
def parameters(struct, ax, title="Parameters:"):
    """
    Prints parameters in given axes
    """
    paramstr = title
    if struct.cables.size:
        paramstr += f"\nk = {struct.k}"
    if type(struct) == TS.TensegrityStruct or type(struct) == FSS.FreeStandingStruct:
        if struct.bars.size:
            paramstr += f"\nc = {struct.c}"
            paramstr += "\n" + rf"$\rho g$ = {struct.bar_density}"
    if np.all(struct.masses==0):
        paramstr += "\n"+r"$mg=0$"
    else:
        for i in range(len(struct.masses)):
            if i%4==0:
                paramstr += "\n"
            paramstr += rf"$m_{i+1}g=${round(struct.masses[i],4)}  "

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, paramstr, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)
    ax.axis('off')
    
def main_plot(struct, struct_BFGS, norms, filename="struct.png", many_params=False):
    """
    PLots a figure containing initialization structure, resulting structure, 
    nodes, edges, parameters and a convergence plot
    """
    fig = plt.figure(layout='constrained', figsize=(14, 5)) #creating figure
    subfigs = fig.subfigures(1, 2, width_ratios=[5,3]) #creating subfigs (left and right)
    axs = [0]*7 #to be filled with axes
    gs = GridSpec(1, 2) #gridspec for tensegrity structure plot
    axs[0] = subfigs[0].add_subplot(gs[0,0],projection='3d') #adding 3d axes
    axs[1] = subfigs[0].add_subplot(gs[0,1],projection='3d')

    subfigsRight = subfigs[1].subfigures(2, 1, height_ratios=[3,2]) #right subfigs (up and down)

    if many_params:
        gs = GridSpec(2, 3, height_ratios=[5,3]) #gridspec for info box and convergence plot
        axs[2] = subfigsRight[0].add_subplot(gs[0,0]) #initialized nodes
        axs[3] = subfigsRight[0].add_subplot(gs[0,1]) #resulting nodes
        axs[4] = subfigsRight[0].add_subplot(gs[0,2]) #edges
        axs[5] = subfigsRight[0].add_subplot(gs[1,:]) #parameters
    else:
        gs = GridSpec(2, 2, height_ratios=[5,3]) #gridspec for info box and convergence plot
        axs[2] = subfigsRight[0].add_subplot(gs[0,0]) #initialized nodes
        axs[3] = subfigsRight[0].add_subplot(gs[0,1]) #resulting nodes
        axs[4] = subfigsRight[0].add_subplot(gs[1,0]) #edges
        axs[5] = subfigsRight[0].add_subplot(gs[1,1]) #parameters
        
    axs[6] = subfigsRight[1].add_subplot()

    subfigs[0].set_facecolor('0.9') #setting facecolor
    subfigsRight[0].set_facecolor('0.9')
    subfigsRight[1].set_facecolor('0.9')

    #initialization plot
    plot_structure(struct, axs[0], title="Initialized structure")

    #resulting structure
    plot_structure(struct_BFGS, axs[1], title="Resulting structure")

    #initialization nodes
    nodes(struct, axs[2], title="Initialized nodes:")

    #resulting nodes
    nodes(struct_BFGS, axs[3], title="Resulting nodes:")

    #edges
    edges(struct, axs[4], many_params)

    #paramters
    parameters(struct, axs[5])

    #convergence plot
    convergence_plot(norms, axs[6])

    #saving and showing figure
    plt.savefig(filename)
    plt.show()