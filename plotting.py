import importlib
import matplotlib.pyplot as plt
import numpy as np
import TensegrityStruct as TS
importlib.reload(TS)
import FreeStandingStruct2 as FSS
importlib.reload(FSS)

def plot(struct):
    """
    Returns a plot of the structure
    """
    #fig, ax = plt.subplots(projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Set the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    points = struct.nodes
    for i in range(struct.num_of_free_nodes):
        point = points[i]
        ax.scatter(point[0], point[1], point[2], color = "red", label = "fixed")

    for i in range(struct.num_of_free_nodes, struct.num_of_nodes):
        point = points[i]
        ax.scatter(point[0], point[1], point[2], color = "blue", label="free")

    title=f"The first {struct.num_of_fixed} nodes are fixed"
    #plt.legend(np.around(points, decimals = 3), bbox_to_anchor = (1 , 1), title=title)

    if struct.cables.size:
        cable_indices = struct.cables[:, :-1]
        for point in cable_indices:
            ax.plot(points[point, 0], points[point, 1], points[point, 2],"--", color="green", label="cable")

    if type(struct) == TS.TensegrityStruct or type(struct) == FSS.FreeStandingStruct:
        if struct.bars.size:
            bar_indices = struct.bars[:, :-1]
            for point in bar_indices:
                ax.plot(points[point, 0], points[point, 1], points[point, 2],"-", color="hotpink", label="bar")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    return fig, ax

def animate(struct):
    """
    Uses the plot() function in order to animate the structure
    providing a full 3d-overview of the structure
    """
    fig, ax = plot(struct)

    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.005)

def textbox(struct):
    """
    Same as the plot function, but with text for our 
    parameters
    """
    fig, ax = plot(struct)
    textstr = "Our structure has the following connections and parameters\n"
    for cable in struct.cables:
        #textstr+=r"Node " + str(cable[0]+1) +" is connected to node " + str(cable[1]+1) +\
        #      " by a cable with resting length " + str(cable[2]) + "\n"
        textstr += r"$l_{%s %s}$: %s "%(cable[0]+1, cable[1]+1, cable[2])+"   "
    textstr+= "k = " +str(struct.k)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, 0.5, s=textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    return fig, ax

def convergence_plot(norms):
    fig, ax = plt.subplots()
    ax.set_title("Convergence plot")
    ax.set_yscale('log')
    ax.plot(np.arange(norms.size), norms)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Gradient norm")
    return ax

def nodes(struct, ax):
    text = "Nodes:"
    if struct.num_of_fixed:
        text += f"\nThe first {struct.num_of_fixed} are fixed"
    plt.legend(struct.nodes, bbox_to_anchor = (1 , 1), title=text)



