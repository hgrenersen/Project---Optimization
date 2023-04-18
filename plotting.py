import matplotlib.pyplot as plt

def plot(struct, plot_bars = False):
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
        ax.scatter(point[0], point[1], point[2], color = "red")

    for i in range(struct.num_of_free_nodes, struct.num_of_nodes):
        point = points[i]
        ax.scatter(point[0], point[1], point[2], color = "blue")

    title=f"The first {struct.num_of_fixed} nodes are fixed"
    plt.legend(points,bbox_to_anchor = (1 , 1), title=title)
    cable_indices = struct.cables[:, :-1]

    for point in cable_indices:
        ax.plot(points[point, 0], points[point, 1], points[point, 2],"--", color="green")

    if plot_bars:
        bar_indices = struct.bars[:, :-1]
        for point in bar_indices:
            ax.plot(points[point, 0], points[point, 1], points[point, 2],"-", color="hotpink")

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