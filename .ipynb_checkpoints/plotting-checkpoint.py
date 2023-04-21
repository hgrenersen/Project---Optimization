import importlib
import matplotlib.pyplot as plt
import numpy as np
import TensegrityStruct as TS
importlib.reload(TS)
import FreeStandingStruct as FSS
importlib.reload(FSS)

def plot_structure(struct, ax, show_edge_legend=True, show_node_legend=True, show_nodes = False, show_edges = False, show_parameters = False):
    """
    Returns a plot of the structure
    """
    #fig, ax = plt.subplots(projection='3d')
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    # Set the axis labels
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
        cable_indices = struct.cables[:, :-1]
        for point in cable_indices:
            ax.plot(points[point, 0], points[point, 1], points[point, 2],"--", color="green", label="cable")

    if type(struct) == TS.TensegrityStruct or type(struct) == FSS.FreeStandingStruct:
        if struct.bars.size:
            bar_indices = struct.bars[:, :-1]
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

    if show_node_legend:
        node_legend = ax.legend(node_dict.values(), node_dict.keys(), loc = 'upper left', title = "Nodes")
        ax.add_artist(node_legend)
        #ax.legend(node_dict.values(), node_dict.keys(), loc = 'upper left', title = "Nodes")
    if show_edge_legend:
        edge_legend = ax.legend(edge_dict.values(), edge_dict.keys(), loc = 'upper right', title = "Edges")
        ax.add_artist(edge_legend)
        #ax2 = ax.twinx()
      #  ax2.get_yaxis().set_visible(False)
       # ax2.legend(edge_dict.values(), edge_dict.keys(), loc='upper right', title = "Edges")

    #Adding text boxes with information
    if show_nodes:
        nodestr = "Nodes:"
        i = 1
        for node in struct.nodes:
            nodestr += f"\n{i}: ["
            for coord in node:
                if np.abs(coord) < 1e-4 and coord != 0:
                    nodestr += f"{coord:.3e}  ".ljust(10)
                else:
                    nodestr += f"{round(coord,4)}  ".ljust(10)
            nodestr += "]"
            #nodestr += np.array2string(struct.nodes, formatter = {'float_kind':lambda x: "%.3e" % x})
            i+=1
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text2D(1.15, 0.95, nodestr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
    if show_edges:
        edgestr = "Edges:"
        i=0
        if struct.cables.size:
            for cable in struct.cables:
                if i%4==0:
                    edgestr += "\n"
                edgestr += r"  $\ell_{%s %s}$: %s "%(cable[0]+1, cable[1]+1, cable[2])
                i+=1
        if type(struct) == TS.TensegrityStruct or type(struct) == FSS.FreeStandingStruct:
            if struct.bars.size:
                for bar in struct.bars:
                    if i%4==0:
                        edgestr += "\n"
                    edgestr += r"  $\ell_{%s %s}$: %s "%(bar[0]+1, bar[1]+1, bar[2])
                    i+=1
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text2D(1.15, 0.5, edgestr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        

    if show_parameters:
        paramstr = "Parameters:"
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
        # place a text box in upper left in axes coords
        ax.text2D(1.15, 0.2, paramstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

def convergence_plot(norms, ax):
    ax.set_title("Convergence plot")
    ax.set_yscale('log')
    ax.plot(np.arange(norms.size), norms)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Gradient norm")
    ax.grid()

def nodes(struct, ax):
    nodestr = "Nodes:"
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
    ax.text(0.05, 0.95, nodestr, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    ax.axis('off')
    
def edges(struct, ax):
    edgestr = "Edges:"
    i=0
    if struct.cables.size:
        for cable in struct.cables:
            if i%4==0:
                edgestr += "\n"
            edgestr += r"  $\ell_{%s %s}$: %s "%(cable[0]+1, cable[1]+1, cable[2])
            i+=1
    if type(struct) == TS.TensegrityStruct or type(struct) == FSS.FreeStandingStruct:
        if struct.bars.size:
            for bar in struct.bars:
                if i%4==0:
                    edgestr += "\n"
                edgestr += r"  $\ell_{%s %s}$: %s "%(bar[0]+1, bar[1]+1, bar[2])
                i+=1
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, edgestr, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    ax.axis('off')
    
def parameters(struct, ax):
    paramstr = "Parameters:"
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
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, paramstr, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    ax.axis('off')
    