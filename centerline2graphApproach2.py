#Author: Joyce John
#The vessel mask is read

import random
import nibabel as nib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import vtk_nifit_render
from nilearn import image
from mayavi import mlab
from skimage import measure
import skimage.io
import skimage.morphology as morphology



niftipath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/tempcenterline.nii'
nifti_im_nii = nib.load(niftipath)
nifti_im = np.asarray(nifti_im_nii.get_data())
img = nifti_im

# # do some closing (noise removing)
# d = morphology.ball(2);
# imgBC = morphology.binary_closing(img,selem=d)
#
# # do the skeletonization
# imgSk = morphology.skeletonize_3d(imgBC)
# imgSkT = imgSk.copy()


# print(imgSkT.shape)
print(img.shape)

class Vertex:
    def __init__(self, point, degree=0, edges=None):
        self.point = np.asarray(point)
        self.degree = degree
        self.edges = []
        self.visited = False
        if edges is not None:
            self.edges = edges

    def __str__(self):
        return str(self.point)


class Edge:
    def __init__(self, start, end=None, pixels=None):
        self.start = start
        self.end = end
        self.pixels = []
        if pixels is not None:
            self.pixels = pixels
        self.visited = False


import collections
import networkx as nx


def get_neighbours(p, exclude_p=False, shape=None):

    ndim = len(p)

    # generate an (m, ndims) array containing all strings over the alphabet {0, 1, 2}:
    offset_idx = np.indices((3,) * ndim).reshape(ndim, -1).T

    # use these to index into np.array([-1, 0, 1]) to get offsets
    offsets = np.r_[-1, 0, 1].take(offset_idx)

    # optional: exclude offsets of 0, 0, ..., 0 (i.e. p itself)
    if exclude_p:
        offsets = offsets[np.any(offsets, 1)]

    neighbours = p + offsets    # apply offsets to p

    # optional: exclude out-of-bounds indices
    if shape is not None:
        valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
        neighbours = neighbours[valid]
    print offsets
    return neighbours,offsets

# p = np.transpose(np.nonzero(img))[0]
p = [ 10, 10,  10]
neighbours,offsets = get_neighbours(p,img.shape)
# print img.shape[2]

# def mapping(x):
#     return [x**2,y**2,z**2]
def network_plot_3D(G, angle, save=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()
    print "number"+str(n)
    # Get the maximum number of edges adjacent to a single node
    # edge_max = max([G.degree(i) for i in range(n)])
    # Define color range proportional to number of edges adjacent to a single node
    # colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi,  s=20 + 20 * G.degree(key), edgecolors='k', alpha=0.7) #c=colors[key]

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)
    # Hide the axes
    # ax.set_axis_off()
    # if save is not False:
    #     # plt.savefig("C:\scratch\\data\" +str(angle).zfill(3)+".png")
    #     plt.close('all')
    #     else:
    #     plt.show()
    plt.show()


    return

def network_plot_3Dxxx(G, angle, save=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()
    print "number"+str(n)
    # Get the maximum number of edges adjacent to a single node
    # edge_max = max([G.degree(i) for i in range(n)])
    # Define color range proportional to number of edges adjacent to a single node
    # colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
    # 3D network plot
    # with mlab.style.context(('ggplot')):

    # fig = mlab.figure(figsize=(10, 7))
    # ax = Axes3D(fig)

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    for key, value in pos.items():
        xi = value[0]
        yi = value[1]
        zi = value[2]
        s = 20 + 20 * G.degree(key)

        # Scatter plot
        mlab.points3d(xi, yi, zi, s) #c=colors[key]

    # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
    # Those two points are the extrema of the line to be plotted
    for i, j in enumerate(G.edges()):
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))

        # Plot the connecting lines
        mlab.plot3d(x, y, z,tube_radius=0.025, colormap='Spectral')

    # Set the initial view
    # ax.view_init(30, angle)
    # Hide the axes
    # ax.set_axis_off()
    # if save is not False:
    #     # plt.savefig("C:\scratch\\data\" +str(angle).zfill(3)+".png")
    #     plt.close('all')
    #     else:
    #     plt.show()
    mlab.show()


    return

def display (g):
    # draw all graphs
    G_3d = nx.convert_node_labels_to_integers(g)
    nodeCoord = nx.get_node_attributes(g,'pos')
    print type(nodeCoord)


    # print "label is" + str(G_3d.nodes)
    # G_3d = nx.relabel_nodes(g,mapping)

    # pos = nx.spring_layout(G_3d,dim=3)
    pos = nx.kamada_kawai_layout(G_3d,dim=3)

    xyz = np.array([pos[v] for v in sorted(G_3d)])
    # xyz = np.array([pos[v] for v in G_3d])
    # xyz = np.array([nodeCoord[x] for x in nodeCoord])
    # print "pos"
    # print xyz
    scalars = np.array(list(G_3d.nodes())) + 5
    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.1,
        scale_mode="none",
        colormap="Blues",
        resolution=20,
    )
    pts.mlab_source.dataset.lines = np.array(list(G_3d.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    mlab.show()

def buildTree(img, start=None):
    # copy image since we set visited pixels to black
    img = img.copy()
    img[np.nonzero(img)] = 1

    shape = img.shape
    nWhitePixels = np.sum(img)

    print "shape before Graph building is" + str(shape)

    # neighbor offsets (8 nbors)
    # nbPxOff = np.array([[-1, -1], [-1, 0], [-1, 1],
    #                     [0, -1], [0, 1],
    #                     [1, -1], [1, 0], [1, 1]
    #                     ])
    nbPxOff = offsets
    queue = collections.deque()

    # a list of all graphs extracted from the skeleton
    graphs = []


    blackedPixels = 0
    # we build our graph as long as we have not blacked all white pixels!
    while nWhitePixels != blackedPixels:

        # if start not given: determine the first white pixel
        # if start is None:
        #     it = np.nditer(img, flags=['multi_index'])
        #     while not it[0]:
        #         it.iternext()
        #
        #     start = it.multi_index
        if start is None:
            # if not np.nonzero(img):
            #     print "all pixels blacked"
            #     break;
            start = np.transpose(np.nonzero(img))[0]
            print type(start)
            # it = np.nditer(img, flags=['multi_index'])
            # while not it[0]:
            #     it.iternext()

            # start = it.multi_index

        startV = Vertex(start)
        queue.append(startV)
        print("Start vertex: ", startV.point)

        # set start pixel to False (visited)
        img[startV.point[0], startV.point[1], startV.point[2]] = False
        blackedPixels += 1

        # create a new graph
        G = nx.Graph()

        G.add_node(startV,pos=startV.point)

        # build graph in a breath-first manner by adding
        # new nodes to the right and popping handled nodes to the left in queue
        while len(queue):
            currV = queue[0];  # get current vertex
            print("Current vertex: ", currV.point)

            # check all neigboor pixels
            for nbOff in nbPxOff:

                # pixel index

                # print currV.point
                pxIdx = currV.point + nbOff

                # print currV.point,pxIdx

                if (pxIdx[0] < 0 or pxIdx[0] >= shape[0]) or (pxIdx[1] < 0 or pxIdx[1] >= shape[1]) or (pxIdx[2] < 0 or pxIdx[2] >= shape[2]):
                    continue;  # current neigbor pixel out of image

                if img[pxIdx[0], pxIdx[1], pxIdx[2]]:
                    # print( "nb: ", pxIdx, " white ")
                    # pixel is white
                    newV = Vertex([pxIdx[0], pxIdx[1], pxIdx[2]])

                    # add edge from currV <-> newV
                    G.add_edge(currV, newV, object=Edge(currV, newV))
                    # G.add_edge(newV,currV)

                    # add node newV
                    G.add_node(newV, pos=newV.point)

                    # push vertex to queue
                    queue.append(newV)

                    # set neighbor pixel to black
                    img[pxIdx[0], pxIdx[1], pxIdx[2]] = False
                    blackedPixels += 1

            # pop currV
            queue.popleft()
        # end while

        # empty queue
        # current graph is finished ->store it
        graphs.append(G)
        G_composed = nx.compose_all(graphs)

        # network_plot_3D(G,0)
        # display(G)


        # reset start
        start = None
        # H = nx.compose(H, G)

    # end while
    # mlab.show()

    return graphs, G

def getEndNodes(g):
    return [n for n in nx.nodes(g) if nx.degree(g, n) == 1]
    # return [n for n in nx.nodes_iter(g) if nx.degree(g, n) == 1]


# temp_graph = nx.Graph()
# # set all connectivity
# for i,g in enumerate(graphs):
#     endNodes = getEndNodes(g)
#     print("graph %i: %i end nodes" % (i,len(endNodes)))
#     graphs[i] = {"graph": g, "endNodes":endNodes}
#
#
#
#
# #transform graph- convert pixel graph into a simpler graph where all line pixels are edges
#
def mergeEdges(graph):
    # copy the graph
    g = graph.copy()

    # v0 -----edge 0--- v1 ----edge 1---- v2
    #        pxL0=[]       pxL1=[]           the pixel lists
    #
    # becomes:
    #
    # v0 -----edge 0--- v1 ----edge 1---- v2
    # |_________________________________|
    #               new edge
    #    pxL = pxL0 + [v.point]  + pxL1      the resulting pixel list on the edge
    #
    # an delete the middle one
    # result:
    #
    # v0 --------- new edge ------------ v2
    #
    # where new edge contains all pixels in between!


    # start not at degree 2 nodes
    startNodes = [startN for startN in g.nodes() if nx.degree(g, startN) != 2]

    for v0 in startNodes:

        # start a line traversal from each neighbor
        startNNbs = list(nx.neighbors(g, v0))

        if not len(list(startNNbs)): #Joyce added list bcos of "dict-keyiterator" has no len() error
            continue

        counter = 0
        v1 = startNNbs[counter]  # next nb of v0
        while True:

            if nx.degree(g, v1) == 2:
                # we have a node which has 2 edges = this is a line segement
                # make new edge from the two neighbors
                nbs = list(nx.neighbors(g, v1))

                # if the first neihbor is not n, make it so!
                if nbs[0] != v0:
                    nbs.reverse()

                pxL0 = g[v0][v1]["object"].pixels  # the pixel list of the edge 0
                pxL1 = g[v1][nbs[1]]["object"].pixels  # the pixel list of the edge 1

                # fuse the pixel list from right and left and add our pixel n.point
                g.add_edge(v0, nbs[1],
                           object=Edge(v0, nbs[1], pixels=pxL0 + [v1.point] + pxL1)
                           )

                # delete the node n
                g.remove_node(v1)

                # set v1 to new left node
                v1 = nbs[1]

            else:
                counter += 1
                if counter == len(startNNbs):
                    break;
                v1 = startNNbs[counter]  # next nb of v0

    # weight the edges according to their number of pixels
    for u, v, o in g.edges(data="object"):
        g[u][v]["weight"] = len(o.pixels)
    # network_plot_3D(g, 0)
    # display(g)
    return g

# def line_and_diameter(array):



def connected_component_to_graph(img,wholeGraph):
    s = scipy.ndimage.generate_binary_structure(3, 3)
    label_im, nb_labels = scipy.ndimage.label(img,structure=s)
    unique, counts, indices = np.unique(label_im, return_counts=True, return_index=True)
    print unique, counts, indices
    threshold_labels = []
    # for k in range(0, len(unique)):
    #     if indices[k] > 25:
    #         threshold_labels.append(unique[k])

    # print threshold_labels

    index = np.where(label_im == 3)
    startAt = [x[0] for x in index]
    print startAt
    for each_label in range(0,nb_labels):
        if counts[each_label] != 0:
            print each_label
            each_component = np.where(label_im == unique[each_label], label_im, 0)
            index = np.where(label_im == each_label)
            startAt = [x[0] for x in index]
            print each_component.shape,startAt
            componentGraph, G = buildTree(each_component, start=startAt)
            mergedGraph = mergeEdges(G)
            print "weight and pixels"
            for u, v, o in mergedGraph.edges(data="object"):
                print mergedGraph[u][v]["weight"]
                print o.pixels
            wholeGraph = nx.union(wholeGraph , mergedGraph)

    return wholeGraph




H = nx.Graph()
img[np.nonzero(img)] = 1
# imgSkT = img.copy()

combined_graph = connected_component_to_graph(img,H)
network_plot_3D(combined_graph, 0)
# display(combined_graph)

# graphs , imgB = buildTree(img)
print("built %i graphs" % len(combined_graph))
# display(combined_graph)

def branchPoints (g):
    # start not at degree 2 nodes
    bNodes = np.array([bN.point for bN in g.nodes() if nx.degree(g, bN) > 2])
    pos = nx.get_node_attributes(g, 'pos')
    print "bnodes"
    print bNodes.shape

    # Get number of nodes
    # n = bNodes.number_of_nodes()
    # print "number" + str(n)
    # Get the maximum number of edges adjacent to a single node
    # edge_max = max([G.degree(i) for i in range(n)])
    # Define color range proportional to number of edges adjacent to a single node
    # colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
    # 3D network plot
    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        for i in range(0,bNodes.shape[0]):
            point = bNodes[i,:]
            x = point[0]
            y = point[1]
            z = point[2]
            ax.scatter(x,y,z)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        # for key, value in pos.items():
        #     xi = value[0]
        #     yi = value[1]
        #     zi = value[2]
        #
        #     # Scatter plot
        #     ax.scatter(xi, yi, zi, s=20 + 20 * g.degree(key), edgecolors='k', alpha=0.7)  # c=colors[key]
    plt.show()

branchPoints(combined_graph)


# simpleGraphs = []
#
# for g in graphs:
#     newG = mergeEdges(g["graph"])
#     # display(newG)
#     simpleGraphs.append(
#         {
#             "graph": newG,
#             "endNodes": getEndNodes(newG)
#         }
#     )
# print simpleGraphs
#     # print("merged graph %i: %i end nodes" % (i, len(endNodes)))

# set all connectivity
# for i,g in enumerate(simpleGraphs):
#     # endNodes = getEndNodes(g)
#     print("graph %i: %i end nodes" % (i,len(simpleGraphs[i]["endNodes"])))
#     # simpleGraphs[i] = {"graph": g, "endNodes":endNodes}
#
# for connectivity of vessel centerline
# labels = measure.label(binary_image)

# Define new affine array to write the resampled nifti image
targetaffine4x4 = np.eye(4) * 2
targetaffine4x4[3,3] = 1

nifti_image_from_array = nib.Nifti1Image(img,targetaffine4x4)
save_path = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/tempresample.nii'
nib.save(nifti_image_from_array, save_path)
vtk_nifit_render.vtk_pipeline(save_path)
vtk_nifit_render.vtk_pipeline(niftipath)