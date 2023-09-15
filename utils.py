import numpy as np
import matplotlib.pyplot as plt
import config
import copy
import networkx as nx
import itertools
from scipy.spatial import Delaunay


def normalize_data_matrix(dat):
    """
        Parameters
        ------------
        dat: ndarray
                Each row should represent a vector in dat.shape[1]-dimensional space

        Returns
        ---------
        out:  ndarray
                array with same size as dat where each row has euclidean norm equal to 1
        """

    L2 = np.sqrt(np.sum(dat ** 2, 1))[:, None]
    out = dat / L2
    out[np.where(np.isnan(out))] = 0

    return out


def norm_of_matrix_rows(matrix):
    """
  Compute the norm of each row in a matrix.

  Args:
    matrix: The matrix to compute the norm of.

  Returns:
    A matrix of norms, where each row is the norm of the corresponding row in the input matrix.
  """

    norms = np.linalg.norm(matrix, axis=1)
    return norms


def draw_plot_2D(u, colors, sizes, t=1, save_path=f"{config.MOVIE_IMAGE_DIR}"):
    b = 9
    fig = plt.figure(figsize=(b, b))

    ax = fig.add_subplot(111)
    ax.scatter(u[:, 0], u[:, 1], c=colors, s=sizes)  # plots all points up to current
    ax.scatter(0, 0, color='black', s=sizes[0])  # plots all points up to current

    # plt.title(f'{project}', fontsize=18)

    ax.set_facecolor("white")

    a = 6
    plt.xlim([-a, a])
    plt.ylim([-a, a])
    plt.gca().set_aspect('equal')  # not supported in 3d
    #     plt.pause(.01)

    plt.savefig(f'{save_path}/{t:04d}.png', bbox_inches='tight', pad_inches=0, dpi=20)
    fig.clear(keep_observers=True)


def azimuth(f):
    """
        Calculates the azimuth angle of the camera, given the current fraction of the total number of particles.

        Args:
            f: The current fraction of the total number of particles, in [0, 1].

        Returns:
            The azimuth angle of the camera, in degrees.
        """
    return f * 360


def elevation(f):
    """
        Calculates the elevation angle of the camera, given the current fraction of the total number of particles.

        Args:
            f: The current fraction of the total number of particles, in [0, 1].

        Returns:
            The azimuth angle of the camera, in degrees.
        """
    return 50  # 45 * np.sin(f * np.pi * 2.5)


def cam_dist(f):
    return 3 + 0 * np.sin(f * np.pi * 4)


def draw_plot_3D(u, colors, sizes, n_iterations, t=1, save_path=f"{config.MOVIE_IMAGE_DIR}"):
    f = t / n_iterations
    b = 9
    fig = plt.figure(figsize=(b, b))

    ax = fig.add_subplot(projection='3d')
    ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=colors, s=sizes)  # plots all points up to current
    ax.scatter(0, 0, 0, color='red', s=sizes[0])  # plots all points up to current

    emb = add_core(u)
    tri = Delaunay(emb, qhull_options='QJ')
    G = convert_triangulation_to_graph(tri)
    plot_links(ax, emb, G)

    # plt.title(f'{project}', fontsize=18)
    ax.set_axis_off()
    ax.set_facecolor("white")

    a = 6
    ax.set_xlim((-a, a))
    ax.set_ylim((-a, a))
    ax.set_zlim((-a, a))

    ax.view_init(elev=elevation(f), azim=azimuth(f))
    ax.dist = cam_dist(f)  # AFFECTS size in frame. distance of camera

    #     plt.pause(.01)

    plt.savefig(f'{save_path}/{t:04d}.png', bbox_inches='tight', pad_inches=0, dpi=50)
    fig.clear(keep_observers=True)


def add_core(dat):
    dat2 = copy.deepcopy(dat)
    return np.vstack((np.zeros(shape=(1, dat.shape[1])), dat2))


def convert_triangulation_to_graph(tri):
    simps = tri.simplices

    G = nx.Graph()
    G.add_nodes_from(list(range(len(tri.points))))

    for j, simplex in enumerate(simps):
        combinations = list(itertools.combinations(simplex,
                                                   2))  # produces a list of edge tuples, which happens to be the right format for G.add_edges_from()
        G.add_edges_from(combinations)

    return G


def plot_links(ax, points, G):
    for edge in G.edges:
        i, j = edge
        line = np.vstack((points[i][None, :], points[j][None, :]))
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=[0.6, 0.8, 1], linewidth=1.9)


def get_neighbor_dists(G, points):
    """returns array where each row is the indices of the neighbors of that vertex, in the graph G"""
    # neibs = []
    for j, node in enumerate(G):
        j_neibs = list(G.neighbors(node))
        # neibs.extend([j_neibs])
        G.nodes[node]['neighbor_dist'] = np.linalg.norm(points[j_neibs] - points[j][None, :], axis=1)
    return G
