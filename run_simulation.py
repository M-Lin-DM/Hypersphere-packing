import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
from utils import normalize_data_matrix
from tqdm import tqdm
from knn_forces import KNNSymmetricForces
from utils import norm_of_matrix_rows, add_core, convert_triangulation_to_graph, get_neighbor_dists
from scipy.spatial import Delaunay

n_particles = 24
dim = 4
n_iterations = 1000

points = np.random.rand(n_particles, dim) * 6 - 3
# print(norm_of_matrix_rows(points))

knn = KNNSymmetricForces(zeta=0.2, K=1, alpha=8)

emb = knn.particle_simulation(n_iterations, points)
# print(norm_of_matrix_rows(emb))

emb = add_core(emb)
# # print(emb)
#
tri = Delaunay(emb, qhull_options='QJ')
print(tri.simplices)
print(len(tri.simplices))
#
G = convert_triangulation_to_graph(tri)
# print(G.nodes)
# print(G.edges)
print(G.degree)

G, neibs = get_neighbor_dists(G, emb)

print(neibs)
print(G.nodes.data())


