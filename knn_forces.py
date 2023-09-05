import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
from utils import normalize_data_matrix, draw_plot_2D, draw_plot_3D, norm_of_matrix_rows, add_core, \
    convert_triangulation_to_graph, get_neighbor_dists
from scipy.spatial import Delaunay

from tqdm import tqdm


class KNNSymmetricForces:
    '''Class for updating point positions to satisfy certain local force conditions. Reaching force equilibrium locally affects the microstructural features of the data distribution.
     It's called KNNsymmetric because the force only acts using info on ones KNN, and "symmetric" because two neighbors will have the exact same equilibrium distance in relation to each other.
     Namely r0 = sizes_to_r0_scale*(size_p + size_d)'''

    def __init__(self, zeta=0.1, K=1, alpha=30, beta=0.1):
        self.zeta = zeta  # learning rate for each point
        self.beta = beta
        self.K = K
        self.alpha = alpha  # steepness of sigma force function

    def force_of_r(self, r):
        return 1 / (1 + np.exp(-self.alpha * (r - 2))) - 0.5

    def update_points(self, dat):
        # The algorithm moves each point towards/away from its K-nearest neighbors. A single loop over all points in V. K=1 is optimal.
        # note this function does not act on the core particle. It assumes dat does not contain it.

        nbrs = NearestNeighbors(n_neighbors=self.K + 1, algorithm='kd_tree').fit(dat)
        distances, indices = nbrs.kneighbors(dat)
        indices = indices[:, 1:]
        distances = distances[:, 1:]

        W = copy.deepcopy(dat)

        for p in range(dat.shape[0]):
            neibs = dat[indices[p]]  # position vectors of neighbors (d) of point p

            p2d = neibs - dat[p][None, :]  # set of vectors pointing from p to its KNN
            p2d_hat = normalize_data_matrix(p2d)  # dat array holding normalized vectors pointing from p to its KNN
            p_hat = normalize_data_matrix(dat[p][None, :])
            # UPDATE RULE: push point p in direction that sums attractive and repulsive forces from ints KNN
            W[p] += np.sum(p2d_hat * self.force_of_r(distances[p])[:, None], 0) * self.zeta - p_hat[0,
                                                                                              :] * self.force_of_r(
                np.linalg.norm(dat[p])) * self.zeta  # add forces to nearest neighbors + force towards the core particle
        return W

    def regularize_simplices(self, dat):
        emb = add_core(dat)
        tri = Delaunay(emb, qhull_options='QJ')
        G = convert_triangulation_to_graph(tri)
        for j, k in G.edges:
            if j > 0 and k > 0:  # we dont modify the core or any links with it since particles already experience a force towards it
                j2k = (emb[k] - emb[j])[None, :]  # k - j points from j to k
                separation = np.linalg.norm(j2k, axis=1)
                # print(f"separation: {separation}")
                j2k_hat = normalize_data_matrix(j2k)[0]
                # print(f'j2khat {np.linalg.norm(j2k_hat)}')
                if separation <= 1.9 or separation >= 2.1:
                    emb[j] += j2k_hat * self.force_of_r(separation) * self.beta
                    emb[k] -= j2k_hat * self.force_of_r(separation) * self.beta
        return emb[1:, :]

    def particle_simulation(self, n_iterations, dat):
        dat_new = copy.deepcopy(dat)
        colors = np.zeros(shape=(len(dat_new), 3))
        sizes = 80 * np.ones(shape=(len(dat_new), 1))

        for t in tqdm(range(n_iterations)):
            dat_new = self.update_points(dat_new)
            # draw_plot_2D(dat_new, colors, sizes, t=t)
            # if t > 450:
            #     draw_plot_3D(dat_new, colors, sizes, n_iterations, t=t)
            if 500 < t < 600:
                dat_new = self.regularize_simplices(
                    dat_new)  # microadjustment for pairs of kissing nodes that differ in separation from distance=2.
                print(t)


        return dat_new
