import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
from utils import normalize_data_matrix, draw_plot_2D, draw_plot_3D
from tqdm import tqdm


class KNNSymmetricForces:
    '''Class for updating point positions to satisfy certain local force conditions. Reaching force equilibrium locally affects the microstructural features of the data distribution.
     It's called KNNsymmetric because the force only acts using info on ones KNN, and "symmetric" because two neighbors will have the exact same equilibrium distance in relation to each other.
     Namely r0 = sizes_to_r0_scale*(size_p + size_d)'''

    def __init__(self, zeta=0.1, K=1, alpha=30):
        self.zeta = zeta  # learning rate for each point
        self.K = K
        self.alpha = alpha  # steepness of sigma force function

    def force_of_r(self, r):
        return 1 / (1 + np.exp(-self.alpha * (r - 2))) - 0.5

    def update_points(self, dat):
        # The algorithm moves each point towards/away from its K-nearest neighbors. A single loop over all points in V. K=1 is optimal.

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
            W[p] += np.sum(p2d_hat * self.force_of_r(distances[p])[:, None], 0) * self.zeta - p_hat[0, :] * self.force_of_r(np.linalg.norm(dat[p])) * self.zeta  # add forces to nearest neighbors + force towards the core particle
        return W

    def particle_simulation(self, n_iterations, dat):
        dat_new = copy.deepcopy(dat)
        colors = np.zeros(shape=(len(dat_new), 3))
        sizes = 80 * np.ones(shape=(len(dat_new), 1))

        for t in tqdm(range(n_iterations)):
            dat_new = self.update_points(dat_new)
            # draw_plot_2D(dat_new, colors, sizes, t=t)
            # if t > 500:
                # draw_plot_3D(dat_new, colors, sizes, n_iterations, t=t)

        return dat_new
