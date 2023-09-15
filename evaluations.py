import numpy as np
import matplotlib.pyplot as plt


def plot_neighbor_dists(G):
    heatmap = np.zeros(shape=(40, 40))
    for node in G:
            dists = G.nodes[node]['neighbor_dist']
            heatmap[node, :len(dists)] = dists

    fig, ax = plt.subplots()
    h = ax.imshow(heatmap - 2, cmap="plasma")
    fig.colorbar(h, label="Distance")
    plt.show()
