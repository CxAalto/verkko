import heatmaps as hm
import os
import numpy as np
from matplotlib import pyplot as plt


def test():
    n_points = 10000
    x_vals = np.exp(np.random.randn(n_points))
    y_vals = x_vals * 0 + np.exp(np.random.randn(n_points))

    fig, ax, cbar, im = hm.plot_counts(x_vals, y_vals)
    fname = os.path.join(os.path.dirname(__file__),
                         "gallery/heatmap_counts_example.pdf")
    plt.savefig(fname, format='pdf')

    fig, ax, cbar, im = hm.plot_prob_density(x_vals, y_vals)
    fname = os.path.join(os.path.dirname(__file__),
                         "gallery/heatmap_density_example.pdf")
    plt.savefig(fname, format='pdf')

    fig, ax, cbar, im = hm.plot_conditional_prob_density(x_vals, y_vals)
    fname = os.path.join(os.path.dirname(__file__),
                         "gallery/heatmap_cond_prob_example.pdf")
    plt.savefig(fname, format='pdf')


if __name__ == "__main__":
    test()
