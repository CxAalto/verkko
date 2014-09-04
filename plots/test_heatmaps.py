from heatmaps import plot_counts
import os
import numpy as np
from matplotlib import pyplot as plt


def test():
    n_points = 10000
    x_vals = np.exp(np.random.randn(n_points))
    y_vals = np.exp(np.random.randn(n_points))

    fig, ax, cbar, im = plot_counts(x_vals, y_vals)

    # save fig to gallery

    fname = os.path.join(os.path.dirname(__file__),
                         "gallery/counts_example.pdf")
    plt.show()
    # plt.savefig(fname, format='pdf')

if __name__ == "__main__":
    test()
