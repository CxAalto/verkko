import scipy.stats
import numpy as np
from verkko.binner import bins
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import LogFormatterMathtext


def plot_counts(x,
                y,
                cmap=cm.get_cmap('hot'),
                xscale='log',
                yscale='log',
                xParam=1.5,
                yParam=1.5,
                weights=None,
                ):
    """
    Plots the counts as a heatmap.

    Parameters
    ----------
    x : list-like (1D numpy array)
        values on x-axis
    y : list-like (1D numpy array)
        values on x-axis
        if xbins and ybins are both given
    cmap : matplotlib.cm
        colormap to use for the plot
    xscale : {"log", "linear", "linlog", ...}, optional
        see binner.binner or binner.bins for more info on the options
    yscale : {"log", "linear", "linlog", ...}, optional
        see binner.binner or binner.bins for more info on the options
    cmap : matplotlib.cm
        defaulting to hot 'hot'
    xParam : varies according to xscale
        if xscale == 'log'
            xParam equals to the multiplier
        if xscale == 'lin'
            xParam equals to the number of bins
    yParam : varies according to yscale
        see xParam
    bins2D : binner.bins (or binner.binner) object
        if this parameter is given, it overrides all other bin settings
    """

    X, Y, binned_data, bin_centers, means = \
        _get_count_data_to_plot(x, y, xscale, yscale, xParam, yParam)

    fig = plt.figure()
    y_low = 0.07
    y_height = 0.85
    ax = fig.add_axes([0.1, y_low, 0.8, y_height])
    im = ax.pcolor(X, Y, binned_data,
                   cmap=cmap, norm=colors.LogNorm()
                   )
    ax.plot(bin_centers, means, "go-")

    if "log" in xscale:
        ax.set_xscale('log')
    if "log" in yscale:
        ax.set_yscale('log')

    cax = fig.add_axes([0.92, y_low, 0.03, y_height])
    cbar = fig.colorbar(im, cax, orientation='vertical',
                        format=LogFormatterMathtext())
    return fig, ax, cbar, im


def _get_count_data_to_plot(x, y, xscale, yscale, xParam, yParam):
    """
    See e.g. function plot_counts for interpreting the inner workings
    of this function.
    """

    if type(x) is not np.ndarray:
        x = np.array(x)
    if type(y) is not np.ndarray:
        y = np.array(y)

    xidx, yidx = bins.get_reasonable_data_indices_for_binning(
        x, y, xscale, yscale)
    min_x, max_x = min(x[xidx]), max(x[xidx])
    min_y, max_y = min(y[yidx]), max(y[yidx])

    bins2D = bins.Bins2D(
        float, min_x, max_x, xscale, xParam,
        float, min_y, max_y, yscale, yParam
    )

    binned_data, _, _ = np.histogram2d(
        x, y, bins=bins2D.bin_limits)

    # binned_data = binned_data.astype(np.float64)
    binned_data = binned_data.T
    binned_data = np.ma.masked_array(binned_data, binned_data == 0)

    X, Y = bins2D.edge_grids
    # compute bin average as a function of x:
    x_bin_means, _, _ = scipy.stats.binned_statistic(
        x, x, statistic='mean', bins=bins2D.xbin_lims
    )
    y_bin_means, _, _ = scipy.stats.binned_statistic(
        x, y, statistic='mean', bins=bins2D.xbin_lims
    )
    return X, Y, binned_data, x_bin_means, y_bin_means
