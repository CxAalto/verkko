import numpy as np
import matplotlib.pyplot as plt

from verkko.binner import bins


def plot_pdf(values, ax=None,
             xscale='lin', yscale='lin',
             xParam=None, **kwargs):
    """
    Plots the probability density function of given values.

    Parameters
    ----------

    values : numpy ndarray
        the values for which the experimental pdf is computed
    ax : matplotlib axes object, optional
        axes to plot the figure in
    xscale : str
        'lin' or 'log', or ... (see binner.Bins for more details)
    yscale : str
        'lin' or 'log'
    xParam : different things, optional
        see binner.Bins for more details
    **kwargs : kwargs
        keyword arguments that will be passed to matplotlib hist
        function

    Returns
    -------
    fig : matplotlib Figure
        the parent figure of the axes
    ax : matplotlib Axes object
        the axes in which the pdf is plotted
    """

    if not isinstance(values, np.ndarray):
        values = np.array(values)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if xscale == 'linlog':
        values = np.array(values, dtype=int)
        dataType = int
    else:
        dataType = float


    indices = bins.get_reasonable_data_indices_for_binning(
        values, xscale=xscale)

    prop_vals = values[indices]

    if xParam is None:
        if xscale == 'log':
            xParam = np.sqrt(2)  # just getting a nice factor of two...
        if xscale == 'lin':
            xParam = 50



    xbins = bins.Bins(dataType, np.min(prop_vals),
                      np.max(prop_vals), xscale, xParam)

    ax.hist(values, bins=xbins.bin_limits, normed=True, **kwargs)

    if 'log' in xscale:
        ax.set_xscale('log')
    if 'log' in yscale:
        ax.set_yscale('log')

    return fig, ax

def plot_counts(values, ax=None,
             xscale='lin', yscale='lin',
             xParam=None, label=None, **kwargs):
    """
    Plots the probability density function of given values.

    Parameters
    ----------

    values : numpy ndarray
        the values for which the experimental pdf is computed
    ax : matplotlib axes object, optional
        axes to plot the figure in
    xscale : str
        'lin' or 'log', or ... (see binner.Bins for more details)
    yscale : str
        'lin' or 'log'
    xParam : different things, optional
        see binner.Bins for more details
    **kwargs : kwargs
        keyword arguments that will be passed to matplotlib hist
        function

    Returns
    -------
    fig : matplotlib Figure
        the parent figure of the axes
    ax : matplotlib Axes object
        the axes in which the pdf is plotted
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    indices = bins.get_reasonable_data_indices_for_binning(
        values, xscale=xscale)

    prop_vals = values[indices]

    if xParam is None:
        if xscale == 'log':
            xParam = np.sqrt(2)  # just getting a nice factor of two...
        if xscale == 'lin':
            xParam = 50
    xbins = bins.Bins(float, np.min(prop_vals),
                      np.max(prop_vals), xscale, xParam)
    ax.hist(values, bins=xbins.bin_limits, normed=False, label=label, **kwargs)

    if 'log' in xscale:
        ax.set_xscale('log')
    if 'log' in yscale:
        ax.set_yscale('log')

    return fig, ax



def plot_ccdf(values, ax=None, xscale='lin', xParam=None, yscale='log',
              threshold_data=False, label=None, discrete=True):
    """
    Plot the experimental 1-CDF of values.

    Parameters
    ----------
    See :py:func:`plot_pdf` for explanations of the parameters.

    bin_data : bool
        whether to use thresholds for drawing the plot
        (more efficient drawing if a lot of points present)
    discrete : bool
        If data is discrete, same value can be observed multiple
        times which (if not treated correctly) can result in
        sawtooth kind of figures.
        There is perhaps no reason for not using discrete=bool
        as there is little overhead (if not even) negative
        overhead to the plotting (and it works also for
        floating point numbers)


    Returns
    -------
    fig : matplotlib Figure
        the parent figure of the axes
    ax : matplotlib Axes object
        the axes in which the ccdf is plotted
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if threshold_data:
        print "no thresholding implemented yet in plotting" + \
            " 1 - CDF, defaulting to basic stuff"

    xvals = np.sort(values)
    yvals = np.linspace(1, 1. / len(xvals), len(xvals))

    if discrete:
        # remove duplicate entries for some results
        args = np.array([True]+list(xvals[:-1]!=xvals[1:]))
        xvals = xvals[args]
        yvals = yvals[args]

    ax.plot(xvals, yvals, label=label)
    if 'log' in xscale:
        ax.set_xscale('log')
    if 'log' in yscale:
        ax.set_yscale('log')
    return fig, ax
