import numpy as np


def fdr_bh(q, p_values, return_significant=False):
    """
    Computes the fdr treshold based on the given p-values and the desired
    q-value according to the Benjamini-Hochberg process.

    Arguments
    ---------
    q : float
        the rate of false discoveries
    pvalues : numpy array
        The (non-adjusted) p-values obtained from individual tests
        Can be of any dimension
    return_significant : bool, optional
        Whether to return a bool numpy array with 'surviving' p-values
        marked True

    Returns
    -------
    p_thresh : float
        A single floating point number describing the treshold p-value.
        If no test should be rejected, returns 0.0


    References
    ----------
    Wikipedia, False discovery rate:
    http://en.wikipedia.org/wiki/False_discovery_rate

    Notes
    -----
    Assumes independence for the individual p-values.
    Fast with even 1e6 pvalues (or more...)
    """
    pvalscopy = p_values.flatten()  # pvals as a vector, flatten = copy
    pvalscopy.sort()  # sorted from smallest to largest
    m = len(pvalscopy)  # number of pvals
    compvec = np.arange(1, m + 1) * (q / float(m))  # to which we compare
    fulfilling_indices = np.nonzero(pvalscopy <= compvec)
    if len(fulfilling_indices[0]) == 0:
        p_thresh = 0.0
    else:
        p_thresh = pvalscopy[np.max(fulfilling_indices)]
    if return_significant:
        return p_thresh, p_values <= p_thresh
    else:
        return p_thresh


def bonferroni(alpha, pvalues):
    """
    Return Bonferroni-corrected p-value treshold and
    significant cases.
    """
    p_vals_copy = pvalues.flatten()
    bonf_alpha = alpha / len(p_vals_copy)
    return bonf_alpha, pvalues <= bonf_alpha
