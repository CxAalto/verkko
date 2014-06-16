import numpy as np


def mean_difference(dataArray, n1):
    """
    Computes the mean difference

    Args:
        dataArray:  1 or 2 dimensional array
        n1:         the number of elements in the first group

    Returns:
        the mean difference
    """
    return (np.average(dataArray[:n1], axis=0) -
            np.average(dataArray[n1:], axis=0))


def paired_t_value(dataArray, n1):
    """
    Computes the paired t-value for a one or
    two dimensional numpy array
    See e.g.

    Args:
        dataArray:  array of the values
        n1:         the number of elements in the first group
                    (same as in the second group)

    Returns:
        the t value
    """
    differences = dataArray[:n1] - dataArray[n1:]
    stds = np.std(differences, axis=0, ddof=1)
    avgs = np.average(differences, axis=0)
    return avgs / stds * np.sqrt(n1)


def t_value(dataArray, n1):
    """
    Computes the t-value (variance normalized mean difference) for a one or
    two dimensional numpy array

    Args:
        dataArray:  array of the values
        n1:         the number of elements in the first group

    Returns:
        the t value
    """
    n2 = dataArray.shape[0] - n1
    var1 = np.var(dataArray[:n1], axis=0, ddof=1)
    var2 = np.var(dataArray[n1:], axis=0, ddof=1)
    mu1 = np.average(dataArray[:n1], axis=0)
    mu2 = np.average(dataArray[n1:], axis=0)
    return (mu1 - mu2) / np.sqrt(var1 / n1 + var2 / n2)

# def sim_matrix_group_mean_diff(matrix, n1):
#    return sim_matrix_within_group_means(matrix, n1, False)[2]


def sim_matrix_within_group_means(matrix, n1):
    """
    Computes the mean of the upper triangle (k=1) for the blocks
    (0,n-1)*(0,n-1) and (n,2n-1)*(n,2n-1), and their difference
    (for convenience).
    Also the between groups mean is computed.
    """
    n2 = matrix.shape[0] - n1
    indices1 = np.triu_indices(n1, k=1)
    indices2base = np.triu_indices(n2, k=1)
    indices2I = indices2base[0].copy() + n1
    indices2J = indices2base[1].copy() + n1
    indices2 = (indices2I, indices2J)
    mean1 = np.average(matrix[indices1])
    mean2 = np.average(matrix[indices2])
    return mean1, mean2, mean1 - mean2


def sim_matrix_inter_group_means(mat, paired=True, n1=None):
    """
    Computes the inter-group average (and separately for the same subjects!)

    Args:
        mat: The similarity matrix (numpy array, should be symmetric!)
        paired: If paired setting or not (with paired setting)
        n1:  The number of subjects in the first group
                (only needed if paired==False)

    Returns:
        interMean       : the mean value of the inter-group area of the mat
                            -if paired==True, the 'half diagonal' is not
                            taken into account in computations.
        semidiagMean    : the mean value of the half-diagonal
                            (returned only if paired setting is used)
        semidiagMean-interMean : (convenience for statistical testing,  only
                                    with paired setting)
    """
    if paired:
        n1 = np.shape(mat)[0] / 2
    else:
        assert n1 is not None, "give out n1 in sim_matrix_inter_group_means"
    n2 = np.shape(mat)[0] - n1

    # between group similarities:
    incidenceMatrix12 = np.ones((n1, n2))
    if paired:
        incidenceMatrix12 -= np.eye(n1)
    betweenI = incidenceMatrix12.nonzero()[0].copy()
    betweenI += n1
    betweenJ = incidenceMatrix12.nonzero()[1].copy()
    betweenIndices = (betweenI, betweenJ)
    interMean = np.average(mat[betweenIndices])
    if not paired:
        return interMean

    semidiagIndicesI = np.eye(n1).nonzero()[0].copy()
    semidiagIndicesJ = np.eye(n1).nonzero()[1].copy()
    semidiagIndicesJ += n1
    semidiagIndices = (semidiagIndicesI, semidiagIndicesJ)
    semidiagMean = np.average(mat[semidiagIndices])
    return interMean, semidiagMean, semidiagMean - interMean


def sim_matrix_within_groups_mean_minus_inter_group_mean(mat, paired, n1=None):
    """
    Computes the difference in the average within
    Arguments:
        mat: the similarity matrix (with a paired setting)
        paired: if paired=True, the same subject is not taken into account

    Computes the inter conditions mean by neglecting the "semi-diagonal"
    corresponding to the same subject (paired setting).
    """
    if paired:
        n1 = np.shape(mat)[0] / 2
    else:
        assert n1 is not None, ("give out n1 in sim_matrix_within_group_" +
                                "means_minus_inter_group_mean")
    withinGroupAvgs = np.mean(
        sim_matrix_within_group_means(mat, n1)[0:2])
    interGroupMean = sim_matrix_inter_group_means(mat, paired=paired, n1=n1)
    if paired is True:
        interGroupMean = interGroupMean[0]
    return withinGroupAvgs - interGroupMean
