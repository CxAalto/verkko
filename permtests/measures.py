import numpy as np


def mean_difference(data_array, n1):
    """
    Computes the mean difference

    Args:
        data_array:  1 or 2 dimensional array
        n1:         the number of elements in the first group

    Returns:
        the mean difference
    """
    return (np.average(data_array[:n1], axis=0) -
            np.average(data_array[n1:], axis=0))


def paired_t_value(data_array, n1):
    """
    Computes the paired t-value for a one or
    two dimensional numpy array
    See e.g.

    Args:
        data_array:  array of the values
        n1:         the number of elements in the first group
                    (same as in the second group)

    Returns:
        the t value
    """
    differences = data_array[:n1] - data_array[n1:]
    stds = np.std(differences, axis=0, ddof=1)
    avgs = np.average(differences, axis=0)
    return avgs / stds * np.sqrt(n1)


def unpaired_t_value(data_array, n1):
    """
    Computes the t-value (variance normalized mean difference) for a one or
    two dimensional numpy array

    Args:
        data_array:  array of the values
        n1:         the number of elements in the first group

    Returns:
        the t value
    """
    n2 = data_array.shape[0] - n1
    var1 = np.var(data_array[:n1], axis=0, ddof=1)
    var2 = np.var(data_array[n1:], axis=0, ddof=1)
    mu1 = np.average(data_array[:n1], axis=0)
    mu2 = np.average(data_array[n1:], axis=0)
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
        inter_group_mean: the mean value of the inter-group area of the mat
                            -if paired==True, the 'half diagonal' is not
                            taken into account in computations.
        semidiag_mean: the mean value of the half-diagonal
                            (returned only if paired setting is used)
        semidiag_mean-inter_group_mean: (convenience for statistical testing,
                                        only with paired setting)
    """
    if paired:
        n1 = np.shape(mat)[0] / 2
    else:
        assert n1 is not None, "give out n1 in sim_matrix_inter_group_means"
    n2 = np.shape(mat)[0] - n1

    # between group similarities:
    incidence_matrix12 = np.ones((n1, n2))
    if paired:
        incidence_matrix12 -= np.eye(n1)
    between_I = incidence_matrix12.nonzero()[0].copy()
    between_I += n1
    between_J = incidence_matrix12.nonzero()[1].copy()
    between_indices = (between_I, between_J)
    inter_group_mean = np.average(mat[between_indices])
    if not paired:
        return inter_group_mean

    semidiag_indices_I = np.eye(n1).nonzero()[0].copy()
    semidiag_indices_J = np.eye(n1).nonzero()[1].copy()
    semidiag_indices_J += n1
    semidiag_indices = (semidiag_indices_I, semidiag_indices_J)
    semidiag_mean = np.average(mat[semidiag_indices])
    return inter_group_mean, semidiag_mean, semidiag_mean - inter_group_mean


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
    within_group_means = np.mean(
        sim_matrix_within_group_means(mat, n1)[0:2])
    inter_group_mean = sim_matrix_inter_group_means(mat, paired=paired, n1=n1)
    if paired is True:
        inter_group_mean = inter_group_mean[0]
    return within_group_means - inter_group_mean
