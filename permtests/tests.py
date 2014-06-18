import permute
import measures
import numpy as np
import multiprocessing

# tell nose tests that this file is not a code test as such.
__test__ = False


def permutation_test(data,
                     stat_func,
                     permute_func,
                     n1,
                     n2,
                     paired_study,
                     n_permutations,
                     seed=None):
    """
    Performs a permutation test with user specified test statistics
    yielding an estimate of the pvalue, and the test statistic.

    Args:
        data:           the data on which to perform the test
        permute_func:    the permutation function
        statfunc:       one of the functions in the measures module
        data_array:      1d or 2d array of values (treatment/patients+control)
        n1:             the number of subjcets in the first group
                        (13 treatment -> n=13)
        n2:             # of subjects in the 2nd group
        n_permutations:   number of permutations as int
                        OR 'all' (possible only when paired_study = True)
        paired_study:    True/False: is the test paired or not?
        seed:           seed value for the random number generator
                        (sometimes needed for parallellism)

    Returns:
        Conservative two-sided pvalues and the test statistic
        (tvalue/mean difference)
    """
    #all pairs are paired -> change n_permutations
    if n_permutations == 'all':
        # (n-1): count everything twice for simplicity (a bit foolish...)
        n_permutations = 2**n1
        rng = None  # no rng needed if deterministic permutations
    else:
        rng = permute.get_random_state(seed)

    orig_stat = stat_func(data, n1)
    leftNs = np.zeros(np.shape(orig_stat))
    rightNs = np.zeros(np.shape(orig_stat))
    for i in range(0, int(n_permutations)):

        perm = permute.get_permutation(paired_study, n1, n2, rng, i)
        permdata = permute_func(data, perm)
        stats = stat_func(permdata, n1)
        leftNs += (stats <= orig_stat)
        rightNs += (stats >= orig_stat)
    tailNs = np.minimum(leftNs, rightNs)
    #multiply by two to get two-sided p-values:
    if rng is None:
        return np.minimum(1.0, 2.*tailNs/n_permutations), orig_stat
    else:
        pvals = (tailNs+1.0)/(n_permutations+1.0)
        return np.minimum(1, 2*pvals), orig_stat


def mean_difference_permtest(data_array, n1, n2, paired_study,
                             n_permutations, seed=None, n_cpus=1):
    """
    Performs a simple mean difference permutation test.

    See function permutation_test for explanations of input arguments.
    """
    return parallel_permtest(data_array, measures.mean_difference,
                             permute.permute_array, n1, n2,
                             paired_study, n_permutations, seed, n_cpus)


def t_value_permtest(data_array, n1, n2, paired_study, n_permutations,
                     seed=None, n_cpus=1):
    """
    Performs a simple t-value permutation test.

    See function permutation_test for explanations of input arguments.
    """
    if paired_study:
        stat_func = measures.paired_t_value
    else:
        stat_func = measures.unpaired_t_value
    return parallel_permtest(data_array, stat_func, permute.permute_array,
                             n1, n2, paired_study, n_permutations, seed,
                             n_cpus)


def sim_matrix_within_group_mean_diff_permtest(sim_matrix, n1, n2,
                                               paired_study, n_permutations,
                                               seed=None):
    """
    Performs a simple t-value permutation test.

    See function permutation_test for explanations of input arguments.
    """
    return permutation_test(sim_matrix, measures.sim_matrix_within_group_means,
                            permute.permute_array, n1, n2, paired_study,
                            n_permutations, seed)


def sim_matrix_group_distance_permtest(sim_matrix, n1, n2, paired_study,
                                       n_permutations, seed=None):
    "No parallelism available - nor needed (usually at least)"
    meas = measures.sim_matrix_within_groups_mean_minus_inter_group_mean
    return permutation_test(sim_matrix, meas, permute.permute_array, n1, n2,
                            paired_study, n_permutations, seed)


def sim_matrix_inter_group_means_permtest(matrix, nIt=1e6, seed=None):
    """
    Paired study setup assumed
    (alghough for testing a not paired setting is used.)

    Returns:
        p-values for
            inter_group_mean, semidiag_mean, semidiag_mean - inter_group_mean
        original statistics for
            inter_group_mean, semidiag_mean, semidiag_mean - inter_group_mean
    """
    n1 = np.shape(matrix)[0]/2
    return permutation_test(matrix, measures.sim_matrix_inter_group_means,
                            permute.half_permute_matrix, n1, n1, False, nIt,
                            seed)


def parallel_permtest(data_array, stat_func, permute_func, n1, n2,
                      paired_study, n_permutations, seed=None, n_cpus=1):
    """
    Returns the permutation p-values and the values of the test statistic
    """
    #in case of no parallellism
    if n_cpus == 1:
        return permutation_test(data_array, stat_func, permute_func, n1, n2,
                                paired_study, n_permutations, seed)

    #if parallelism:
    data_arraySlices = np.array_split(data_array, n_cpus, axis=1)
    #create input arguments
    inputArgsList = []

    for i in range(0, n_cpus):
        inputArgsList.append((data_arraySlices[i], stat_func, permute_func, n1,
                             n2, paired_study, n_permutations, seed))

    #run the estimation
    pool = multiprocessing.Pool(processes=n_cpus)
    result = pool.map_async(_parallel_permtest_helper, inputArgsList,
                            chunksize=1)
    #hack (to enable keyboard interruption)
    outputs = result.get(31536000)
    pvals = outputs[0][0]
    orig_stat = outputs[0][1]
    for i in range(1, len(outputs)):
        pvals = np.hstack((pvals, outputs[i][0]))
        orig_stat = np.hstack((orig_stat, outputs[i][1]))
    return pvals, orig_stat


def _parallel_permtest_helper(inputArgs):
    """
    This function needs to be outside of the parallelPermutationTestHelper
    """
    return permutation_test(*inputArgs)
