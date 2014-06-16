import permute
import numpy as np
import measures
import multiprocessing


def permutation_test(data,
                     stat_func,
                     permute_func,
                     n1,
                     n2,
                     paired_study,
                     n_permutations,
                     seed=123456):
    """
    Performs a permutation test with user specified test statistics
    yielding an estimate of the pvalue, and the test statistic.

    Args:
        data:           the data on which to perform the test
        permute_func:    the permutation function
        statfunc:       one of the functions in the measures module
        dataArray:      1d or 2d array of values (treatment/patients+control)
        n1:             the number of subjcets in the first group
                        (13 treatment -> n=13)
        n2:             # of subjects in the 2nd group
        n_permutations:   number of permutations as int
                        OR 'all' (possible only when paired_study = True)
        paired_study:    True/False: is the test paired or not?
        seed:           seed value for the random number generator
                        (sometimes needed for parallellism)

    Returns:
        Two-sided pvalues and the test statistic (tvalue/mean difference)
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


def mean_differencePermutationTest(dataArray, n1, n2, paired_study,
                                   n_permutations, seed=123456, nCpus=1):
    return parallelPermutationTest(dataArray, measures.mean_difference,
                                   measures.permute_array, n1, n2,
                                   paired_study, n_permutations, seed, nCpus)


def tValuePermutationTest(dataArray, n1, n2, paired_study, n_permutations,
                          seed=123456, nCpus=1):
    if paired_study:
        stat_func = measures.paired_tvalue
    else:
        stat_func = measures.t_value
    return parallelPermutationTest(dataArray, stat_func,
                                   measures.permute_array,
                                   n1, n2, paired_study, n_permutations, seed,
                                   nCpus)


def groupMeanDiffSimMatrixPermutationTest(simMatrix, n1, n2, paired_study,
                                          n_permutations, seed=123456):
    "No parallelism available - nor needed (usually at least)"
    return permutation_test(simMatrix, measures.sim_matrix_within_group_means,
                            measures.permute_array, n1, n2, paired_study,
                            n_permutations, seed)


def distanceBetweenGroupsPermutationTest(simMatrix, n1, n2, paired_study,
                                         n_permutations, seed=123456):
    "No parallelism available - nor needed (usually at least)"
    meas = measures.sim_matrix_within_groups_mean_minus_inter_group_mean
    return permutation_test(simMatrix, meas, measures.permute_array, n1, n2,
                            paired_study, n_permutations, seed)


def crossGroupMeanDifSimMatrixPermutationTest(matrix, nIt=1e6, seed=123456):
    """
    Paired study setup assumed
    (alghough for testing a not paired setting is used.)

    Returns:
        p-values for (crossMean, semidiagMean, semidiagMean - crossMean)
        orig. stats for (crossMean, semidiagMean, semidiagMean - crossMean)
    """
    n1 = np.shape(matrix)[0]/2
    return permutation_test(matrix, measures.sim_matrix_inter_group_means,
                            permute.half_permute_matrix, n1, n1, False, nIt,
                            seed)


def parallelPermutationTest(dataArray, stat_func, permute_func, n1, n2,
                            paired_study, n_permutations, seed=123456,
                            nCpus=1):
    """
    Returns the permutation p-values and the values of the test statistic
    """
    #in case of no parallellism
    if nCpus == 1:
        return permutation_test(dataArray, stat_func, permute_func, n1, n2,
                                paired_study, n_permutations, seed)

    #if parallelism:
    dataArraySlices = np.array_split(dataArray, nCpus, axis=1)
    #create input arguments
    inputArgsList = []

    for i in range(0, nCpus):
        inputArgsList.append((dataArraySlices[i], stat_func, permute_func, n1,
                             n2, paired_study, n_permutations, seed))

    #run the estimation
    pool = multiprocessing.Pool(processes=nCpus)
    result = pool.map_async(_parallelPermutationTestHelper, inputArgsList,
                            chunksize=1)
    #hack (to enable keyboard interruption)
    outputs = result.get(31536000)
    pvals = outputs[0][0]
    orig_stat = outputs[0][1]
    for i in range(1, len(outputs)):
        pvals = np.hstack((pvals, outputs[i][0]))
        orig_stat = np.hstack((orig_stat, outputs[i][1]))
    return pvals, orig_stat


def _parallelPermutationTestHelper(inputArgs):
    """
    This function needs to be outside of the parallelPermutationTestHelper
    """
    return permutation_test(*inputArgs)
