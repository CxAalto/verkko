"""
This module contains functions and methods for permuting arrays.
Intended to use by the 'tests' module, but may be suitable also
for other use.

Author: Rainer.Kujala@gmail.com
"""
import numpy as np
from numpy.random import RandomState


def get_random_state(seed=None):
    """
    Get a numpy.random.RandomState object with the given seed.
    If no seed is given or seed = None, a RandomState object with default
    numpy seeding method (from time) is returned.

    Args:
        seed: an int or None
    Returns:
        The RandomState object (a random number generator)
    """
    if seed is not None:
        return RandomState(seed)
    else:
        return RandomState()


def get_permutation(paired, n1, n2, rng, i):
    """
    Helper function to compute a permutation with all the switches.

    Args:
        paired: True or False
        n1: number of instances in the first group
        n2: number of instances in the second group
        rng: A random number generator (None for deterministic permutations)
        i: The "index" of the permutation, when permSamples = 'all' is used
    """
    if paired is True:
        if rng is None:  # go through full permutation dist.
            return _get_paired_permutation(n1, i)
        else:  # a paired setting but a random permutation
            return _get_random_paired_permutation(n1+n2, rng)
    else:  # not a paired setting
        return rng.permutation(n1 + n2)


def _get_random_paired_permutation(nTot, rng):
    n = nTot / 2
    perm = np.arange(0, nTot, dtype=np.int32)
    rands = rng.randint(0, 2, n)
    perm[:n] += rands*n
    perm[n:] -= rands*n
    return perm


def _get_paired_permutation(n, i):
    permBase = np.zeros(n, dtype=np.int32)
    #compute the bin repr.
    for j in np.linspace(n-1, 0, n).astype(np.int32):
        c = i // 2**j
        permBase[j] = c
        i -= 2**j * c
    perm = np.arange(0, 2*n, dtype=np.int32)
    perm[:n] += permBase*n
    perm[n:] -= permBase*n
    return perm


def permute_array(dataArray, perm):
    return dataArray[perm]


def permute_matrix(matrix, perm):
    return matrix[perm, :][:, perm]


def half_permute_matrix(matrix, perm):
    """ Got a full permutation, use only half of the permutation"""
    perm = np.array(perm)
    n1 = np.shape(matrix)[0]/2
    perm = perm[perm >= n1]
    perm = np.hstack((np.arange(0, n1, dtype=np.int32), perm))
    return permute_matrix(matrix, perm)
