import permute
import unittest
import numpy as np


class TestPermutation(unittest.TestCase):

    def setUp(self):
        self.as_arr_alm_eq = np.testing.assert_almost_equal
        self.data = np.array([1., 2, 3, 4, 5, 6])
        self.perm1 = [0, 1, 2, 3, 4, 5]
        self.perm2 = [1, 2, 3, 4, 5, 0]

        self.two_dim_data = np.array([
                                     [1, 1],
                                     [2, 2],
                                     [3, 3],
                                     [4, 4],
                                     [5, 5],
                                     [6, 6]
                                     ])

        self.matrix = np.matrix([
                                [11, 12, 13, 14],
                                [21, 22, 23, 24],
                                [31, 32, 33, 34],
                                [41, 42, 43, 44]
                                ])
        self.mat_perm_1 = [3, 1, 2, 0]
        self.mat_perm_2 = [1, 2, 3, 0]

    def test_permute_array(self):
        newdata1 = permute.permute_array(self.data, self.perm1)
        self.as_arr_alm_eq(newdata1, np.array([1, 2, 3, 4, 5, 6]))
        newdata2 = permute.permute_array(self.data, self.perm2)
        self.as_arr_alm_eq(newdata2, np.array([2, 3, 4, 5, 6, 1]))

        two_dim_data_1 = permute.permute_array(self.two_dim_data, self.perm1)
        result_should_be = np.array([
                                    [1, 1],
                                    [2, 2],
                                    [3, 3],
                                    [4, 4],
                                    [5, 5],
                                    [6, 6]
                                    ])
        self.as_arr_alm_eq(two_dim_data_1, result_should_be)

        two_dim_data_2 = permute.permute_array(self.two_dim_data, self.perm2)
        result_should_be = np.array([
                                    [2, 2],
                                    [3, 3],
                                    [4, 4],
                                    [5, 5],
                                    [6, 6],
                                    [1, 1]
                                    ])
        self.as_arr_alm_eq(two_dim_data_2, result_should_be)

    def test_permute_matrix(self):
        perm_mat = permute.permute_matrix(self.matrix, self.mat_perm_1)
        should_be = np.matrix([
                              [44, 42, 43, 41],
                              [24, 22, 23, 21],
                              [34, 32, 33, 31],
                              [14, 12, 13, 11]
                              ])
        self.as_arr_alm_eq(perm_mat, should_be)




# def get_permutation(paired, n1, n2, rng, i):


# def _get_random_paired_permutation(nTot, rng):

# def _get_paired_permutation(n, i):

# def permute_array(dataArray, perm):

# def permute_matrix(matrix, perm):

# def half_permute_matrix(matrix, perm):

# def get_random_state(seed=None):
