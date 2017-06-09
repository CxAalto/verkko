import corrections
import numpy as np

import unittest


class Test(unittest.TestCase):

    def setUp(self):
        # original p-values used in the BH study:
        # http://www.jstor.org/stable/2346101?seq=7#page_scan_tab_contents
        self.p_vals_bh = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
                                   0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000])

        self.zeros = np.zeros(10)
        self.ones = np.ones(10)

        self.test_data_1 = np.concatenate(
            (np.linspace(0.0, 0.1, 100), np.linspace(0.0, 1, 100)))

    def test_fdr_bh(self):
        q = 0.05
        p_thresh, significants = corrections.fdr_bh(
            q, self.p_vals_bh, return_significant=True)
        assert np.sum(significants) == 4
        assert (significants[:4] == True).all()
        assert (significants[4:] == False).all()
        assert p_thresh == 0.0095

        p_thresh, significants = corrections.fdr_bh(
            0.00001, self.zeros, return_significant=True)
        assert p_thresh == 0.0000
        assert np.sum(significants) == len(significants)

        p_thresh, significants = corrections.fdr_bh(
            0.00001, self.ones, return_significant=True)
        assert p_thresh == 0.0000
        assert np.sum(significants) == 0

        q = 0.2
        p_thresh, significants = corrections.fdr_bh(
            q, self.test_data_1, return_significant=True)
        print(p_thresh)
        assert p_thresh > 0.1
        assert p_thresh < 0.12
        assert np.sum(significants) > 100

    # def test_bonferroni(self):
    # todo!
