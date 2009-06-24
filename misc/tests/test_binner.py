import unittest
from operator import itemgetter
import binner

class TestBins(unittest.TestCase):
    def setUp(self):
        # Construct good data
        self.weights = [7.5,  2,   4,  3, 100, 0.1, 0.9,  2,  3]
        self.coords =  [20,   5,   5, 10,  16,   7,   7, 11, 12]
        self.values =    [10, 0.5, 3.5,  1, 1.6, 100,   2, 10, 20]
        self.data = zip(self.coords, self.values)
        self.weighted_data = zip(self.coords, self.values, self.weights)

        # Construct bad data
        self.bad_coords_A = [20,   -1,   5, 10,  16,   7,   7, 11, 12]
        self.bad_coords_B = [20,   2,   5, 10,  30,   7,   7, 11, 12]
        self.bad_data_A = zip(self.bad_coords_A, self.values)
        self.bad_data_B = zip(self.bad_coords_B, self.values)
        self.bad_wdata_A = zip(self.bad_coords_A, self.values, self.weights)
        self.bad_wdata_B = zip(self.bad_coords_B, self.values, self.weights)
        
        self.bins = binner.Bins(int, 0, 20, 'lin', 9)

    def __test_0printer(self):
        """Doesn't test anything, just prints info.

        NOTE! REMOVE THE UNDERSCORES BEFORE 'test' IN THE NAME TO RUN
        THIS METHOD DURING FULL TEST.
        """
        deco = [(c,d,w) for c,d,w in zip(self.coords, self.values, self.weights)]
        deco.sort()
        print "Coords : " + reduce(lambda x,y: str(x)+"\t"+str(y), map(itemgetter(0), deco))
        print "Values : " + reduce(lambda x,y: str(x)+"\t"+str(y), map(itemgetter(1), deco))
        print "Weights: " + reduce(lambda x,y: str(x)+"\t"+str(y), map(itemgetter(2), deco))
        print
        print self.bins.bin_limits
        print self.bins.widths
        print

    def test_Bins_errors(self):
        """Check that input parameters are correct when constructing bins."""
        # minValue == maxValue
        self.assertRaises(binner.BinLimitError, binner.Bins, int, 7, 7, 'lin', 10)
        # Logarithmic bins with minValue <= 0
        self.assertRaises(binner.BinLimitError, binner.Bins, int, 0, 1, 'log', 1.5)
        # Logarithmic bins with factor <= 1
        self.assertRaises(binner.ParameterError, binner.Bins, int, 1, 10, 'linlog', 1)
        self.assertRaises(binner.ParameterError, binner.Bins, int, 1, 10, 'logarithmic', 0.9)
        # Maximum logarithmic bins with 0 <= diff < 1
        self.assertRaises(binner.ParameterError, binner.Bins, int, 1, 10, 'maxlog', -0.1)
        self.assertRaises(binner.ParameterError, binner.Bins, int, 1, 10, 'linmaxlog', 1.1)
        # Linear-logarithmic bins with non-integer bin limits
        self.assertRaises(binner.BinLimitError, binner.Bins, int, 1.5, 20, 'linlog', 2)
        self.assertRaises(binner.BinLimitError, binner.Bins, int, 1, 10.1, 'linmaxlog')
        # Linear bins with <= 0 bins or with a floating point number of bins.
        self.assertRaises(binner.ParameterError, binner.Bins, int, 1, 10, 'linear', 0)
        self.assertRaises(binner.ParameterError, binner.Bins, int, 1, 10, 'linear', 10.5)
        # Custom bins with non-increasing sequence
        bin_lim = [0, 1, 1.5, 3, 2]
        self.assertRaises(binner.ParameterError, binner.Bins, int, 1, 10, 'custom', bin_lim)
        bin_lim = [0, 1, 1.5, 3, 3]
        self.assertRaises(binner.ParameterError, binner.Bins, int, 1, 10, 'custom', bin_lim)
        
    def test_Linbins_int(self):
        N_bin = 3
        b = binner.Bins(int, 1, 5, 'lin', N_bin)
        self.assertEqual(len(b), N_bin)
        expected_bins = (0, 2, 4, 6)
        self.assertEqual(b.bin_limits, expected_bins)
        expected_centers = [1, 3, 5]
        self.assertEqual(b.centers.tolist(), expected_centers)
        expected_widths = [1, 2, 2]
        self.assertEqual(b.widths.tolist(), expected_widths)

    def test_Linbins_float(self):
        minValue = 1
        maxValue = 5
        N_bin = 3
        b = binner.Bins(float, minValue, maxValue, 'lin', N_bin)
        self.assertEqual(len(b), N_bin)
        expected_bins = (0.0, 2.0, 4.0, 6.0)
        self.assertEqual(b.bin_limits, expected_bins)
        expected_centers = [1, 3, 5]
        self.assertEqual(b.centers.tolist(), expected_centers)
        expected_widths = [1, 2, 1]
        self.assertEqual(b.widths.tolist(), expected_widths)

    def test_Logbins_int(self):
        b = binner.Bins(int, 1, 5, 'log', 2)
        self.assertEqual(len(b), 3)
        expected_bins = (2.0/3, 4.0/3, 8.0/3, 16.0/3)
        self.assertEqual(b.bin_limits, expected_bins)
        expected_centers = [1, 2, 4]
        self.assertEqual(b.centers.tolist(), expected_centers)
        expected_widths = [1, 1, 3]
        self.assertEqual(b.widths.tolist(), expected_widths)

    def test_linlogbins_float(self):
        b = binner.Bins(float, 7, 12, 'linlog', 2)
        expected_widths = [0.5,1,1,1,10.5]
        self.assertEqual(b.widths.tolist(), expected_widths)

        b = binner.Bins(float, 0, 12, 'linlog', 2)
        expected_widths = [0.5,1,1,1,1,1,1,1,1,1,1,10.5]
        self.assertEqual(b.widths.tolist(), expected_widths)


    def test_Maxlogbins(self):
        minValue = 1
        maxValue = 20
        b = binner.Bins(int, minValue, maxValue, 'maxlog')
        self.assert_(b.bin_limits[0] < minValue)
        self.assert_(b.bin_limits[-1] > maxValue)
        for w in b.widths:
            self.assert_((w >= 1).all())

    def test_binFinder(self):
        bins = [1, 5, 9, 13, 17, 21, 25, 29, 33]
        bf = binner._binFinder(bins)
        self.assertEqual(bf(1), 0)
        self.assertEqual(bf(2), 0)
        self.assertEqual(bf(21), 5)
        self.assertEqual(bf(32), 7)
        self.assertEqual(bf(33-(1e-6)), 7)

        bins = [1, 100, 1000, 10000, 10010, 10020, 10100, 10200]
        bf = binner._binFinder(bins)
        self.assertEqual(bf(1.001), 0)
        self.assertEqual(bf(99.99), 0)
        self.assertEqual(bf(10009.9), 3)
        self.assertEqual(bf(10010), 4)
        self.assertEqual(bf(10011), 4)

        bins = [-5.87, -1.74, 1.0/3, 1.1/3, 78.0001, 100124]
        bf = binner._binFinder(bins)
        self.assertEqual(bf(-5.86), 0)
        self.assertEqual(bf(-1.74), 1)
        self.assertEqual(bf(0.3333), 1)
        self.assertEqual(bf(1.0/3), 2)
        self.assertEqual(bf(1.05/3), 2)

        assert bf(-5.88) < 0
        assert bf(-78) < 0
        assert bf(100124) > len(bins)-2
        assert bf(200000) > len(bins)-2

    def test_linBinFinder(self):
        bins = [0.5,  1.5,  2.5,  3.5,  4.5,  5.5]
        bf = binner._linBinFinder(bins)
        self.assertEqual(bf(0.5), 0)
        self.assertEqual(bf(1), 0)
        self.assertEqual(bf(2.5), 2)
        self.assertEqual(bf(5), 4)
        self.assertEqual(bf(5.0), 4)

        # Test out of bounds values
        self.assert_(bf(0.499) < 0)
        self.assert_(bf(0) < 0)
        self.assert_(bf(-100) < 0)
        self.assert_(bf(6) > len(bins)-2)
        self.assert_(bf(111) > len(bins)-2)

        bins = [2.5,   7. ,  11.5,  16. ,  20.5]
        bf = binner._linBinFinder(bins)
        self.assertEqual(bf(2.5), 0)
        self.assertEqual(bf(7.6), 1)
        self.assertEqual(bf(18), 3)
        self.assertEqual(bf(20.49), 3)

    def test_logBinFinder(self):
        bins = [ 1, 2, 4, 8, 16, 32, 64 ]
        bf = binner._logBinFinder(bins)
        self.assertEqual(bf(1.5), 0)
        self.assertEqual(bf(2), 1)
        self.assertEqual(bf(2.1), 1)
        self.assertEqual(bf(10), 3)
        self.assertEqual(bf(63.429151), 5)

        # Test out of bounds values
        assert bf(0.999) < 0
        assert bf(0) < 0
        assert bf(-100) < 0
        assert bf(64) > len(bins)-2
        assert bf(111) > len(bins)-2

        bins = [ 5, 7.5, 11.25, 16.875, 25.3125, 37.96875 ]
        bf = binner._logBinFinder(bins)
        self.assertEqual(bf(5.5), 0)
        self.assertEqual(bf(25.3124), 3)
        self.assertEqual(bf(25.3125), 4)

    def test_linlogBinFinder(self):
        bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 21, 42, 84]
        bf = binner._linlogBinFinder(bins)
        self.assertEqual(bf(1), 0)
        self.assertEqual(bf(4), 3)
        self.assertEqual(bf(7.1), 6)
        self.assertEqual(bf(10.6), 10)
        self.assertEqual(bf(22), 11)
        self.assertEqual(bf(80), 12)

        # Test out of bounds values
        assert bf(0.499) < 0
        assert bf(0) < 0
        assert bf(-100) < 0
        assert bf(84) > len(bins)-2
        assert bf(111) > len(bins)-2

        # Large bin values.
        bins = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 42, 168, 672, 2688]
        bf = binner._linlogBinFinder(bins)
        self.assertEqual(bf(6), 0)
        self.assertEqual(bf(10), 4)
        self.assertEqual(bf(100), 6)
        self.assertEqual(bf(42), 6)
        self.assertEqual(bf(168), 7)

        # First bin centered at zero.
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        bf = binner._linlogBinFinder(bins)
        self.assertEqual(bf(0), 0)
        self.assertEqual(bf(4), 4)
        

    def test_conformity(self):
        """ The general search in the base class binFinder whould
        always give exactly the same result as any implementation for
        any special case.
        """
        bins = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 42, 168, 672, 2688]
        bf_A = binner._linlogBinFinder(bins)
        bf_B = binner._binFinder(bins)
        self.assertEqual(bf_A(6), bf_B(6))
        self.assertEqual(bf_A(168), bf_B(168))
        self.assertEqual(bf_A(2687), bf_B(2687))
        self.assertEqual(bf_A(10.5), bf_B(10.5))

    def test_Count(self):
        # Check correct result
        binned_data = self.bins.bin_count(self.coords)
        expected_result = [0,0,2,2,2,1,1,0,1]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.BinLimitError, self.bins.bin_count, self.bad_coords_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_count, self.bad_coords_B)

    def test_CountDiv(self):
        # Check correct result
        binned_data = self.bins.bin_count_divide(self.coords)
        expected_result = [0,0,2./3,1,2./3,1./2,1./3,0,1./2]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.BinLimitError, self.bins.bin_count_divide, self.bad_coords_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_count_divide, self.bad_coords_B)

    def test_Sum(self):
        # Check correct result
        binned_data = self.bins.bin_sum(self.data)
        expected_result = [None,None,4,102,11,20,1.6,None,10]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_sum, self.coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_sum, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_sum, self.bad_data_B)

    def test_SumDivide(self):
        # Check correct result
        binned_data = self.bins.bin_sum_divide(self.data)
        expected_result = [None,None,4./3,51,11./3,10,1.6/3,None,5]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_sum_divide, self.coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_sum_divide, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_sum_divide, self.bad_data_B)

    def test_Average(self):
        # Check correct result
        binned_data = self.bins.bin_average(self.data)
        expected_result = [None,None,2,51,5.5,20,1.6,None,10]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_average, self.coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_average, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_average, self.bad_data_B)

    def test_Average_Variance(self):
        # Check correct result
        binned_data = self.bins.bin_average(self.data, True)
        expected_average = [None,None,2,51,5.5,20,1.6,None,10]
        expected_variance = [None,None,2.25,2401.0,20.25,0.0,0.0,None,0.0]
        self.assertEqual(binned_data[0].tolist(), expected_average)
        self.assertEqual(binned_data[1].tolist(), expected_variance)

    def test_WeightedAverage(self):
        # Check correct result
        binned_data = self.bins.bin_weighted_average(self.weighted_data)
        expected_result = [None,None,2.5,11.8,23./5,20,1.6,None,10]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_weighted_average, self.coords)
        self.assertRaises(binner.DataTypeError, self.bins.bin_weighted_average, self.data)
        self.assertRaises(binner.BinLimitError, self.bins.bin_weighted_average, self.bad_wdata_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_weighted_average, self.bad_wdata_B)

    def test_WeightedAverage_Variance(self):
        # Check correct result
        binned_data = self.bins.bin_weighted_average(self.weighted_data, True)
        expected_average = [None,None,2.5,11.8,23./5,20,1.6,None,10]
        expected_variance = [None,None,2.0, 0.9*4+1000-11.8**2,
                             (3+200)/5.0-(23./5)**2, 0.0,0.0,None,0.0]
        self.assertEqual(binned_data[0].tolist(), expected_average)
        self.assertEqual(binned_data[1].tolist(), expected_variance)

    def test_Median(self):
        # Check correct result
        new_data = zip(self.coords + [4,11], self.values + [1,30])
        binned_data = self.bins.bin_median(new_data)
        expected_result = [None,None,1,51,10,20,1.6,None,10]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_median, self.coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_median, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_median, self.bad_data_B)


class Test_Bins2D(unittest.TestCase):
    def setUp(self):
        self.x_coords = [1, 1, 2, 4, 4, 3, 1, 2]
        self.y_coords = [1, 4, 3, 5, 6, 1, 8, 6]
        values = [1, 3, 7, 5, 6, 3, 1, 10]
        weights = [1, 3, 1, 0, 100, 0, 1, 10]

        self.coords = zip(self.x_coords, self.y_coords)
        self.data = zip(self.x_coords, self.y_coords, values)
        self.weighted_data = zip(self.x_coords, self.y_coords, values, weights)

        bad_x_coords = [1, 1, 2, 4, 4, 5, 1, 2]
        bad_y_coords = [1, 4, 0, 5, 6, 1, 8, 6]
        self.bad_coords_A = zip(bad_x_coords, self.y_coords)
        self.bad_coords_B = zip(self.x_coords, bad_y_coords)
        self.bad_data_A = zip(bad_x_coords, self.y_coords, values)
        self.bad_data_B = zip(self.x_coords, bad_y_coords, values) 
        self.bad_wdata_A = zip(bad_x_coords, self.y_coords, values, weights)
        self.bad_wdata_B = zip(self.x_coords, bad_y_coords, values, weights) 

        self.bins = binner.Bins2D(int, 1, 4, 'custom', [0.5, 1.5, 4.5], \
                                  int, 1, 10, 'custom', [0.5, 4.5, 5.5, 6.5, 10])

    def test_Count(self):
        binned_data = self.bins.bin_count(self.coords)
        expected_result = [[2,0,0,1],[2,1,2,0]]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_count, self.x_coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_count, self.bad_coords_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_count, self.bad_coords_B)

    def test_Count_with_data(self):
        binned_data = self.bins.bin_count(self.data)
        expected_result = [[2,0,0,1],[2,1,2,0]]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.BinLimitError, self.bins.bin_count, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_count, self.bad_data_B)

    def test_Count_divide(self):
        binned_data = self.bins.bin_count_divide(self.coords)
        expected_result = [[2.0/4,0,0,1.0/3],
                           [2.0/12,1.0/3,2.0/3,0]]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_count_divide, self.x_coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_count_divide, self.bad_coords_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_count_divide, self.bad_coords_B)

    def test_Average(self):
        binned_data = self.bins.bin_average(self.data)
        expected_result = [[2,None,None,1],[5,5,8,None]]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_average, self.x_coords)
        self.assertRaises(binner.DataTypeError, self.bins.bin_average, self.coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_average, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_average, self.bad_data_B)

    def test_Average_Variance(self):
        binned_data = self.bins.bin_average(self.data, True)
        expected_average = [[2,None,None,1],[5,5,8,None]]
        expected_variance = [[1.0,None,None,0.0],[4.0,0.0,4.0,None]]
        self.assertEqual(binned_data[0].tolist(), expected_average)
        self.assertEqual(binned_data[1].tolist(), expected_variance)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_average, self.x_coords)
        self.assertRaises(binner.DataTypeError, self.bins.bin_average, self.coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_average, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_average, self.bad_data_B)

    def test_Sum(self):
        binned_data = self.bins.bin_sum(self.data)
        expected_result = [[4,None,None,1], [10,5,16,None]]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_sum, self.x_coords)
        self.assertRaises(binner.DataTypeError, self.bins.bin_sum, self.coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_sum, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_sum, self.bad_data_B)

    def test_Sum_divide(self):
        binned_data = self.bins.bin_sum_divide(self.data)
        expected_result = [[4.0/4,None,None,1.0/3],
                           [10.0/12,5.0/3,16.0/3,None]]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_sum_divide, self.x_coords)
        self.assertRaises(binner.DataTypeError, self.bins.bin_sum_divide, self.coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_sum_divide, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_sum_divide, self.bad_data_B)

    def test_Weighted_average(self):
        binned_data = self.bins.bin_weighted_average(self.weighted_data)
        expected_result = [[10./4,None,None,1],[7,0,700./110,None]]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_weighted_average, self.x_coords)
        self.assertRaises(binner.DataTypeError, self.bins.bin_weighted_average, self.coords)
        self.assertRaises(binner.DataTypeError, self.bins.bin_weighted_average, self.data)
        self.assertRaises(binner.BinLimitError, self.bins.bin_weighted_average, self.bad_wdata_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_weighted_average, self.bad_wdata_B)

    def test_Weighted_average_Variance(self):
        binned_data = self.bins.bin_weighted_average(self.weighted_data, True)
        expected_average = [[10./4,None,None,1],[7,0,700./110,None]]
        expected_variance = [[0.75,None,None,0.0],[0.0,0.0,4600/110.0-(700./110)**2,None]]
        self.assertEqual(binned_data[0].tolist(), expected_average)
        self.assertEqual(binned_data[1].tolist(), expected_variance)


    def test_Median(self):
        binned_data = self.bins.bin_median(self.data)
        expected_result = [[2,None,None,1],[5,5,8,None]]
        self.assertEqual(binned_data.tolist(), expected_result)

        # Check exceptions
        self.assertRaises(binner.DataTypeError, self.bins.bin_median, self.x_coords)
        self.assertRaises(binner.DataTypeError, self.bins.bin_median, self.coords)
        self.assertRaises(binner.BinLimitError, self.bins.bin_median, self.bad_data_A)
        self.assertRaises(binner.BinLimitError, self.bins.bin_median, self.bad_data_B)


if __name__ == '__main__':
    # The if clause exists only for debugging; it makes it easier to
    # run only one test instead of all of them. Also, the tests are
    # normally run by running binner.py.
    if True:
        # Run only one test.
        suite = unittest.TestSuite()
        suite.addTest(TestBins("test_AverageVariance"))
        unittest.TextTestRunner().run(suite)
    else:
        # Run all tests.
        unittest.main()
