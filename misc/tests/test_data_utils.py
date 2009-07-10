import unittest
import numpy as np
import data_utils as du

class Test_DataUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_cumulative_dist(self):
        # Function for converting integers to strings with given
        # accuracy.
        to_str = lambda x: "%.10f" % x

        # TEST DESCENDING: P(X >= x)
        correct_data = [0.1,   1,   4,   8,   9]
        correct_prob = [  1, 0.8, 0.4, 0.2, 0.1]
        correct_prob = map(to_str, correct_prob)

        # Test with data only
        data = [1, 0.1, 9, 4, 1, 1, 8, 1, 4, 1.0/10]
        cum_data, cum_prob = du.cumulative_dist(data)
        self.assertEqual(cum_data, correct_data)
        self.assertEqual(map(to_str, cum_prob), correct_prob)

        # Test with count
        data =  [1, 0.1, 9, 4, 8]
        count = [4,   2, 1, 2, 1]
        cum_data, cum_prob = du.cumulative_dist(data, count)
        self.assertEqual(cum_data, correct_data)
        self.assertEqual(map(to_str, cum_prob), correct_prob)
        
        # TEST ASCENTING: P(X <= x)
        correct_data = [0.1,   1,   4,   8,   9]
        correct_prob = [0.2, 0.6, 0.8, 0.9, 1.0]
        correct_prob = map(to_str, correct_prob)

        # Test with data only
        data = [1, 0.1, 9, 4, 1, 1, 8, 1, 4, 1.0/10]
        cum_data, cum_prob = du.cumulative_dist(data, format='ascenting')
        self.assertEqual(cum_data, correct_data)
        self.assertEqual(map(to_str, cum_prob), correct_prob)

        # Test with count
        data =  [1, 0.1, 9, 4, 8]
        count = [4,   2, 1, 2, 1]
        cum_data, cum_prob = du.cumulative_dist(data, count, format='ascenting')
        self.assertEqual(cum_data, correct_data)
        self.assertEqual(map(to_str, cum_prob), correct_prob)

        # TEST EXCEPTIONS
        bad_format = 'descenting'
        self.assertRaises(ValueError, du.cumulative_dist, data, format=bad_format)
        bad_count = [4,   1, 1, 2, 1, 7]
        self.assertRaises(ValueError, du.cumulative_dist, data, bad_count)

if __name__ == '__main__':
    if False:
        suite = unittest.TestSuite()
        suite.addTest(Test_DataUtils("test_cumulative_dist"))
        unittest.TextTestRunner().run(suite)
    else:
        unittest.main()
