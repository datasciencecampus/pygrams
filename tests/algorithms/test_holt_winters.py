from sys import platform as sys_pf

from scripts.algorithms.holtwinters_predictor import HoltWintersPredictor

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")

import unittest
import numpy.testing as np_test

import platform;

print(platform.platform())
import sys;

print("Python", sys.version)
import numpy as np;

print("NumPy", np.__version__)
import scipy;

print("SciPy", scipy.__version__)
import sklearn;

print("Scikit-Learn", sklearn.__version__)
import statsmodels;

print("Statsmodels", statsmodels.__version__)


class HoltWintersTests(unittest.TestCase):

    def test_negatives_in_sequence(self):
        time_series = [1, 1, -1, 1, 1]
        num_predicted_periods = 3

        with self.assertRaises(NotImplementedError) as nie:
            HoltWintersPredictor(time_series, num_predicted_periods)

        self.assertEqual(nie.exception.args[0], 'Unable to correct for negative or zero values')

    def test_zeros_in_sequence(self):
        time_series = [1, 1, 0, 1, 1]
        num_predicted_periods = 3
        expected_prediction = [0.8] * num_predicted_periods
        hw = HoltWintersPredictor(time_series, num_predicted_periods)

        actual_prediction = hw.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=4)

    def test_static_sequence(self):
        time_series = [1.0, 1.0, 1.0, 1.0, 1.0]
        num_predicted_periods = 3
        expected_prediction = [1] * num_predicted_periods
        hw = HoltWintersPredictor(time_series, num_predicted_periods)

        actual_prediction = hw.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=4)
