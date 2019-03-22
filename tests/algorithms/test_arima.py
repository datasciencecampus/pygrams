from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import unittest

import numpy.testing as np_test

from scripts.algorithms.arima import ARIMAForecast

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import pmdarima; print("pmdarima", pmdarima.__version__)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import statsmodels; print("Statsmodels", statsmodels.__version__)


class ArimaTests(unittest.TestCase):

    def test_non_float_sequence(self):
        time_series = [1, 1, 1, 1, 1]
        num_predicted_periods = 3

        with self.assertRaises(ValueError) as cm:
            ARIMAForecast(time_series, num_predicted_periods)

        self.assertEqual(cm.exception.args[0], 'Time series must be all float values')

    def test_static_sequence(self):
        time_series = [1.0, 1.0, 1.0, 1.0, 1.0]
        num_predicted_periods = 3
        expected_prediction = [1] * num_predicted_periods
        arima = ARIMAForecast(time_series, num_predicted_periods)

        actual_prediction = arima.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=4)

    def test_linearly_increasing_sequence(self):
        time_series = [8.9, 11.0, 13.0, 15.1, 17.0, 18.9, 21.0]
        num_predicted_periods = 4
        expected_prediction1 = [21., 20., 20., 19]
        expected_prediction2 = [23., 25., 27., 29.]
        arima = ARIMAForecast(time_series, num_predicted_periods)
        config = arima.configuration
        expected_prediction = expected_prediction2 if config == (0, 1, 0) else expected_prediction1

        actual_prediction = arima.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=0)
