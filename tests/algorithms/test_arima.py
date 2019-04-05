from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import unittest

import numpy.testing as np_test

from scripts.algorithms.arima import ARIMAForecast

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import os
import pandas as pd
import numpy as np; print("NumPy", np.__version__)
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

    def test_linear_sequence(self):
        time_series = [1.0, 2.0, 3.0, 4.0, 5.0]
        num_predicted_periods = 3
        expected_prediction = [6.0, 7.0, 8.0]
        arima = ARIMAForecast(time_series, num_predicted_periods)

        actual_prediction = arima.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=4)

    def test_flakey_sequence(self):
        time_series = [1.0, -2.0, 3.0, -4.0, 5.0]
        num_predicted_periods = 3
        expected_prediction = [np.nan, np.nan, np.nan]
        arima = ARIMAForecast(time_series, num_predicted_periods)

        actual_prediction = arima.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=4)

    def test_linearly_increasing_sequence_fuel_cell(self):
        time_series = pd.read_csv(os.path.join('tests','data', 'fuel_cell_quarterly.csv')).values.tolist()
        time_series = [item for sublist in time_series for item in sublist]
        num_predicted_periods = 4
        expected_prediction = [333., 333., 334., 335.]
        arima = ARIMAForecast(np.array(time_series).astype(float), num_predicted_periods)

        actual_prediction = arima.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=0)

    def test_linearly_decreasing_sequence_image_data(self):
        time_series = pd.read_csv(os.path.join('tests','data', 'image_data_quarterly.csv')).values.tolist()
        time_series = [item for sublist in time_series for item in sublist]
        num_predicted_periods = 4
        expected_prediction = [562., 561., 558., 556.]
        arima = ARIMAForecast(np.array(time_series).astype(float), num_predicted_periods)

        actual_prediction = arima.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=0)
