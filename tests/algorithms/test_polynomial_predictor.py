from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")

import unittest
import numpy.testing as np_test
from scripts.algorithms.polynomial_predictor import PolynomialPredictor


class PolynomialPredictorTests(unittest.TestCase):

    def test_static_sequence(self):
        time_series = [1.0, 1.0, 1.0, 1.0, 1.0]
        num_predicted_periods = 3
        expected_prediction = [1] * num_predicted_periods
        predictor = PolynomialPredictor(time_series, num_predicted_periods)

        actual_prediction = predictor.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=4)

    def test_linearly_increasing_sequence(self):
        time_series = [8.9, 11.0, 13.0, 15.1, 17.0, 18.9, 21.0]
        num_predicted_periods = 4
        expected_prediction = [23.0, 25.0, 27.0, 29.0]
        predictor = PolynomialPredictor(time_series, num_predicted_periods)

        actual_prediction = predictor.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=0)

    def test_quadratically_increasing_sequence(self):
        values = list(map(lambda x: (x ** 2) - (3 * x) + 2, range(15)))
        num_predicted_periods = 4
        time_series = values[:-num_predicted_periods]
        expected_prediction = values[-num_predicted_periods:]
        predictor = PolynomialPredictor(time_series, num_predicted_periods)

        actual_prediction = predictor.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=1)
