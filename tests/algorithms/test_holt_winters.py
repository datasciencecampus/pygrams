import unittest

import numpy.testing as np_test

from pygrams.algorithms.holtwinters_predictor import HoltWintersPredictor


class HoltWintersTests(unittest.TestCase):

    def test_negatives_in_sequence(self):
        time_series = [1, 1, -1, 1, 1]
        num_predicted_periods = 3
        expected_prediction = [0.8] * num_predicted_periods
        hw = HoltWintersPredictor(time_series, num_predicted_periods)

        actual_prediction = hw.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=3)

    def test_zeros_in_sequence(self):
        time_series = [1, 1, 0, 1, 1]
        num_predicted_periods = 3
        expected_prediction = [0.8] * num_predicted_periods
        hw = HoltWintersPredictor(time_series, num_predicted_periods)

        actual_prediction = hw.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=3)

    def test_static_sequence(self):
        time_series = [1.0, 1.0, 1.0, 1.0, 1.0]
        num_predicted_periods = 3
        expected_prediction = [1] * num_predicted_periods
        hw = HoltWintersPredictor(time_series, num_predicted_periods)

        actual_prediction = hw.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=3)
