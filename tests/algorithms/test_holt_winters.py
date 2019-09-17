import unittest

import numpy.testing as np_test

from scripts.algorithms.holtwinters_predictor import HoltWintersPredictor


class HoltWintersTests(unittest.TestCase):

    def test_negatives_in_sequence(self):
        time_series = [1, 1, -1, 1, 1]
        num_predicted_periods = 3

        try:
            HoltWintersPredictor(time_series, num_predicted_periods)
            self.fail('Expected to throw due to negative values')

        except NotImplementedError as nie:
            self.assertEqual(nie.args[0], 'Unable to correct for negative or zero values')

        except ValueError as ve:
            self.assertEqual(ve.args[0],
                             'endog must be strictly positive when using multiplicative trend or seasonal components.')

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
