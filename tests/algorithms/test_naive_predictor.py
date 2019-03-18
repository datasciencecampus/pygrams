from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")

import unittest

import numpy.testing as np_test

from scripts.algorithms.naive_predictor import NaivePredictor


class NaivePredictorTests(unittest.TestCase):

    def test_static_sequence(self):
        time_series = [1.0, 1.0, 1.0, 1.0, 1.0]
        num_predicted_periods = 3
        expected_prediction = [1] * num_predicted_periods
        predictor = NaivePredictor(time_series, num_predicted_periods)

        actual_prediction = predictor.predict_counts()

        np_test.assert_almost_equal(actual_prediction, expected_prediction, decimal=4)
