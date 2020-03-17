import unittest
import numpy as np
import os

from scripts.algorithms.ssm import StateSpaceModel


class StateSpaceModelTests(unittest.TestCase):

    def test_run_smoothing(self):
        quarterly_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            4, 3]
        expected_smooth_series_no_negatives = np.array([0] * len(quarterly_values))
        expected_derivatives = np.array([0] * len(quarterly_values))
        _, _1, smooth_series_s, _intercept = StateSpaceModel(quarterly_values).run_smoothing()
        smooth_series = smooth_series_s[0].tolist()[0]
        smooth_series_no_negatives = np.clip(smooth_series, a_min=0, a_max=None)
        derivatives = smooth_series_s[1].tolist()[0]

        if os.name == 'nt':
            np.testing.assert_almost_equal(expected_smooth_series_no_negatives, smooth_series_no_negatives)
            np.testing.assert_almost_equal(expected_derivatives, derivatives)

    def test_run_smoothing_single_one(self):
        quarterly_values = [1] + ([0] * 1000)
        _, _1, smooth_series_s, _intercept = StateSpaceModel(quarterly_values).run_smoothing()
