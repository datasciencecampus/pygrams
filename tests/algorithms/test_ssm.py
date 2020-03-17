import unittest

from scripts.algorithms.ssm import StateSpaceModel


class StateSpaceModelTests(unittest.TestCase):

    def test_run_smoothing(self):
        quarterly_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            4, 3]
        _, _1, smooth_series_s, _intercept = StateSpaceModel(quarterly_values).run_smoothing()

    def test_run_smoothing_single_one(self):
        quarterly_values = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0]
        _, _1, smooth_series_s, _intercept = StateSpaceModel(quarterly_values).run_smoothing()
