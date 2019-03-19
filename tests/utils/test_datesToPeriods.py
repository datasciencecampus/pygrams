import unittest

import numpy as np
import numpy.testing as npt

from scripts.utils.date_utils import tfidf_with_dates_to_weekly_term_counts


class test_usptoDatesToPeriods(unittest.TestCase):

    @staticmethod
    def run_test_with_conversion(combined_array):
        numpy_matrix = np.array(combined_array)
        tfidf_matrix = np.array(numpy_matrix[:, 1:])
        publication_week_dates = np.array(numpy_matrix[:, 0])
        return tfidf_with_dates_to_weekly_term_counts(tfidf_matrix, publication_week_dates)

    def test_week_single_entry(self):
        tfidf = [
            [200801, 0, 0.3, 0],
        ]
        expected_term_counts = np.array([
            [0, 1, 0],
        ])
        expected_term_totals = [1]
        expected_week_dates = [200801]
        actual_term_counts, actual_term_totals, actual_week_dates = self.run_test_with_conversion(tfidf)
        npt.assert_array_equal(expected_term_counts, actual_term_counts.todense())
        npt.assert_array_equal(expected_term_totals, actual_term_totals)
        npt.assert_array_equal(expected_week_dates, actual_week_dates)

    def test_week_combining_and_gaps(self):
        combined_array = [
            [200801, 0, 0.3, 0],
            [200801, 0, 0, 0],
            [200802, 0, 0, 2.3],
            [200804, 0.1, 0.3, 0],
            [200804, 0.2, 0, 0.1],
            [200806, 0, 0.3, 0],
        ]
        expected_term_counts = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [2, 1, 1],
            [0, 0, 0],
            [0, 1, 0]
        ])
        expected_term_totals = [2, 1, 0, 2, 0, 1]
        expected_week_dates = [200801, 200802, 200803, 200804, 200805, 200806]
        actual_term_counts, actual_term_totals, actual_week_dates = self.run_test_with_conversion(combined_array)
        actual_term_counts_dense = actual_term_counts.todense()
        npt.assert_array_equal(expected_term_counts, actual_term_counts_dense)
        npt.assert_array_equal(expected_term_totals, actual_term_totals)
        npt.assert_array_equal(expected_week_dates, actual_week_dates)

    def test_week_combining_and_split_across_years(self):
        combined_array = [
            [200850, 0, 0.3, 0],
            [200852, 0, 0, 0],
            [200852, 0, 0, 2.3],
            [200902, 0.1, 0.3, 0],
            [200902, 0.2, 0, 0.1],
            [200904, 0, 0.3, 0],
        ]
        expected_term_counts = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [2, 1, 1],
            [0, 0, 0],
            [0, 1, 0]
        ])
        expected_week_dates = [200850, 200851, 200852, 200901, 200902, 200903, 200904]
        expected_term_totals = [1, 0, 2, 0, 2, 0, 1]
        actual_term_counts, actual_term_totals, actual_week_dates = self.run_test_with_conversion(combined_array)
        actual_term_counts_dense = actual_term_counts.todense()
        npt.assert_array_equal(expected_term_counts, actual_term_counts_dense)
        npt.assert_array_equal(expected_term_totals, actual_term_totals)
        npt.assert_array_equal(expected_week_dates, actual_week_dates)

    def test_week_combining_and_split_across_years_create_up_to_week_52(self):
        combined_array = [
            [200850, 0, 0.3, 0],
            [200850, 0, 0, 0],
            [200850, 0, 0, 2.3],
            [200902, 0.1, 0.3, 0],
            [200902, 0.2, 0, 0.1],
            [200904, 0, 0.3, 0],
        ]
        expected_term_counts = np.array([
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [2, 1, 1],
            [0, 0, 0],
            [0, 1, 0]
        ])
        expected_week_dates = [200850, 200851, 200852, 200901, 200902, 200903, 200904]
        expected_term_totals = [3, 0, 0, 0, 2, 0, 1]
        actual_term_counts, actual_term_totals, actual_week_dates = self.run_test_with_conversion(combined_array)
        npt.assert_array_equal(expected_term_counts, actual_term_counts.todense())
        npt.assert_array_equal(expected_term_totals, actual_term_totals)
        npt.assert_array_equal(expected_week_dates, actual_week_dates)

    def test_week_combining_and_split_across_years_include_empty_week_53(self):
        combined_array = [
            [200850, 0, 0.3, 0],
            [200852, 0, 0, 0],
            [200852, 0, 0, 2.3],
            [200853, 0, 0, 0],
            [200902, 0.1, 0.3, 0],
            [200902, 0.2, 0, 0.1],
            [200904, 0, 0.3, 0],
        ]
        expected_term_counts = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [2, 1, 1],
            [0, 0, 0],
            [0, 1, 0]
        ])
        expected_week_dates = [200850, 200851, 200852, 200853, 200901, 200902, 200903, 200904]
        expected_term_totals = [1, 0, 2, 1, 0, 2, 0, 1]
        actual_term_counts, actual_term_totals, actual_week_dates = self.run_test_with_conversion(combined_array)
        npt.assert_array_equal(expected_term_counts, actual_term_counts.todense())
        npt.assert_array_equal(expected_term_totals, actual_term_totals)
        npt.assert_array_equal(expected_week_dates, actual_week_dates)

    def test_week_combining_and_split_across_years_include_non_zero_week_53(self):
        combined_array = [
            [200850, 0, 0.3, 0],
            [200852, 0, 0, 0],
            [200852, 0, 0, 2.3],
            [200853, 0, 1, 0],
            [200902, 0.1, 0.3, 0],
            [200902, 0.2, 0, 0.1],
            [200904, 0, 0.3, 0],
        ]
        expected_term_counts = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 0],
            [2, 1, 1],
            [0, 0, 0],
            [0, 1, 0]
        ])
        expected_week_dates = [200850, 200851, 200852, 200853, 200901, 200902, 200903, 200904]
        expected_term_totals = [1, 0, 2, 1, 0, 2, 0, 1]
        actual_term_counts, actual_term_totals, actual_week_dates = self.run_test_with_conversion(combined_array)
        npt.assert_array_equal(expected_term_counts, actual_term_counts.todense())
        npt.assert_array_equal(expected_term_totals, actual_term_totals)
        npt.assert_array_equal(expected_week_dates, actual_week_dates)
