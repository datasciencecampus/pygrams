from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")

import math
import os
import unittest

import numpy as np
import pandas as pd

from scripts.algorithms.lstm import LstmForecasterMultipleModelSingleLookAhead


class LstmTest(unittest.TestCase):

    def setUp(self):
        self.look_back = 20
        self.look_ahead = 5

    @staticmethod
    def import_terms():

        def sum_by_weeks(week_counts, num_weeks):
            return np.array(
                [week_counts[i:i + num_weeks].sum(dtype=np.int32) for i in range(0, len(week_counts), num_weeks)])

        [term_counts_per_week, term_ngrams, number_of_patents_per_week, week_iso_dates] = pd.read_pickle(
            os.path.join('data', 'USPTO-granted-full-random-500000-term_counts.pkl.bz2'))

        term_counts_per_week_csc = term_counts_per_week.tocsc()

        # term_ngram = 'optical disc'  # 2797322 -2.45
        # term_ngram = 'plasma display'  # 3027785 -2.21
        # term_ngram = 'mobile device'  # 2533111 13.16
        term_ngram = 'control unit'  # 838925 12.35
        # term_ngram = 'display device'  # 1159899 5.42

        term_index = [ti for ti, tn in enumerate(term_ngrams) if tn == term_ngram][0]
        term_ngram = term_ngrams[term_index]

        occurrence_per_week = term_counts_per_week[:, term_index].toarray()
        # occurrence_time_series = occurrence_per_week

        occurrence_per_quarter = sum_by_weeks(occurrence_per_week, 12)
        occurrence_time_series = occurrence_per_quarter

        number_of_patents_per_quarter = sum_by_weeks(np.array(number_of_patents_per_week), 12)
        pass

    # number_of_patents_per_quarter
    # 5323, 5567, 5140, 4566, 4940, 6862, 6443, 6535, 5762, 5947, 5541, 5436, 5466, 5164, 6197, 5351, 5606, 5364, 6121, 5893, 6104, 6310, 7492, 8227, 8269, 7745, 7921, 7671, 8082, 8341, 7950, 9003, 9063, 9545, 9209, 9416, 9561, 10028, 10509, 8787, 11353, 11433, 11199, 9445, 10919, 10312, 10947, 10658, 10896, 11129, 10905, 10675, 10478, 11650, 11461, 11750, 10284, 11048, 11069, 9891

    def verify_lstm(self, occurrence_time_series, maximum_error_delta, term_name):
        occurrence_time_series = occurrence_time_series[0:-1]

        lstm_forecaster = LstmForecasterMultipleModelSingleLookAhead(occurrence_time_series, self.look_back,
                                                                     self.look_ahead, term_name, random_seed=0)

        seed = occurrence_time_series[:-self.look_ahead]
        actual = occurrence_time_series[-self.look_ahead:].ravel()
        prediction = lstm_forecaster.predict_counts(seed)

        lstm_forecaster.plot_prediction(occurrence_time_series, prediction)

        number_of_errors = 0
        maximum_error = 0
        for index in range(self.look_ahead):
            abs_error = math.fabs(actual[index] - prediction[index])
            if abs_error > maximum_error:
                maximum_error = abs_error
            if abs_error > maximum_error_delta:
                number_of_errors += 1

        return number_of_errors, maximum_error

    def test_simple_sine_wave(self):
        occurrence_time_series = np.array([math.sin(math.radians(angle)) for angle in range(0, 360 * 5, 5)])

        maximum_error_delta = 0.05
        number_of_errors, maximum_error = self.verify_lstm(occurrence_time_series, maximum_error_delta,
                                                           'Sine wave')
        self.assertLess(maximum_error, 0.005)
        self.assertEqual(number_of_errors, 0)

    def test_simple_sine_wave_long_look_behind(self):
        occurrence_time_series = np.array([math.sin(math.radians(angle)) for angle in range(0, 360 * 5, 5)])

        self.look_back = 30
        maximum_error_delta = 0.05
        number_of_errors, maximum_error = self.verify_lstm(occurrence_time_series, maximum_error_delta,
                                                           'Sine wave')
        self.assertLess(maximum_error, 0.005)
        self.assertEqual(number_of_errors, 0)

    def test_simple_sine_wave_long_look_behind_and_ahead(self):
        occurrence_time_series = np.array([math.sin(math.radians(angle)) for angle in range(0, 360 * 5, 5)])

        self.look_back = 30
        self.look_ahead = 10
        maximum_error_delta = 0.05
        number_of_errors, maximum_error = self.verify_lstm(occurrence_time_series, maximum_error_delta,
                                                           'Sine wave')
        self.assertLess(maximum_error, 0.015)
        self.assertEqual(number_of_errors, 0)

    def test_complex_sine_wave(self):
        occurrence_time_series = np.array([math.sin(math.radians(angle) + math.sin(math.radians(angle * 3)))
                                           for angle in range(0, 360 * 5, 5)])

        maximum_error_delta = 0.05
        number_of_errors, maximum_error = self.verify_lstm(occurrence_time_series, maximum_error_delta,
                                                           'Complex sine wave')
        self.assertLess(maximum_error, 0.015)
        self.assertEqual(number_of_errors, 0)

    def test_complex_sine_wave_long_look_behind(self):
        occurrence_time_series = np.array([math.sin(math.radians(angle) + math.sin(math.radians(angle * 3)))
                                           for angle in range(0, 360 * 5, 5)])

        self.look_back = 30
        maximum_error_delta = 0.05
        number_of_errors, maximum_error = self.verify_lstm(occurrence_time_series, maximum_error_delta,
                                                           'Complex sine wave')
        self.assertLess(maximum_error, 0.015)
        self.assertEqual(number_of_errors, 0)

    def test_complex_sine_wave_long_look_behind_and_ahead(self):
        occurrence_time_series = np.array([math.sin(math.radians(angle) + math.sin(math.radians(angle * 3)))
                                           for angle in range(0, 360 * 5, 5)])

        self.look_back = 30
        self.look_ahead = 10
        maximum_error_delta = 0.05
        number_of_errors, maximum_error = self.verify_lstm(occurrence_time_series, maximum_error_delta,
                                                           'Complex sine wave')
        self.assertLess(maximum_error, 0.015)
        self.assertEqual(number_of_errors, 0)

    def test_patent_occurrence_display_device_per_quarter(self):
        # Emergence score: 5.42
        display_device_per_quarter = np.array(
            [42, 52, 39, 33, 39, 52, 56, 61, 40, 62, 77, 41, 39, 42, 56, 50, 50, 52, 71, 46, 69, 62, 85, 90, 80, 75, 72,
             68, 89, 111, 79, 106, 100, 110, 107, 108, 94, 106, 116, 108, 130, 126, 129, 117, 131, 126, 145, 144, 145,
             162, 142, 179, 149, 176, 178, 186, 169, 174, 182, 146])

        maximum_error_delta = 5
        number_of_errors, maximum_error = self.verify_lstm(display_device_per_quarter, maximum_error_delta,
                                                           'display device')
        self.assertLess(maximum_error, 0.015)
        self.assertEqual(number_of_errors, 0)

    def test_patent_occurrence_control_unit_per_quarter(self):
        # Emergence score: 12.35
        control_unit_per_quarter = np.array(
            [49, 48, 56, 43, 37, 70, 57, 67, 66, 54, 56, 51, 54, 50, 48, 43, 69, 58, 74, 69, 74, 72, 97, 91, 95, 104,
             90, 88, 109, 112, 122, 133, 129, 131, 115, 138, 155, 136, 152, 129, 175, 197, 174, 158, 185, 205, 183, 193,
             187, 179, 173, 192, 180, 208, 192, 209, 183, 186, 163, 163])

        maximum_error_delta = 5
        number_of_errors, maximum_error = self.verify_lstm(control_unit_per_quarter, maximum_error_delta,
                                                           'control unit')
        self.assertLess(maximum_error, 0.015)
        self.assertEqual(number_of_errors, 0)

    def test_patent_occurrence_mobile_device_per_quarter(self):
        # Emergence score: 13.16
        mobile_device_per_quarter = np.array(
            [2, 2, 2, 5, 9, 5, 5, 6, 5, 9, 8, 6, 9, 13, 4, 8, 9, 11, 12, 8, 10, 12, 19, 18, 16, 23, 17, 22, 17, 31, 34,
             37, 34, 43, 54, 52, 57, 58, 73, 65, 80, 95, 102, 58, 102, 89, 125, 105, 120, 122, 115, 118, 102, 123, 121,
             130, 113, 124, 119, 92])

        maximum_error_delta = 5
        number_of_errors, maximum_error = self.verify_lstm(mobile_device_per_quarter, maximum_error_delta,
                                                           'mobile device')
        self.assertLess(maximum_error, 0.015)
        self.assertEqual(number_of_errors, 0)

    def test_patent_occurrence_plasma_display_per_quarter(self):
        # Emergence score: -2.21
        plasma_display_per_quarter = np.array(
            [6, 11, 5, 5, 6, 10, 8, 11, 10, 6, 8, 6, 3, 13, 7, 9, 7, 16, 13, 8, 15, 11, 12, 16, 7, 9, 7, 13, 7, 11, 7,
             10, 6, 3, 5, 7, 4, 4, 5, 2, 0, 0, 3, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

        maximum_error_delta = 5
        number_of_errors, maximum_error = self.verify_lstm(plasma_display_per_quarter, maximum_error_delta,
                                                           'plasma display')
        self.assertLess(maximum_error, 0.015)
        self.assertEqual(number_of_errors, 0)

    def test_patent_occurrence_optical_disc_per_quarter(self):
        # Emergence score: -2.45
        optical_disc_per_quarter = np.array(
            [3, 6, 9, 4, 7, 15, 6, 11, 12, 4, 8, 13, 9, 11, 11, 8, 14, 8, 7, 15, 6, 11, 10, 7, 15, 15, 11, 16, 17, 10,
             9, 10, 20, 12, 12, 10, 4, 7, 11, 10, 5, 2, 4, 2, 3, 0, 2, 2, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0])

        maximum_error_delta = 5
        number_of_errors, maximum_error = self.verify_lstm(optical_disc_per_quarter, maximum_error_delta,
                                                           'optical disc')
        self.assertLess(maximum_error, 0.015)
        self.assertEqual(number_of_errors, 0)


# if __name__ == "__main__":
#
#     lstm_test = LstmTest()
#     lstm_test.import_terms()


if __name__ == '__main__':
    unittest.main()
