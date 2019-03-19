import os
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from scripts.algorithms.emergence import Emergence
from scripts.utils.utils import get_row_indices_and_values


class EmergenceTests(unittest.TestCase):

    # temp method to find term ids to use in tests
    def find_term_index(self, term):
        for term_index in range(0, len(self.term_ngrams)):
            if self.term_ngrams[term_index] == term:
                return term_index
        self.fail(f'Failed to find term {term}')

    @classmethod
    def setUpClass(cls):
        [cls.term_counts_per_week, cls.term_ngrams, cls.num_patents_per_week, cls.week_iso_dates] = pd.read_pickle(
            os.path.join('data', 'USPTO-random-500000-term_counts.pkl.bz2'))
        cls.term_counts_per_week_csc = cls.term_counts_per_week.tocsc()
        cls.em = Emergence(cls.num_patents_per_week)

    def escore_wrapper(self, term_counts_per_week_csc, term_index):
        term_weeks, term_counts = get_row_indices_and_values(term_counts_per_week_csc, term_index)
        if self.em.init_vars(term_weeks, term_counts):
            return self.em.calculate_escore()
        return 0

    def emergence_wrapper(self, term_counts_per_week_csc, term_index):
        term_weeks, term_counts = get_row_indices_and_values(term_counts_per_week_csc, term_index)
        if self.em.init_vars(term_weeks, term_counts):
            return self.em.calculate_emergence(term_weeks)
        return False

    def set_up_emergent_term(self):
        # Aim:
        # escore = 2 * active period trend + recent trend + mid-year to last year slope
        # active period trend = (term counts 5+6+7)/(sqrt(total 5) + sqrt(total 6) + sqrt(total 7)
        #   - (term counts 1+2+3)/(sqrt(total 1) + sqrt(total 2) + sqrt(total 3))
        # recent trend = 10 * (term count 6+7)/(sqrt(total 6) + sqrt(total 7))
        #   - (term counts 4 + 5)/(sqrt(total 4) + sqrt(total 5))
        # mid-year to last year slope = 10 * ((term counts 7 / sqrt(total 7)) - (term counts 4/sqrt(total 4))) / (7-4)
        #
        # Also: emergent if:
        # term present for >3 years
        # >7 docs with term
        # # term records in active / # term records in base
        # # base term records / # base all records < 15%
        # single author set...
        weeks_per_period = 52
        self.term_counts_matrix = np.zeros(shape=(weeks_per_period * 10, 9), dtype=np.int)

        # [0: emergent term, 1: non-emergent due to base, 2: non-emergent constant count,
        #  3: non-emergent decreasing count, 4: not 10 years data, 5: background, 6: background,
        #  7: two occurrences over 10 years, 8: all but one term occurs in base]

        # period 1 - base
        self.term_counts_matrix[0, :] = [1, 0, 1, 1, 0, 0, 0, 1, 1]
        self.term_counts_matrix[2, :] = [0, 1, 0, 1, 0, 1, 0, 0, 1]

        # period 2 - base
        self.term_counts_matrix[0 + (1 * weeks_per_period), :] = [0, 0, 0, 1, 0, 0, 0, 0, 1]
        self.term_counts_matrix[3 + (1 * weeks_per_period), :] = [0, 0, 1, 1, 0, 1, 1, 0, 1]
        self.term_counts_matrix[7 + (1 * weeks_per_period), :] = [0, 1, 0, 1, 1, 0, 0, 0, 1]

        # period 3 - base
        self.term_counts_matrix[12 + (2 * weeks_per_period), :] = [0, 0, 0, 1, 0, 0, 0, 0, 1]
        self.term_counts_matrix[20 + (2 * weeks_per_period), :] = [0, 0, 1, 1, 0, 1, 1, 0, 1]
        self.term_counts_matrix[30 + (2 * weeks_per_period), :] = [0, 1, 0, 0, 1, 0, 0, 0, 1]

        # period 4 - active
        self.term_counts_matrix[5 + (3 * weeks_per_period), :] = [1, 0, 0, 1, 0, 0, 0, 0, 0]
        self.term_counts_matrix[8 + (3 * weeks_per_period), :] = [0, 0, 1, 0, 0, 1, 1, 0, 0]
        self.term_counts_matrix[9 + (3 * weeks_per_period), :] = [0, 1, 0, 1, 1, 0, 0, 0, 0]

        # period 5 - active
        self.term_counts_matrix[20 + (4 * weeks_per_period), :] = [1, 0, 0, 1, 0, 0, 0, 0, 0]
        self.term_counts_matrix[40 + (4 * weeks_per_period), :] = [0, 0, 1, 1, 0, 1, 1, 0, 0]
        self.term_counts_matrix[51 + (4 * weeks_per_period), :] = [1, 1, 0, 0, 1, 1, 0, 0, 0]

        # period 6 - active
        self.term_counts_matrix[10 + (5 * weeks_per_period), :] = [1, 0, 0, 1, 0, 0, 0, 0, 0]
        self.term_counts_matrix[11 + (5 * weeks_per_period), :] = [1, 1, 1, 0, 0, 1, 1, 0, 0]
        self.term_counts_matrix[12 + (5 * weeks_per_period), :] = [0, 1, 0, 0, 1, 0, 0, 0, 0]

        # period 7 - active
        self.term_counts_matrix[21 + (6 * weeks_per_period), :] = [1, 0, 0, 1, 0, 0, 0, 0, 0]
        self.term_counts_matrix[32 + (6 * weeks_per_period), :] = [0, 1, 1, 0, 0, 1, 1, 0, 0]
        self.term_counts_matrix[43 + (6 * weeks_per_period), :] = [1, 1, 0, 0, 1, 0, 0, 0, 0]

        # period 8 - active
        self.term_counts_matrix[12 + (7 * weeks_per_period), :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.term_counts_matrix[13 + (7 * weeks_per_period), :] = [1, 0, 1, 1, 0, 1, 1, 0, 0]
        self.term_counts_matrix[14 + (7 * weeks_per_period), :] = [1, 1, 0, 0, 1, 0, 0, 0, 0]

        # period 9 - active
        self.term_counts_matrix[28 + (8 * weeks_per_period), :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.term_counts_matrix[29 + (8 * weeks_per_period), :] = [1, 1, 1, 0, 0, 1, 1, 0, 0]
        self.term_counts_matrix[51 + (8 * weeks_per_period), :] = [1, 1, 0, 0, 1, 0, 0, 0, 0]

        # period 10 - active
        self.term_counts_matrix[49 + (9 * weeks_per_period), :] = [1, 1, 0, 1, 0, 0, 0, 0, 0]
        self.term_counts_matrix[50 + (9 * weeks_per_period), :] = [1, 1, 1, 0, 0, 1, 1, 0, 0]
        self.term_counts_matrix[51 + (9 * weeks_per_period), :] = [1, 1, 0, 0, 1, 0, 0, 1, 0]

        self.term_counts_per_week_csc = csc_matrix(self.term_counts_matrix)

        self.num_patents_per_week = self.term_counts_matrix.sum(axis=1) > 0
        self.num_patents_per_week = self.num_patents_per_week.astype(dtype=np.int32)

        self.em = Emergence(self.num_patents_per_week)

    def test_emergent(self):
        escore_expected = 6.35
        self.set_up_emergent_term()
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, 0)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, 0)
        self.assertTrue(potentially_emergent_actual)
        self.assertAlmostEqual(escore_expected, escore_actual, places=2)

    def test_term_with_less_than_10_years_data(self):
        escore_expected = 0
        self.set_up_emergent_term()
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, 4)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, 4)
        self.assertFalse(potentially_emergent_actual)
        self.assertAlmostEqual(escore_expected, escore_actual, places=2)

    def test_term_with_less_than_7_occurrences(self):
        escore_expected = 0
        self.set_up_emergent_term()
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, 7)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, 7)
        self.assertFalse(potentially_emergent_actual)
        self.assertAlmostEqual(escore_expected, escore_actual, places=2)

    def test_term_counts_base2all_over_threshold_and_emergent(self):
        escore_expected = 6.35
        self.set_up_emergent_term()
        self.em.TERM_BASE_RECS_THRESHOLD = 1
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, 0)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, 0)
        self.assertTrue(potentially_emergent_actual)
        self.assertAlmostEqual(escore_expected, escore_actual, places=2)

    def test_term_counts_base2all_over_threshold_but_not_emergent(self):
        escore_expected = 9.24
        self.set_up_emergent_term()
        self.em.TERM_BASE_RECS_THRESHOLD = 1
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, 1)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, 1)
        self.assertFalse(potentially_emergent_actual)
        self.assertAlmostEqual(escore_expected, escore_actual, places=2)

    def test_term_with_base_but_no_emergent_instances(self):
        escore_expected = 0
        self.set_up_emergent_term()
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, 8)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, 8)
        self.assertFalse(potentially_emergent_actual)
        self.assertAlmostEqual(escore_expected, escore_actual, places=2)

    def test_non_emergent_with_constant_usage_term(self):
        escore_expected = 0
        self.set_up_emergent_term()
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, 2)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, 2)
        self.assertIsInstance(potentially_emergent_actual, bool)
        self.assertAlmostEqual(escore_expected, escore_actual, places=2)

    def test_non_emergent_with_decreasing_usage_term(self):
        escore_expected = -4.04
        self.set_up_emergent_term()
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, 3)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, 3)
        self.assertFalse(potentially_emergent_actual)
        self.assertAlmostEqual(escore_expected, escore_actual, places=2)

    def test_3d_image(self):
        term = '3d image'
        term_index = self.find_term_index(term)
        escore_expected = 1.356549020713896
        em_expected = True
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, term_index)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, term_index)
        self.assertEqual(em_expected, potentially_emergent_actual, term + ": em failed")
        self.assertEqual(escore_expected, escore_actual, term + ": escore failed")

    def test_3d_display(self):
        term = '3d display'
        term_index = self.find_term_index(term)
        escore_expected = 0.6398282485477932
        em_expected = True
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, term_index)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, term_index)
        self.assertEqual(em_expected, potentially_emergent_actual, term + ": em failed")
        self.assertEqual(escore_expected, escore_actual, term + ": escore failed")

    def test_ac_power_supply(self):
        term = 'ac power supply'
        term_index = self.find_term_index(term)
        escore_expected = 0.22047218851712613
        em_expected = True
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, term_index)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, term_index)
        self.assertEqual(em_expected, potentially_emergent_actual, term + ": em failed")
        self.assertEqual(escore_expected, escore_actual, term + ": escore failed")

    def test_acid_molecule(self):
        term = 'acid molecule'
        term_index = self.find_term_index(term)
        escore_expected = -1.5641832751818254
        em_expected = False
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, term_index)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, term_index)
        self.assertEqual(em_expected, potentially_emergent_actual, term + ": em failed")
        self.assertEqual(escore_expected, escore_actual, term + ": escore failed")

    def test_acid_molecule_encoding(self):
        term = 'acid molecule encoding'
        term_index = self.find_term_index(term)
        escore_expected = -0.18173715415163488
        em_expected = False
        potentially_emergent_actual = self.emergence_wrapper(self.term_counts_per_week_csc, term_index)
        escore_actual = self.escore_wrapper(self.term_counts_per_week_csc, term_index)
        self.assertEqual(em_expected, potentially_emergent_actual, term + ": em failed")
        self.assertEqual(escore_expected, escore_actual, term + ": escore failed")
