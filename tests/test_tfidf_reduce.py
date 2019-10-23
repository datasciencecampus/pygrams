import os
import unittest

import pandas as pd

from scripts.pipeline import Pipeline
from scripts.utils.date_utils import date_to_year_week
from tests import support


class TestTfidfReduce(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        min_n = 2
        max_n = 3
        max_df = 0.3
        ngram_range = (min_n, max_n)

        date_to = date_to_year_week(pd.to_datetime('today').date())
        date_from = date_to_year_week(pd.to_datetime('1900-01-01').date())

        docs_mask_dict = {}
        docs_mask_dict['filter_by'] = 'union'
        docs_mask_dict['cpc'] = None
        docs_mask_dict['time'] = None
        docs_mask_dict['cite'] = []
        docs_mask_dict['columns'] = None
        docs_mask_dict['date'] = {
            'to': date_to,
            'from': date_from
        }
        docs_mask_dict['timeseries_date'] = {
            'to': date_to,
            'from': date_from
        }
        docs_mask_dict['date_header'] = 'publication_date'

        filename = os.path.join('data', 'USPTO-random-100.csv')

        cls.__pipeline = Pipeline(filename, docs_mask_dict, ngram_range=ngram_range, text_header='abstract',
                                  max_df=max_df, output_name='test', calculate_timeseries=True)

        cls.__term_score_tuples = cls.__pipeline.term_score_tuples

    def test_terms(self):
        term_score_tuples = self.__term_score_tuples
        actual_terms = [x for _, x in term_score_tuples]

        expected_terms = ['electronic device',
                          'reaction product',
                          'first second',
                          'system provide',
                          'support member',
                          'substrate surface',
                          'unit couple',
                          'data line',
                          'unit form',
                          'light emit',
                          'second layer',
                          'exhaust gas',
                          'breast cancer',
                          'anchor anchor',
                          'bottom face',
                          'pharmaceutical composition',
                          'resource define',
                          'update electronic',
                          'communication device',
                          'physical contact'
                          ]

        self.assertListEqual(actual_terms[:20], expected_terms)

    def test_scores(self):
        term_score_tuples = self.__term_score_tuples
        actual_scores = [x for x, _ in term_score_tuples]
        expected_scores = [
                            3.0527220691623183,
                            2.9999999706316087,
                            2.2381454887637147,
                            2.230131148163388,
                            2.154072530925622,
                            1.9999999747370003,
                            1.9428090291641258,
                            1.8645696190816343,
                            1.8164965663393273,
                            1.7909478939875219,
                            1.778715966795346,
                            1.7409794973448056,
                            1.7071068010737243,
                            1.7071067687152195,
                            1.7071067687152195,
                            1.7071067687152195,
                            1.7071067687152195,
                            1.7071067687152195,
                            1.7003553944678127,
                            1.6631362012358317]

        support.assert_list_almost_equal(self, actual_scores[:20], expected_scores[:20])

    def test_timeseries_mat(self):
        timeseries_data = self.__pipeline.timeseries_data
        self.assertEqual(sum(timeseries_data[2]), 62)