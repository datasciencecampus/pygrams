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
        
        expected_terms = ['hard disk drive',
'stationary household appliance',
'plurality transmit path',
'handheld electronic device',
'light emit diode',
'organic light emit',
'production pharmaceutical formulation',
'electron source substrate',
'specify laser diode',
'acid provide treatment',
'amide derivative valproic',
'aqueous polymer coat',
'coat superparamagnetic nanoparticles',
'derivative valproic acid',
'disclose aqueous polymer',
'polymer coat superparamagnetic',
'provide treatment epilepsy',
'superparamagnetic nanoparticles nanoparticles',
'valproic acid provide',
'signal conversion device'
                          ]

        self.assertListEqual(actual_terms[:20], expected_terms)

    def test_scores(self):
        term_score_tuples = self.__term_score_tuples
        actual_scores = [x for x, _ in term_score_tuples]
        expected_scores = [0.4411597312432845,
                           0.42008401133852175,
                           0.39902364202939306,
                           0.3750831625076115,
                           0.321230276498425,
                           0.321230276498425,
                           0.32025631050624814,
                           0.31905076990550885,
                           0.30337887539713954,
                           0.3015113476990308,
                           0.3015113476990308,
                           0.3015113476990308,
                           0.3015113476990308,
                           0.3015113476990308,
                           0.3015113476990308,
                           0.3015113476990308,
                           0.3015113476990308,
                           0.3015113476990308,
                           0.3015113476990308,
                           0.29584763849200485]


        support.assert_list_almost_equal(self, actual_scores[:20], expected_scores)

    def test_timeseries_mat(self):
        timeseries_data = self.__pipeline.timeseries_data
        self.assertEqual(sum(timeseries_data[2]), 100)
