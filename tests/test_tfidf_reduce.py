import os
import unittest

import pandas as pd

from scripts.pipeline import Pipeline
from scripts.utils.date_utils import date_to_year_week
from tests import support


class TestTfidfReduce(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
        docs_mask_dict['date_header'] = 'publication_date'

        filename = os.path.join('tests', 'data', 'USPTO-random-100.csv')

        cls.__pipeline = Pipeline(filename, docs_mask_dict, ngram_range=ngram_range,
                                  text_header='abstract', term_counts=True,
                                  max_df=max_df, output_name='test')

        cls.__term_score_tuples = cls.__pipeline.term_score_tuples

    def test_terms(self):
        term_score_tuples = self.__term_score_tuples
        actual_terms = [x for _, x in term_score_tuples]
        expected_terms = ['mounting surface',
                          'transmit path',
                          'electronic element',
                          'link document',
                          'amide derivative',
                          'valproic acid',
                          'voice message',
                          'jack mechanism',
                          'pharmaceutical formulation',
                          'light beam',
                          'angular velocity',
                          'contact beam',
                          'conductive material',
                          'endodontic instrument',
                          'mass offset',
                          'section bend',
                          'component material',
                          'terminal channel',
                          'stationary household appliance',
                          'fault point'
                          ]

        self.assertListEqual(actual_terms[:20], expected_terms)

    def test_scores(self):
        term_score_tuples = self.__term_score_tuples
        actual_scores = [x for x, _ in term_score_tuples]
        expected_scores = [0.8728715609439694,
                           0.8259734063804905,
                           0.7754588414852185,
                           0.7620007620011429,
                           0.7071067811865476,
                           0.7071067811865476,
                           0.7071067811865475,
                           0.6882472016116852,
                           0.6666666666666666,
                           0.6246950475544241,
                           0.6198903382379372,
                           0.6031800939323297,
                           0.5806718350868961,
                           0.5773502691896257,
                           0.5773502691896257,
                           0.5773502691896257,
                           0.5669467095138407,
                           0.5597177778726654,
                           0.5570860145311556,
                           0.5568900989230109]

        support.assert_list_almost_equal(self, actual_scores[:20], expected_scores)

    def test_timeseries_mat(self):
        timeseries_mat = self.__pipeline.term_counts_data
        self.assertEqual(sum(timeseries_mat[2]), 100)
