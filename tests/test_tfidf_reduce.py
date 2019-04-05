import unittest
import os
import numpy as np
import pandas as pd
import scripts.data_factory as factory
from scripts import FilePaths
from scripts.filter_terms import FilterTerms
from scripts.text_processing import StemTokenizer
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import TFIDF
from scripts.utils import utils
from tests import support


class TestTfidfReduce(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        min_n = 2
        max_n = 3
        max_df = 0.3
        ngram_range = (min_n, max_n)
        filename = os.path.join('tests', 'data', 'USPTO-random-100.csv')
        cls.__df = factory.get(filename)
        tfidf_obj = TFIDF(cls.__df['abstract'], ngram_range=ngram_range, max_document_frequency=max_df,
                          tokenizer=StemTokenizer())

        doc_weights = list(np.ones(len(cls.__df)))

        # term weights - embeddings
        filter_output_obj = FilterTerms(tfidf_obj.feature_names, None, None)
        term_weights = filter_output_obj.ngram_weights_vec

        tfidf_mask_obj = TfidfMask(tfidf_obj, ngram_range=ngram_range)
        tfidf_mask_obj.update_mask(doc_weights, term_weights)
        tfidf_mask = tfidf_mask_obj.tfidf_mask

        # mask the tfidf matrix
        tfidf_matrix = tfidf_obj.tfidf_matrix
        tfidf_masked = tfidf_mask.multiply(tfidf_matrix)
        tfidf_masked = utils.remove_all_null_rows(tfidf_masked)

        print(f'Processing TFIDF matrix of {tfidf_masked.shape[0]:,} / {tfidf_matrix.shape[0]:,} documents')

        cls.__tfidf_reduce_obj = TfidfReduce(tfidf_masked, tfidf_obj.feature_names)
        cls.__term_score_tuples = cls.__tfidf_reduce_obj.extract_ngrams_from_docset('sum')

    def test_terms(self):
        term_score_tuples = self.__term_score_tuples
        actual_terms = [x for _, x in term_score_tuples]
        expected_terms = ['mount surfac',
                          'transmit path',
                          'electron element',
                          'link document',
                          'amid deriv',
                          'valproic acid',
                          'voic messag',
                          'pharmaceut formul',
                          'jack mechan',
                          'light beam',
                          'contact beam',
                          'angular veloc',
                          'shorter tuft',
                          'conduct materi',
                          'endodont instrument',
                          'mass offset',
                          'section bend',
                          'compon materi',
                          'connect portion',
                          'termin channel'
                          ]

        self.assertListEqual(actual_terms[:20], expected_terms)

    def test_scores(self):
        term_score_tuples = self.__term_score_tuples
        actual_scores = [x for x, _ in term_score_tuples]
        expected_scores = [0.9449111825230679,
                           0.8259734063804905,
                           0.7754588414852185,
                           0.7276068751089988,
                           0.7071067811865476,
                           0.7071067811865476,
                           0.7071067811865475,
                           0.6666666666666666,
                           0.6396021490668312,
                           0.6246950475544241,
                           0.6031800939323297,
                           0.6000595413031171,
                           0.5834599659915781,
                           0.5806718350868961,
                           0.5773502691896257,
                           0.5773502691896257,
                           0.5773502691896257,
                           0.5669467095138407,
                           0.5611088299627696,
                           0.5597177778726654]

        support.assert_list_almost_equal(self, actual_scores[:20], expected_scores)

    def test_timeseries_mat(self):
        timeseries_mat = self.__tfidf_reduce_obj.create_terms_count(self.__df, 'publication_date')
        self.assertEqual(sum(timeseries_mat[2]), 100)