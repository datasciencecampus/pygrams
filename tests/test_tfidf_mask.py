import pandas as pd
import unittest
import numpy as np

from scripts.documents_filter import DocumentsFilter
from scripts.documents_weights import DocumentsWeights
from scripts.text_processing import LemmaTokenizer, StemTokenizer
from scripts.tfidf_wrapper import TFIDF
from scripts import FilePaths
from scripts.filter_terms import FilterTerms
from scripts.terms_graph import TermsGraph
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.utils import utils

# text for record 95, the only Y02 in set
# An engine control system with a variable turbocharger may include an engine
# including a cylinder generating power by combustion of a fuel, a variable
# turbocharger including a turbine rotated by exhaust gas exhausted by the engine,
# and a compressor rotated in synchronization with the turbine and compressing
# external air and supplying the compressed air to the cylinder, a vane adjusting
# flow area of exhaust gas supplied to the turbine, and a controller dividing an
# operation region of a vehicle into a steady-speed driving region, an acceleration
# driving region, and a deceleration driving region from a fuel amount supplied to
# the cylinder and a required torque of the engine, and controlling opening of the
# vane and an injection timing of fuel injected into the cylinder.


class TestTfidfMask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        min_n = 2
        cls.__max_n = 3
        max_df = 0.3

        cls.__df = pd.read_pickle(FilePaths.us_patents_random_100_pickle_name)
        cls.__tfidf_obj = TFIDF(docs_df=cls.__df, ngram_range=(min_n, cls.__max_n), max_document_frequency=max_df,
                                tokenizer=StemTokenizer(), text_header='abstract')

    def init_mask(self, cpc):
        docs_mask_dict = {}
        docs_mask_dict['filter_by'] = 'union'
        docs_mask_dict['cpc'] = cpc
        docs_mask_dict['time'] = None
        docs_mask_dict['cite'] = []
        docs_mask_dict['columns'] = None
        docs_mask_dict['dates'] = [None]

        doc_filters = DocumentsFilter(self.__df, docs_mask_dict).doc_weights
        doc_weights = DocumentsWeights(self.__df, docs_mask_dict['time'], docs_mask_dict['cite'],
                                       docs_mask_dict['dates'][-1:]).weights
        doc_weights = [a * b for a, b in zip(doc_filters, doc_weights)]

        # term weights - embeddings
        filter_output_obj = FilterTerms(self.__tfidf_obj.feature_names, None, None)
        term_weights = filter_output_obj.ngram_weights_vec

        tfidf_mask_obj = TfidfMask(self.__tfidf_obj, doc_weights, max_ngram_length=self.__max_n)
        tfidf_mask_obj.update_mask(doc_weights, term_weights)
        self.__tfidf_mask = tfidf_mask_obj.tfidf_mask
    def test_num_non_zeros_no_clean_rows(self):
        self.init_mask('Y02')
        self.assertEqual(2059, len(self.__tfidf_mask.data))

    def test_terms(self):
        self.init_mask('Y02')
        expected_terms = ['exhaust ga',
                          'variabl turbocharg',
                          'turbin rotat',
                          'compressor rotat',
                          'compress air',
                          'control divid',
                          'oper region',
                          'drive region',
                          'inject time',
                          'fuel inject',
                          'engin control system',
                          'cylind gener power',
                          'exhaust ga exhaust',
                          'compress extern air',
                          'vane adjust flow',
                          'exhaust ga suppli',
                          'steady-spe drive region',
                          'acceler drive region',
                          'deceler drive region',
                          'fuel amount suppli']
        tfidf_matrix = self.__tfidf_obj.tfidf_matrix
        tfidf_masked = self.__tfidf_mask.multiply(tfidf_matrix)
        tfidf_masked = utils.remove_all_null_rows(tfidf_masked)

        tfidf_reduce_obj = TfidfReduce(tfidf_masked, self.__tfidf_obj.feature_names)
        term_score_tuples = tfidf_reduce_obj.extract_ngrams_from_row(0)
        actual_terms = [x for _, x in term_score_tuples]
        for tup0, tup1 in term_score_tuples:
            print (str(tup0) + ": " + str(tup1))
        self.assertEqual(expected_terms, actual_terms)

    def test_num_non_zeros_clean_rows(self):
        self.init_mask('Y02')
        tfidf_mask_nozero_rows = utils.remove_all_null_rows(self.__tfidf_mask)
        self.assertEqual(20, len(tfidf_mask_nozero_rows.data))

    def test_no_negative_weights(self):
        self.init_mask(None)
        data = self.__tfidf_mask.data
        num_negatives = (data < 0).sum()
        self.assertEqual(num_negatives, 0)

    def test_non_zeros_clean_rows(self):
        self.init_mask('Y02')
        tfidf_mask_nozero_rows = utils.remove_all_null_rows(self.__tfidf_mask)

        expected_values = [1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           0.2962962962962961,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0,
                           0.25]

        actual_values = list(tfidf_mask_nozero_rows.data)

        self.assertListEqual(expected_values, actual_values)
