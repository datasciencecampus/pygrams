import unittest

import pandas as pd

from scripts import FilePaths
from scripts.documents_filter import DocumentsFilter
from scripts.documents_weights import DocumentsWeights
from scripts.filter_terms import FilterTerms
from scripts.text_processing import StemTokenizer
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import TFIDF
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
        cls.__max_n = 3
        cls.__max_df = 0.3

        cls.__df = pd.read_pickle(FilePaths.us_patents_random_100_pickle_name)

    def init_mask(self, cpc, min_n, uni_factor=0.8):
        docs_mask_dict = {
            'filter_by': 'union',
            'cpc': cpc,
            'time': None,
            'cite': [],
            'columns': None,
            'date': None,
            'date_header': ''
        }

        self.__tfidf_obj = TFIDF(self.__df['abstract'], ngram_range=(min_n, self.__max_n),
                                 max_document_frequency=self.__max_df, tokenizer=StemTokenizer())

        doc_filters = DocumentsFilter(self.__df, docs_mask_dict).doc_weights
        doc_weights = DocumentsWeights(self.__df, docs_mask_dict['time'], docs_mask_dict['cite'],
                                       docs_mask_dict['date_header']).weights
        doc_weights = [a * b for a, b in zip(doc_filters, doc_weights)]

        # term weights - embeddings
        filter_output_obj = FilterTerms(self.__tfidf_obj.feature_names, None)
        term_weights = filter_output_obj.ngram_weights_vec

        tfidf_mask_obj = TfidfMask(self.__tfidf_obj, ngram_range=(min_n, self.__max_n), uni_factor=uni_factor)
        tfidf_mask_obj.update_mask(doc_weights, term_weights)
        self.__tfidf_mask = tfidf_mask_obj.tfidf_mask

    def test_num_non_zeros_no_clean_rows(self):
        self.init_mask('Y02', 2)
        self.assertEqual(2059, len(self.__tfidf_mask.data))

    def test_terms(self):
        self.init_mask('Y02', 2)
        expected_terms = ['variabl turbocharg',
                          'acceler drive region',
                          'compress air',
                          'compress extern air',
                          'compressor rotat',
                          'control divid',
                          'cylind gener power',
                          'deceler drive region',
                          'engin control system',
                          'exhaust ga exhaust',
                          'exhaust ga suppli',
                          'fuel amount suppli',
                          'fuel inject',
                          'inject time',
                          'oper region',
                          'steady-spe drive region',
                          'turbin rotat',
                          'vane adjust flow',
                          'drive region',
                          'exhaust ga']

        tfidf_matrix = self.__tfidf_obj.tfidf_matrix
        tfidf_masked = self.__tfidf_mask.multiply(tfidf_matrix)
        tfidf_masked = utils.remove_all_null_rows(tfidf_masked)

        tfidf_reduce_obj = TfidfReduce(tfidf_masked, self.__tfidf_obj.feature_names)
        term_score_tuples = tfidf_reduce_obj.extract_ngrams_from_row(0)
        term_score_tuples.sort(key=lambda tup: (-tup[0], tup[1]))
        actual_terms = [x for _, x in term_score_tuples]

        self.assertEqual(expected_terms, actual_terms)

    def test_num_non_zeros_clean_rows_clean_unigrams(self):
        self.init_mask('Y02', 1, uni_factor=0.4)
        tfidf_mask_nozero_rows = utils.remove_all_null_rows(self.__tfidf_mask)
        self.assertEqual(26, len(tfidf_mask_nozero_rows.data))

    def test_num_non_zeros_clean_rows(self):
        self.init_mask('Y02', 2)
        tfidf_mask_nozero_rows = utils.remove_all_null_rows(self.__tfidf_mask)
        self.assertEqual(20, len(tfidf_mask_nozero_rows.data))

    def test_no_negative_weights(self):
        self.init_mask(None, 2)
        data = self.__tfidf_mask.data
        num_negatives = (data < 0).sum()
        self.assertEqual(num_negatives, 0)

    def test_non_zeros_clean_rows(self):
        self.init_mask('Y02', 2)
        tfidf_mask_nozero_rows = utils.remove_all_null_rows(self.__tfidf_mask)
        vectorizer = self.__tfidf_obj.vectorizer
        expected_term1_val = 0.25
        expected_term2_val = 0.2962962962962961

        term1 = 'exhaust ga'  # 0.25
        term2 = 'drive region'  # 0.2962962962962961
        idx_term1 = vectorizer.vocabulary_.get(term1)
        idx_term2 = vectorizer.vocabulary_.get(term2)

        indexof_idx_term1 = tfidf_mask_nozero_rows.indices.tolist().index(idx_term1)
        indexof_idx_term2 = tfidf_mask_nozero_rows.indices.tolist().index(idx_term2)

        actual_values = list(tfidf_mask_nozero_rows.data)

        self.assertEqual(expected_term1_val, actual_values[indexof_idx_term1])
        self.assertAlmostEqual(expected_term2_val, actual_values[indexof_idx_term2])
