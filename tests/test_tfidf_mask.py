import unittest

import pandas as pd

from scripts import FilePaths
from scripts.documents_filter import DocumentsFilter
from scripts.filter_terms import FilterTerms
from scripts.text_processing import StemTokenizer
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import tfidf_from_text
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
from scripts.utils.date_utils import generate_year_week_dates


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
            'date_header': None
        }

        self.__tfidf_obj = tfidf_from_text(self.__df['abstract'], ngram_range=(min_n, self.__max_n),
                                           max_document_frequency=self.__max_df, tokenizer=StemTokenizer())
        cpc_dict = utils.cpc_dict(self.__df)

        self.__dates = generate_year_week_dates(self.__df, docs_mask_dict['date_header'])
        doc_filters = DocumentsFilter(self.__dates, docs_mask_dict, cpc_dict, self.__df.shape[0]).doc_filters

        # term weights - embeddings
        filter_output_obj = FilterTerms(self.__tfidf_obj.feature_names, None)
        term_weights = filter_output_obj.ngram_weights_vec

        tfidf_mask_obj = TfidfMask(self.__tfidf_obj, ngram_range=(min_n, self.__max_n), uni_factor=uni_factor, unbias=True)
        tfidf_mask_obj.update_mask(doc_filters, term_weights)
        self.__tfidf_mask = tfidf_mask_obj.tfidf_mask

    def test_num_non_zeros_no_clean_rows(self):
        self.init_mask('Y02', 2)
        self.assertEqual(6917, len(self.__tfidf_mask.data))

    def test_terms(self):
        self.init_mask('Y02', 2)
        expected_terms = ['acceler drive region',
                         'adjust flow exhaust',
                         'air suppli compress',
                         'amount suppli cylind',
                         'compress air cylind',
                         'compress extern air',
                         'compressor rotat synchron',
                         'control divid oper',
                         'control system variabl',
                         'control vane inject',
                         'cylind gener power',
                         'cylind torqu engin',
                         'deceler drive region',
                         'divid oper region',
                         'drive region fuel',
                         'engin control system',
                         'engin cylind gener',
                         'exhaust ga exhaust',
                         'exhaust ga suppli',
                         'extern air suppli',
                         'flow exhaust ga',
                         'fuel amount suppli',
                         'fuel inject cylind',
                         'ga exhaust engin',
                         'ga suppli turbin',
                         'gener power combust',
                         'inject time fuel',
                         'oper region vehicl',
                         'power combust fuel',
                         'region fuel amount',
                         'region vehicl steady-spe',
                         'rotat exhaust ga',
                         'rotat synchron turbin',
                         'steady-spe drive region',
                         'suppli compress air',
                         'suppli cylind torqu',
                         'synchron turbin compress',
                         'system variabl turbocharg',
                         'time fuel inject',
                         'turbin compress extern',
                         'turbin rotat exhaust',
                         'turbocharg engin cylind',
                         'turbocharg turbin rotat',
                         'vane adjust flow',
                         'vane inject time',
                         'variabl turbocharg engin',
                         'variabl turbocharg turbin',
                         'vehicl steady-spe drive',
                         'drive region',
                         'variabl turbocharg',
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
        self.assertEqual(57, len(tfidf_mask_nozero_rows.data))

    def test_num_non_zeros_clean_rows_clean_unigrams_and_df(self):
        self.init_mask('Y02', 1, uni_factor=0.4)
        tfidf_mask_nozero_rows, dates = utils.remove_all_null_rows_global(self.__tfidf_mask, self.__dates)
        self.assertEqual(57, len(tfidf_mask_nozero_rows.data))
        self.assertEqual(1, tfidf_mask_nozero_rows.shape[0])
        self.assertIsNone(self.__dates)

    def test_num_non_zeros_clean_rows(self):
        self.init_mask('Y02', 2)
        tfidf_mask_nozero_rows = utils.remove_all_null_rows(self.__tfidf_mask)
        self.assertEqual(51, len(tfidf_mask_nozero_rows.data))

    def test_num_non_zeros_clean_rows_and_df(self):
        self.init_mask('Y02', 2)
        tfidf_mask_nozero_rows, dates = utils.remove_all_null_rows_global(self.__tfidf_mask, self.__dates)
        self.assertEqual(51, len(tfidf_mask_nozero_rows.data))
        self.assertEqual(1, tfidf_mask_nozero_rows.shape[0])
        self.assertIsNone(self.__dates)

    def test_no_negative_weights(self):
        self.init_mask(None, 2)
        data = self.__tfidf_mask.data
        num_negatives = (data < 0).sum()
        self.assertEqual(num_negatives, 0)

    def test_non_zeros_clean_rows(self):
        self.init_mask('Y02', 2)
        tfidf_mask_nozero_rows = utils.remove_all_null_rows(self.__tfidf_mask)
        vocabulary = self.__tfidf_obj.vocabulary
        expected_term1_val = 0.0625
        expected_term2_val = 0.19753086419753094

        term1 = 'exhaust ga'  # 0.25
        term2 = 'drive region'  # 0.2962962962962961
        idx_term1 = vocabulary.get(term1)
        idx_term2 = vocabulary.get(term2)

        indexof_idx_term1 = tfidf_mask_nozero_rows.indices.tolist().index(idx_term1)
        indexof_idx_term2 = tfidf_mask_nozero_rows.indices.tolist().index(idx_term2)

        actual_values = list(tfidf_mask_nozero_rows.data)

        self.assertEqual(expected_term1_val, actual_values[indexof_idx_term1])
        self.assertAlmostEqual(expected_term2_val, actual_values[indexof_idx_term2])