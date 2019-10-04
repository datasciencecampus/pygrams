import unittest

import pandas as pd

from scripts import FilePaths
from scripts.filter_terms import FilterTerms
from scripts.text_processing import LemmaTokenizer
from scripts.tfidf_wrapper import tfidf_from_text
from tests.support import assert_list_almost_equal


class TestTermsFilter(unittest.TestCase):

    def setUp(self):
        df = pd.read_pickle(FilePaths.us_patents_random_100_pickle_name)
        tfidf_obj = tfidf_from_text(df['abstract'], ngram_range=(1, 3), max_document_frequency=0.1,
                                    tokenizer=LemmaTokenizer())
        self.feature_names = tfidf_obj.feature_names

    def test_embeddings_filter_binary(self):
        user_queries = ['pharmacy', 'health', 'chemist']
        weights_vec_expected = [0.0,
                                0.0,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0]
        weights_vec_actual = FilterTerms(self.feature_names, user_queries, threshold=0.8,
                                         file_name='../models/glove/glove2vec.6B.50d.txt').ngram_weights_vec[410:430]

        self.assertListEqual(weights_vec_expected, weights_vec_actual)
        # assert what you like here. ie. first 20 or last 20 values, count of 1s or zeros, etc

    def test_embeddings_filter_cosine_dist(self):
        user_queries = ['pharmacy', 'health', 'chemist']
        weights_vec_actual = FilterTerms(self.feature_names, user_queries,
                                         file_name='../models/glove/glove2vec.6B.50d.txt').ngram_weights_vec[410:430]
        weights_vec_expected = [0.5419303846700578,
                                0.48111016385510563,
                                0.746749058966686,
                                0.02883977284936675,
                                0.02883977284936675,
                                0.4473920633233143,
                                0.02883977284936675,
                                0.07745187589425416,
                                0.2333812492827283,
                                0.3102688829728565,
                                0.37904818340304136,
                                0.37904818340304136,
                                0.02883977284936675,
                                0.02883977284936675,
                                -0.10002208308653802,
                                -0.10002208308653802,
                                -0.100022083086538028,
                                0.02136222212894982,
                                0.22949645707300867,
                                0.09082483113254906]

        assert_list_almost_equal(self, weights_vec_expected, weights_vec_actual)


if __name__ == '__main__':
    unittest.main()
