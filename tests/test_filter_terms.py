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
        weights_vec_expected = [1.0,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                1.0,
                                0.0,
                                0.0,
                                1.0,
                                0.0]
        weights_vec_actual = FilterTerms(self.feature_names, user_queries, threshold=0.75,
                                         file_name='../models/glove/glove2vec.6B.50d.txt').ngram_weights_vec[410:430]

        self.assertListEqual(weights_vec_expected, weights_vec_actual)
        # assert what you like here. ie. first 20 or last 20 values, count of 1s or zeros, etc

    def test_embeddings_filter_cosine_dist(self):
        user_queries = ['pharmacy', 'health', 'chemist']
        weights_vec_actual = FilterTerms(self.feature_names, user_queries,
                                         file_name='../models/glove/glove2vec.6B.50d.txt').ngram_weights_vec[410:430]
        weights_vec_expected = [0.4989265853810251,
                                0.4989265853810251,
                                0.24968991916009714,
                                0.4692156257668608,
                                0.4692156257668608,
                                0.5583217759394512,
                                0.5583217759394512,
                                0.5583217759394512,
                                0.5583217759394512,
                                0.5583217759394512,
                                0.5583217759394512,
                                0.49430560632716464,
                                0.49430560632716464,
                                0.49430560632716464,
                                0.5418258868323199,
                                0.5418258868323199,
                                0.49430560632716464,
                                0.49430560632716464,
                                0.5418258868323199,
                                0.49430560632716464]

        assert_list_almost_equal(self, weights_vec_expected, weights_vec_actual)


if __name__ == '__main__':
    unittest.main()
