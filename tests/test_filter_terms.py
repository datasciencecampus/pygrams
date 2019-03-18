import pandas as pd
import unittest
from scripts import FilePaths
from scripts.filter_terms import FilterTerms
from scripts.text_processing import LemmaTokenizer
from scripts.tfidf_wrapper import TFIDF
from tests.support import assert_list_almost_equal


class TestDocumentsFilter(unittest.TestCase):

    def setUp(self):
        df = pd.read_pickle(FilePaths.us_patents_random_100_pickle_name)
        tfidf_obj = TFIDF(df['abstract'], ngram_range=(1, 3), max_document_frequency=0.1,
                          tokenizer=LemmaTokenizer())
        self.feature_names = tfidf_obj.feature_names

    def test_embeddings_filter_binary(self):
        user_queries = ['pharmacy', 'health', 'chemist']
        weights_vec_expected = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0]
        weights_vec_actual = FilterTerms(self.feature_names, user_queries, threshold=0.8).ngram_weights_vec[410:430]

        self.assertListEqual(weights_vec_expected, weights_vec_actual)
        # assert what you like here. ie. first 20 or last 20 values, count of 1s or zeros, etc

    def test_embeddings_filter_cosine_dist(self):
        user_queries = ['pharmacy', 'health', 'chemist']
        weights_vec_actual = FilterTerms(self.feature_names, user_queries).ngram_weights_vec[410:430]
        weights_vec_expected = [0.5728331683597565,
                                0.5728331683597565,
                                0.5728331683597565,
                                0.023525821108745026,
                                0.551300224350135,
                                0.551300224350135,
                                0.551300224350135,
                                0.551300224350135,
                                0.551300224350135,
                                0.32695037912444064,
                                0.3324986904828807,
                                0.41004053511285365,
                                0.6309494801963349,
                                0.6397470276515971,
                                0.2626750291615634,
                                0.2626750291615634,
                                0.47060086220739433,
                                -0.10829696922978878,
                                0.19429777744446344,
                                0.19429777744446344]

        assert_list_almost_equal(self, weights_vec_expected, weights_vec_actual)


if __name__ == '__main__':
    unittest.main()
