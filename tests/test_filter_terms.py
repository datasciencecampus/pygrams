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
        tfidf_obj = TFIDF(docs_df=df, ngram_range=(1, 3), max_document_frequency=0.1,
                          tokenizer=LemmaTokenizer(), text_header='abstract')
        self.feature_names = tfidf_obj.feature_names

    def test_embeddings_filter_binary(self):
        user_queries = 'pharmacy, health, chemist'
        weights_vec_expected = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0]
        weights_vec_actual = FilterTerms(self.feature_names, user_queries, threshold=0.8).ngram_weights_vec[410:430]

        self.assertListEqual(weights_vec_expected, weights_vec_actual)
        # assert what you like here. ie. first 20 or last 20 values, count of 1s or zeros, etc

    def test_embeddings_filter_cosine_dist(self):
        user_queries = 'pharmacy,  health, chemist'
        weights_vec_actual = FilterTerms(self.feature_names, user_queries).ngram_weights_vec[410:430]
        weights_vec_expected = [0.20709515901628847,
                                0.20709515901628847,
                                0.20709515901628847,
                                0.0,
                                0.551300224350135,
                                0.551300224350135,
                                0.551300224350135,
                                0.551300224350135,
                                0.551300224350135,
                                0.27164987710129196,
                                0.27164987710129196,
                                0.09792939071736236,
                                0.10161789725328345,
                                0.35670915168378176,
                                0.06740994123419917,
                                0.20015427221498908,
                                0.14549011491283229,
                                0.0,
                                0.006259676100908145,
                                0.1255539484346405]

        assert_list_almost_equal(self, weights_vec_expected, weights_vec_actual)

if __name__ == '__main__':
    unittest.main()
