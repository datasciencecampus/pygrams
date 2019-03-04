import pandas as pd
import unittest
from scripts import FilePaths
from scripts.filter_terms import FilterTerms
from scripts.text_processing import LemmaTokenizer
from scripts.tfidf_wrapper import TFIDF

# try to inspect some values. ie. does the code do the right thing?
# Does it keep relevant stuff or not?
# What is the best binary threshold?
# feel free to write some more tests


class TestDocumentsFilter(unittest.TestCase):

    def setUp(self):

        df = pd.read_pickle(FilePaths.us_patents_random_100_pickle_name)
        tfidf_obj = TFIDF(docs_df=df, ngram_range=(1, 3), max_document_frequency=0.1,
                          tokenizer=LemmaTokenizer(), text_header='abstract')
        self.feature_names = tfidf_obj.feature_names
        #change this to the model name!
        self.__model_name='blah'

    def test_embeddings_filter_binary(self):
        user_queries='pharmacy, health, chemist'
        termf_filter = FilterTerms(self.feature_names, user_queries)
        weights_vec = termf_filter.get_embeddings_vec(threshold=0.35)

        self.assertTrue(weights_vec[0:19] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "assertion fails")
        # assert what you like here. ie. first 20 or last 20 values, count of 1s or zeros, etc

    def test_embeddings_filter_cosine_dist(self):
        user_queries = 'pharmacy,  health, chemist'
        termf_filter = FilterTerms(self.feature_names, user_queries)
        weights_vec = termf_filter.get_embeddings_vec(threshold=None)

        self.assertTrue(weights_vec[0:19] == [0.21542439491593243, 0.17356495208915834, 0.18345394311393398, 0.15885341418551066, 0.22712907652174147, 0.3356339126593535, 0.18222455099936702, 0.18222455099936702, 0.2927502617700309, 0.18222455099936702, 0.2818905519100835, 0.2818905519100835, 0.28348328548493773, 0.13729809635518406, 0.21276843688627714, 0.1987974549511532, 0.12008022214367764, 0.2733594117022526, 0.2927502617700309], "assertion fails")

        # assert what you like here. ie. first 20 or last 20 values, median value or avg, etc
if __name__ == '__main__':
    unittest.main()