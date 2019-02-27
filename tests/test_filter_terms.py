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
        user_queries=['pharmacy, health, chemist']
        termf_filter = FilterTerms(self.feature_names, user_queries)
        weights_vec = termf_filter.get_embeddings_vec(True)

        # assert what you like here. ie. first 20 or last 20 values, count of 1s or zeros, etc

    def test_embeddings_filter_cosine_dist(self):
        user_queries = ['pharmacy,  health, chemist']
        termf_filter = FilterTerms(self.feature_names, user_queries)
        weights_vec = termf_filter.get_embeddings_vec(False)

        # assert what you like here. ie. first 20 or last 20 values, median value or avg, etc
if __name__ == '__main__':
    unittest.main()