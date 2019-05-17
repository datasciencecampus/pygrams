import unittest
from numpy.testing import assert_almost_equal
import pandas as pd

from scripts import FilePaths
from scripts.nmf_wrapper import nmf_topic_modelling
from scripts.text_processing import LemmaTokenizer
from scripts.tfidf_wrapper import tfidf_from_text


class TestNMFWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        df = pd.read_pickle(FilePaths.us_patents_random_100_pickle_name)
        cls.__tfidf_obj = tfidf_from_text(df['abstract'], ngram_range=(1, 3), max_document_frequency=0.1,
                                    tokenizer=LemmaTokenizer())

    def test_nmf_topic1(self):
        nmf_topics = 5
        nmf = nmf_topic_modelling(nmf_topics, self.__tfidf_obj.tfidf_matrix, self.__tfidf_obj.feature_names)
        actual_topic_1_max_score = nmf.components_[0][3302]
        expected_topic_1_max_score = 0.2044937886411859

        assert_almost_equal(actual_topic_1_max_score, expected_topic_1_max_score, decimal=3)