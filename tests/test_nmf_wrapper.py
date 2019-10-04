import unittest

import pandas as pd
from numpy.testing import assert_almost_equal

from scripts import FilePaths
from scripts.nmf_wrapper import nmf_topic_modelling
from scripts.text_processing import LemmaTokenizer
from scripts.tfidf_wrapper import tfidf_from_text


class TestNMFWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        df = pd.read_pickle(FilePaths.us_patents_random_100_pickle_name)
        tfidf_obj = tfidf_from_text(df['abstract'], ngram_range=(1, 3), max_document_frequency=0.1,
                                    tokenizer=LemmaTokenizer())
        nmf_topics = 5
        cls.__nmf = nmf_topic_modelling(nmf_topics, tfidf_obj.tfidf_matrix)

    def test_nmf_topic1(self):
        actual_topic_1_score = self.__nmf.components_[0][1748]
        expected_topic_1_score = 0.006004153474348771

        assert_almost_equal(actual_topic_1_score, expected_topic_1_score, decimal=3)

    def test_nmf_topic2(self):
        actual_topic_2_score = self.__nmf.components_[1][3269]
        expected_topic_2_score = 0.06834714897242646

        assert_almost_equal(actual_topic_2_score, expected_topic_2_score, decimal=3)

    def test_nmf_topic3(self):
        actual_topic_3_score = self.__nmf.components_[2][6291]
        expected_topic_3_score = 0.04205993683629748

        assert_almost_equal(actual_topic_3_score, expected_topic_3_score, decimal=3)

    def test_nmf_topic4(self):
        actual_topic_4_score = self.__nmf.components_[3][849]
        expected_topic_4_score = 0.020628785653052678

        assert_almost_equal(actual_topic_4_score, expected_topic_4_score, decimal=3)

    def test_nmf_topic5(self):
        actual_topic_5_score = self.__nmf.components_[4][7006]
        expected_topic_5_score = 0.004466893271055121

        assert_almost_equal(actual_topic_5_score, expected_topic_5_score, decimal=3)