import unittest

import pandas as pd

from scripts import FilePaths
from scripts.algorithms.tfidf import TFIDF
from scripts.utils.pickle2df import PatentsPickle2DataFrame


class TestCitation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_frame = PatentsPickle2DataFrame(FilePaths.us_patents_random_1000_pickle_name).data_frame
        cls.tfidf_inst = TFIDF(data_frame)

        cls.citation_count_dict = pd.read_pickle(FilePaths.us_patents_citation_dictionary_1of2_pickle_name)
        citation_count_dict_pt2 = pd.read_pickle(FilePaths.us_patents_citation_dictionary_1of2_pickle_name)
        cls.citation_count_dict.update(citation_count_dict_pt2)

    def test_citation_before(self):
        actual_top_five_bef_citation = self.tfidf_inst.detect_popular_ngrams_in_corpus()[1][0:5]
        expected_top_five_bef_citation = [(1.439758952315001, 'first'), (1.0793756164007968, 'system'), (1.056505391786121, 'devic'), (1.0376976573916799, 'imag'), (1.0187146109156324, 'data')]

        print(f"The top five TFIDF before without citation weighting should be {expected_top_five_bef_citation}")
        print(f"The top five TFIDF before without citation weighting  {actual_top_five_bef_citation}")
        self.assertSameTFIDF(expected_top_five_bef_citation, actual_top_five_bef_citation)

    def test_citation_after(self):
        actual_top_five_aft_citation = self.tfidf_inst.detect_popular_ngrams_in_corpus(
            citation_count_dict=self.citation_count_dict)[1][0:5]
        expected_top_five_aft_citation = [(1.439758952315001, 'first'), (1.0793756164007968, 'system'), (1.056505391786121, 'devic'), (1.0376976573916799, 'imag'), (1.0187146109156324, 'data')]
        print(f"The top five TFIDF after citation weighting should be {expected_top_five_aft_citation}")
        print(f"The top five TFIDF after citation weighting were {actual_top_five_aft_citation}")
        self.assertSameTFIDF(expected_top_five_aft_citation, actual_top_five_aft_citation)

    def test_citations_all(self):
        assert self.tfidf_inst.detect_popular_ngrams_in_corpus() \
               != self.tfidf_inst.detect_popular_ngrams_in_corpus(citation_count_dict=self.citation_count_dict)

    def assertSameTFIDF(self, expected_tfidf, actual_tfidf):
        for expected, actual in zip(expected_tfidf, actual_tfidf):
            self.assertEqual(expected[1], actual[1])
            self.assertAlmostEqual(expected[0], actual[0], places=12)
