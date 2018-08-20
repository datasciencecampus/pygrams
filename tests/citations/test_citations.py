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
        expected_top_five_bef_citation = [(21.164206847352048, 'first'), (15.978452815105912, 'devic'),
                                          (15.291909509744947, 'second'), (14.632029117170827, 'data'),
                                          (14.60830421617836, 'layer')]

        print(f"The top five TFIDF before without citation weighting should be {expected_top_five_bef_citation}")
        print(f"The top five TFIDF before without citation weighting  {actual_top_five_bef_citation}")
        assert expected_top_five_bef_citation == actual_top_five_bef_citation

    def test_citation_after(self):
        actual_top_five_aft_citation = self.tfidf_inst.detect_popular_ngrams_in_corpus(
            citation_count_dict=self.citation_count_dict)[1][0:5]
        expected_top_five_aft_citation = [(1.4356925568041987, 'first'), (1.0939948002413247, 'devic'),
                                          (1.088760298174519, 'signal'), (1.0877552160409762, 'system'),
                                          (1.038396555922477, 'imag')]
        print(f"The top five TFIDF after citation weighting should be {expected_top_five_aft_citation}")
        print(f"The top five TFIDF after citation weighting were {actual_top_five_aft_citation}")
        assert expected_top_five_aft_citation == actual_top_five_aft_citation

    def test_citations_all(self):
        assert self.tfidf_inst.detect_popular_ngrams_in_corpus() \
               != self.tfidf_inst.detect_popular_ngrams_in_corpus(citation_count_dict=self.citation_count_dict)
