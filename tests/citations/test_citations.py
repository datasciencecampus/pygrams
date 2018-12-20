import unittest

import pandas as pd

from scripts import FilePaths
from scripts.algorithms.tfidf import TFIDF
from scripts.utils.pickle2df import PatentsPickle2DataFrame


class TestCitation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_frame = PatentsPickle2DataFrame(FilePaths.us_patents_random_1000_pickle_name).data_frame
        cls.tfidf_inst = TFIDF(data_frame, ngram_range=(2, 3))

        cls.citation_count_dict = pd.read_pickle(FilePaths.us_patents_citation_dictionary_1of2_pickle_name)
        citation_count_dict_pt2 = pd.read_pickle(FilePaths.us_patents_citation_dictionary_2of2_pickle_name)
        cls.citation_count_dict.update(citation_count_dict_pt2)

    def test_citation_before(self):
        actual_popular_terms, _ = self.tfidf_inst.detect_popular_ngrams_in_docs_set()
        actual_top_five_bef_citation = actual_popular_terms[0:5]
        expected_top_five_bef_citation = ['semiconductor substrat', 'corn plant', 'semiconductor devic', 'liquid crystal display', 'pharmaceut composit']

        print(f"The top five TFIDF before without citation weighting should be {expected_top_five_bef_citation}")
        print(f"The top five TFIDF before without citation weighting  {actual_top_five_bef_citation}")
        self.assertListEqual(expected_top_five_bef_citation, actual_top_five_bef_citation)

    def test_citation_after(self):
        actual_popular_terms, _ = self.tfidf_inst.detect_popular_ngrams_in_docs_set(citation_count_dict=self.citation_count_dict)
        actual_top_five_aft_citation = actual_popular_terms[0:5]
        expected_top_five_aft_citation = ['network interfac', 'actuat member', 'form part', 'surgic clip applier', 'transpar transistor']
        print(f"The top five TFIDF after citation weighting should be {expected_top_five_aft_citation}")
        print(f"The top five TFIDF after citation weighting were {actual_top_five_aft_citation}")
        self.assertListEqual(expected_top_five_aft_citation, actual_top_five_aft_citation)

    def test_citations_all(self):
        assert self.tfidf_inst.detect_popular_ngrams_in_docs_set() \
               != self.tfidf_inst.detect_popular_ngrams_in_docs_set(citation_count_dict=self.citation_count_dict)


