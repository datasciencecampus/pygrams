import unittest

from scripts.algorithms.term_focus import TermFocus
from scripts.text_processing import  LemmaTokenizer
from scripts.tfidf_wrapper import TFIDF
from tests.utils import ReferenceData


@unittest.skip("Temporarily shut down module")
class TestFocus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.num_ngrams = 5
        cold_tfidf = TFIDF(ReferenceData.cold_df, tokenizer=LemmaTokenizer(), ngram_range=(2, 3))
        random_tfidf = TFIDF(ReferenceData.random_df, tokenizer=LemmaTokenizer(), ngram_range=(2, 3))
        cls.tfocus = TermFocus(cold_tfidf, random_tfidf)

    def test_popular_ngrams_by_no_focus(self):
        expected_output = {'cold air flow', 'constant temperature', 'freezing chamber', 'ice store ready',
                           'utility chamber'}
        _, actual_output, _ = self.tfocus.detect_and_focus_popular_ngrams('sum', False, None, None, 1, self.num_ngrams)

        self.assertEqual(expected_output, actual_output)

    def test_popular_ngrams_by_set_focus(self):
        expected_output = {'cold air flow', 'constant temperature', 'freezing chamber', 'ice store ready',
                           'utility chamber'}
        _, actual_output, _ = self.tfocus.detect_and_focus_popular_ngrams('sum', False, 'set', None, 1, self.num_ngrams)
        self.assertEqual(expected_output, actual_output)

    def test_popular_ngrams_by_chi2_focus(self):
        expected_output = {'constant temperature', 'ice store', 'ice store ready', 'refrigerating chamber',
                           'store ready'}
        _, actual_output, _ = self.tfocus.detect_and_focus_popular_ngrams('sum', False, 'chi2', None, 1,
                                                                          self.num_ngrams)
        self.assertEqual(expected_output, actual_output)

    def test_popular_ngrams_by_mutual_focus(self):
        expected_output = {'refrigerating chamber', 'upper section', 'upper space', 'utility chamber', 'warm section'}
        _, actual_output, _ = self.tfocus.detect_and_focus_popular_ngrams('sum', False, 'mutual', None, 1,
                                                                          self.num_ngrams)
        self.assertEqual(expected_output, actual_output)
