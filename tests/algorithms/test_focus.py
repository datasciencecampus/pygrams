import unittest

from scripts.algorithms import term_focus
from scripts.algorithms.tfidf import TFIDF, LemmaTokenizer
from tests.utils import ReferenceData


class TestFocus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.num_ngrams = 5
        cls.cold_tfidf = TFIDF(ReferenceData.cold_df, tokenizer=LemmaTokenizer(), ngram_range=(2, 3))
        cls.random_tfidf = TFIDF(ReferenceData.random_df, tokenizer=LemmaTokenizer(), ngram_range=(2, 3))

    def test_popular_ngrams_by_no_focus(self):
        expected_output = set(
            ['ice stored', 'storage container', 'stored ready', 'constant temperature', 'ice stored ready'])

        _, actual_output, _ = term_focus.detect_and_focus_popular_ngrams('sum', False, None, None, 1, self.num_ngrams,
                                                                         self.cold_tfidf, self.random_tfidf)

        self.assertEqual(expected_output, actual_output)

    def test_popular_ngrams_by_set_focus(self):
        expected_output = set(
            ['ice stored', 'storage container', 'stored ready', 'constant temperature', 'ice stored ready'])

        _, actual_output, _ = term_focus.detect_and_focus_popular_ngrams('sum', False, 'set', None, 1, self.num_ngrams,
                                                                         self.cold_tfidf, self.random_tfidf)

        self.assertEqual(expected_output, actual_output)

    def test_popular_ngrams_by_chi2_focus(self):
        expected_output = set(
            ['ice stored', 'storage container', 'stored ready', 'constant temperature', 'ice stored ready'])

        _, actual_output, _ = term_focus.detect_and_focus_popular_ngrams('sum', False, 'chi2', None, 1, self.num_ngrams,
                                                                         self.cold_tfidf, self.random_tfidf)

        self.assertEqual(expected_output, actual_output)

    def test_popular_ngrams_by_mutual_focus(self):
        expected_output = set(
            ['ice stored', 'storage container', 'stored ready', 'constant temperature', 'ice stored ready'])

        _, actual_output, _ = term_focus.detect_and_focus_popular_ngrams('sum', False, 'mutual', None, 1,
                                                                         self.num_ngrams,
                                                                         self.cold_tfidf, self.random_tfidf)

        self.assertEqual(expected_output, actual_output)
