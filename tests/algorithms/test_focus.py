import unittest

from scripts.algorithms.term_focus import TermFocus
from scripts.algorithms.tfidf import TFIDF, LemmaTokenizer
from tests.utils import ReferenceData

class FakeArgs(object):
    __slots__ = ['pick', 'time', 'focus']

class TestFocus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.num_ngrams = 5
        cold_tfidf = TFIDF(ReferenceData.cold_df, tokenizer=LemmaTokenizer(), ngram_range=(2, 3))
        random_tfidf = TFIDF(ReferenceData.random_df, tokenizer=LemmaTokenizer(), ngram_range=(2, 3))
        cls.tfocus = TermFocus(cold_tfidf, random_tfidf)
        cls.args = FakeArgs()

    def test_popular_ngrams_by_no_focus(self):
        self.args.pick='sum'
        self.args.time = False
        self.args.focus = None

        expected_output = {'cold air flow', 'constant temperature', 'freezing chamber', 'ice store ready', 'utility chamber'}
        _, actual_output, _ = self.tfocus.detect_and_focus_popular_ngrams(self.args, None, 1, self.num_ngrams)

        self.assertEqual(expected_output, actual_output)

    def test_popular_ngrams_by_set_focus(self):
        self.args.pick = 'sum'
        self.args.time = False
        self.args.focus = 'set'

        expected_output = {'cold air flow', 'constant temperature', 'freezing chamber', 'ice store ready', 'utility chamber'}
        _, actual_output, _ = self.tfocus.detect_and_focus_popular_ngrams(self.args, None, 1, self.num_ngrams)
        self.assertEqual(expected_output, actual_output)

    def test_popular_ngrams_by_chi2_focus(self):
        self.args.pick = 'sum'
        self.args.time = False
        self.args.focus = 'chi2'

        expected_output = {'constant temperature', 'ice store', 'ice store ready', 'refrigerating chamber', 'store ready'}
        _, actual_output, _ = self.tfocus.detect_and_focus_popular_ngrams(self.args, None, 1, self.num_ngrams)
        self.assertEqual(expected_output, actual_output)

    def test_popular_ngrams_by_mutual_focus(self):
        self.args.pick = 'sum'
        self.args.time = False
        self.args.focus = 'mutual'

        expected_output = {'refrigerating chamber', 'upper section', 'upper space', 'utility chamber', 'warm section'}
        _, actual_output, _ = self.tfocus.detect_and_focus_popular_ngrams(self.args, None, 1, self.num_ngrams)
        self.assertEqual(expected_output, actual_output)
