import unittest

import pandas as pd
from nltk import word_tokenize

from scripts import FilePaths
from scripts.algorithms.tfidf import TFIDF, StemTokenizer, WordAnalyzer, \
    lowercase_strip_accents_and_ownership


# Sample abstracts taken from the USPTO Bulk Download Service: https://bulkdata.uspto.gov
# Data used was downloaded from "Patent Grant Full Text Data"

class TestTFIDF(unittest.TestCase):

    def test_stematizer(self):
        words = ['freezing', 'frozen', 'freeze', 'reading']
        stematizer = StemTokenizer()
        expected_words = ['freez', 'frozen', 'freez', 'read']
        actual_words = [stematizer(word)[0] for word in words]
        self.assertListEqual(expected_words, actual_words)

    def test_popular_terms_from_1000_samples_via_stems(self):
        abstract_from_US09315032_20160419 = (
            "A print apparatus includes a carriage configured to move reciprocally along one "
            "direction; a liquid supply tube, one end thereof being connected to the carriage; a housing in an "
            "interior of which the carriage and at least a part of the liquid supply tube are arranged; and a control "
            "unit configured to control reciprocal movement of the carriage. The liquid supply tube is connected to "
            "the carriage from a counter-home position side that is the opposite side in the one direction to a home "
            "position side at which the carriage is located before the start of printing. The control unit carries out "
            "a movement control, before the start of printing, to cause the carriage to move at a first speed toward "
            "the counter-home position side from the home position side, and then to move at a second speed faster "
            "than the first speed toward the home position side.")
        df = pd.read_pickle(FilePaths.us_patents_random_1000_pickle_name)
        tfidf_engine = TFIDF(df, ngram_range=(2, 4), tokenizer=StemTokenizer())
        actual_popular_terms, tfidf_weighted_tuples = tfidf_engine.extract_popular_ngrams(
            abstract_from_US09315032_20160419)
        expected_popular_terms = ['first speed', 'control unit']
        self.assertListEqual(expected_popular_terms, actual_popular_terms)


class Test_lowercase_strip_accents_and_ownership(unittest.TestCase):

    def test_lowercase(self):
        doc = 'Test ABCdefGH IJ. Again'
        expected = 'test abcdefgh ij. again'
        actual = lowercase_strip_accents_and_ownership(doc)
        self.assertEqual(expected, actual)

    def test_accented(self):
        doc = 'Test type âêîôûŵŷ, äëïöüẅÿ àèìòùẁỳ OR áéíóúẃý, hold. Again'
        expected = 'test type aeiouwy, aeiouwy aeiouwy or aeiouwy, hold. again'
        actual = lowercase_strip_accents_and_ownership(doc)
        self.assertEqual(expected, actual)

    def test_ownership(self):
        doc = "Ian's simple test"
        expected = 'ian simple test'
        actual = lowercase_strip_accents_and_ownership(doc)
        self.assertEqual(expected, actual)


class TestWordAnalyzer(unittest.TestCase):

    def setUp(self):
        self.word_tokenizer = word_tokenize
        self.preprocess = lowercase_strip_accents_and_ownership

    def test_preprocess(self):
        actual = self.preprocess('Hello there')
        self.assertEqual('hello there', actual)

    def test_WordAnalyser_unigrams_no_stopwords(self):
        ngram_range = (1, 1)
        WordAnalyzer.init(tokenizer=self.word_tokenizer, preprocess=self.preprocess, ngram_range=ngram_range)

        doc = 'test words'
        expected_ngrams = ['test', 'words']
        actual_ngrams = WordAnalyzer.analyzer(doc)
        self.assertListEqual(expected_ngrams, actual_ngrams)

    def test_WordAnalyser_unigrams_with_stopwords(self):
        ngram_range = (1, 1)
        WordAnalyzer.init(tokenizer=self.word_tokenizer, preprocess=self.preprocess, ngram_range=ngram_range)

        doc = 'Some test words to ignore safely'
        expected_ngrams = ['test', 'words', 'ignore', 'safely']
        actual_ngrams = WordAnalyzer.analyzer(doc)
        self.assertListEqual(expected_ngrams, actual_ngrams)

    def test_WordAnalyser_unigrams_with_punctuation(self):
        ngram_range = (1, 1)
        WordAnalyzer.init(tokenizer=self.word_tokenizer, preprocess=self.preprocess, ngram_range=ngram_range)

        doc = "Some test words, to ignore except-hyphens but including someone's ownership"
        expected_ngrams = ['test', 'words', 'ignore', 'except-hyphens', 'ownership']
        actual_ngrams = WordAnalyzer.analyzer(doc)
        self.assertListEqual(expected_ngrams, actual_ngrams)

    def test_WordAnalyser_ngrams_dont_cross_punctuation_or_stop_words(self):
        ngram_range = (1, 3)
        WordAnalyzer.init(tokenizer=self.word_tokenizer, preprocess=self.preprocess, ngram_range=ngram_range)

        doc = "Some test words, except-hyphens metal but someone's metal fish bucket"
        expected_ngrams = ['test', 'words', 'except-hyphens', 'metal', 'metal', 'fish', 'bucket',
                           'test words', 'except-hyphens metal', 'metal fish', 'fish bucket',
                           'metal fish bucket']
        actual_ngrams = WordAnalyzer.analyzer(doc)
        self.assertListEqual(expected_ngrams, actual_ngrams)
