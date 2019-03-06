import unittest

import pandas as pd
from nltk import word_tokenize

from scripts import FilePaths
from scripts.text_processing import  StemTokenizer, WordAnalyzer, \
    lowercase_strip_accents_and_ownership


# Sample abstracts taken from the USPTO Bulk Download Service: https://bulkdata.uspto.gov
# Data used was downloaded from "Patent Grant Full Text Data"
from scripts.tfidf_wrapper import TFIDF


class TestTFIDF(unittest.TestCase):

    def test_stematizer(self):
        words = ['freezing', 'frozen', 'freeze', 'reading']
        stematizer = StemTokenizer()
        expected_words = ['freez', 'frozen', 'freez', 'read']
        actual_words = [stematizer(word)[0] for word in words]
        self.assertListEqual(expected_words, actual_words)



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
