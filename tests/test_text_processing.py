import unittest

from nltk import word_tokenize

from scripts.text_processing import StemTokenizer, WordAnalyzer, \
    lowercase_strip_accents_and_ownership, LemmaTokenizer

# Sample abstracts taken from the USPTO Bulk Download Service: https://bulkdata.uspto.gov
# Data used was downloaded from "Patent Grant Full Text Data"


class TestStematizer(unittest.TestCase):

    def test_stematizer(self):
        words = ['freezing', 'frozen', 'freeze', 'reading']
        stematizer = StemTokenizer()
        expected_words = ['freez', 'frozen', 'freez', 'read']
        actual_words = [stematizer(word)[0] for word in words]
        self.assertListEqual(expected_words, actual_words)


class TestLowercaseStripAccentsAndOwnership(unittest.TestCase):

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
        expected_ngrams = [
            'test',
            'words',
            'except-hyphens',
            'metal',
            'metal',
            'fish',
            'bucket',
            'test words',
            'except-hyphens metal',
            'metal metal',
            'metal fish',
            'fish bucket',
            'except-hyphens metal metal',
            'metal metal fish',
            'metal fish bucket'

        ]
        actual_ngrams = WordAnalyzer.analyzer(doc)
        self.assertListEqual(expected_ngrams, actual_ngrams)

    def test_WordAnalyser_ngrams(self):
        ngram_range = (1, 3)
        WordAnalyzer.init(tokenizer=LemmaTokenizer(), preprocess=self.preprocess, ngram_range=ngram_range)

        doc = "Conductive structures in features of an insulator layer on a substrate are fabricated by a particular " \
              "process. In this process, a layer of conductive material is applied over the insulator layer so that " \
              "the layer of conductive material covers field regions adjacent the features and fills in the features " \
              "themselves. A grain size differential between the conductive material which covers the field regions " \
              "and the conductive material which fills in the feature is then established by annealing the layer of " \
              "conductive material. Excess conductive material is then removed to uncover the field regions and leave " \
              "the conductive structures. The layer of conductive material is applied so as to define a first layer " \
              "thickness over the field regions and a second layer thickness in and over the features. These " \
              "thicknesses are dimensioned such that d 1 ≦0.5d 2 , with d 1  being the first layer thickness and d 2  " \
              "being the second layer thickness. Preferably, the first and second layer thicknesses are dimensioned " \
              "such that d 1 ≦0.3d 2 . "
        expected_ngrams = [
                          'conductive',
                          'structure',
                          'feature',
                          'insulator',
                          'layer',
                          'substrate',
                          'fabricate',
                          'particular',
                          'process',
                          'process',
                          'layer',
                          'conductive',
                          'material',
                          'apply',
                          'insulator',
                          'layer',
                          'layer',
                          'conductive',
                          'material',
                          'field',
                          'region',
                          'feature',
                          'fill',
                          'feature',
                          'themselves',
                          'grain',
                          'differential',
                          'conductive',
                          'material',
                          'field',
                          'region',
                          'conductive',
                          'material',
                          'fill',
                          'feature',
                          'establish',
                          'anneal',
                          'layer',
                          'conductive',
                          'material',
                          'conductive',
                          'material',
                          'remove',
                          'uncover',
                          'field',
                          'region',
                          'leave',
                          'conductive',
                          'structure',
                          'layer',
                          'conductive',
                          'material',
                          'apply',
                          'define',
                          'first',
                          'layer',
                          'thickness',
                          'field',
                          'region',
                          'second',
                          'layer',
                          'thickness',
                          'feature',
                          'thickness',
                          'dimension',
                          '0.5d',
                          'first',
                          'layer',
                          'thickness',
                          'second',
                          'layer',
                          'thickness',
                          'preferably',
                          'first',
                          'second',
                          'layer',
                          'thickness',
                          'dimension',
                          '0.3d',
                          'conductive structure',
                          'structure feature',
                          'feature insulator',
                          'insulator layer',
                          'layer substrate',
                          'substrate fabricate',
                          'fabricate particular',
                          'particular process',
                          'layer conductive',
                          'conductive material',
                          'material apply',
                          'apply insulator',
                          'insulator layer',
                          'layer layer',
                          'layer conductive',
                          'conductive material',
                          'material cover',
                          'cover field',
                          'field region',
                          'region adjacent',
                          'adjacent feature',
                          'feature fill',
                          'fill feature',
                          'feature themselves',
                          'grain differential',
                          'differential conductive',
                          'conductive material',
                          'material cover',
                          'cover field',
                          'field region',
                          'region conductive',
                          'conductive material',
                          'material fill',
                          'fill feature',
                          'feature establish',
                          'establish anneal',
                          'anneal layer',
                          'layer conductive',
                          'conductive material',
                          'conductive material',
                          'material remove',
                          'remove uncover',
                          'uncover field',
                          'field region',
                          'region leave',
                          'leave conductive',
                          'conductive structure',
                          'layer conductive',
                          'conductive material',
                          'material apply',
                          'apply define',
                          'define first',
                          'first layer',
                          'layer thickness',
                          'thickness field',
                          'field region',
                          'region second',
                          'second layer',
                          'layer thickness',
                          'thickness feature',
                          'thickness dimension',
                          'first layer',
                          'layer thickness',
                          'second layer',
                          'layer thickness',
                          'first second',
                          'second layer',
                          'layer thickness',
                          'thickness dimension',
                          'conductive structure feature',
                          'structure feature insulator',
                          'feature insulator layer',
                          'insulator layer substrate',
                          'layer substrate fabricate',
                          'substrate fabricate particular',
                          'fabricate particular process',
                          'layer conductive material',
                          'conductive material apply',
                          'material apply insulator',
                          'apply insulator layer',
                          'insulator layer layer',
                          'layer layer conductive',
                          'layer conductive material',
                          'conductive material cover',
                          'material cover field',
                          'cover field region',
                          'field region adjacent',
                          'region adjacent feature',
                          'adjacent feature fill',
                          'feature fill feature',
                          'fill feature themselves',
                          'grain differential conductive',
                          'differential conductive material',
                          'conductive material cover',
                          'material cover field',
                          'cover field region',
                          'field region conductive',
                          'region conductive material',
                          'conductive material fill',
                          'material fill feature',
                          'fill feature establish',
                          'feature establish anneal',
                          'establish anneal layer',
                          'anneal layer conductive',
                          'layer conductive material',
                          'conductive material remove',
                          'material remove uncover',
                          'remove uncover field',
                          'uncover field region',
                          'field region leave',
                          'region leave conductive',
                          'leave conductive structure',
                          'layer conductive material',
                          'conductive material apply',
                          'material apply define',
                          'apply define first',
                          'define first layer',
                          'first layer thickness',
                          'layer thickness field',
                          'thickness field region',
                          'field region second',
                          'region second layer',
                          'second layer thickness',
                          'layer thickness feature',
                          'first layer thickness',
                          'second layer thickness',
                          'first second layer',
                          'second layer thickness',
                          'layer thickness dimension'
        ]
        actual_ngrams = WordAnalyzer.analyzer(doc)
        self.assertListEqual(expected_ngrams, actual_ngrams)
