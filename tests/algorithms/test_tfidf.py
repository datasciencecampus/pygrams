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
            '''A system and method are disclosed for providing improved processing of video data. A multi-instance 
        encoding module receives combined video and audio input, which is then separated into a video and audio 
        source streams. The video source stream is pre-processed and corresponding video encoder instances are 
        initiated. The preprocessed video source stream is split into video data components, which are assigned to a 
        corresponding encoder instance. Encoding operations are performed by each video encoder instance to generate 
        video output components. The video output components are then assembled in a predetermined sequence to 
        generate an encoded video output stream. Concurrently, the audio source stream is encoded with an audio 
        encoder to generate an encoded audio output stream. The encoded video and audio output streams are combined 
        to generate a combined encoded output stream, which is provided as combined video and audio output. ''')

        df = pd.read_pickle(FilePaths.us_patents_random_1000_pickle_name)
        df2 = pd.DataFrame([['US09315032-20160419', abstract_from_US09315032_20160419, None, None]],
                           columns=['patent_id', 'abstract', 'classifications_cpc', 'publication_date'])
        df = df.append(df2)

        tfidf_engine = TFIDF(df, ngram_range=(2, 4), tokenizer=StemTokenizer(), normalize_doc_length=False)
        actual_popular_terms, ngrams_scores = tfidf_engine.detect_popular_ngrams_in_docs_set(docs_set=[1000])
        actual_scores = [score[0] for score in ngrams_scores]
        expected_popular_terms = ['audio encod', 'audio sourc stream', 'encod audio output stream']
        expected_scores = [0.17716887713942972, 0.08858443856971486, 0.08858443856971486, 0.08858443856971486, 0.08858443856971486]

        expected_in_results = expected_popular_terms[0] in actual_popular_terms and expected_popular_terms[
            1] in actual_popular_terms

        self.assertEquals(expected_in_results, True)
        self.assertEqual(expected_scores, actual_scores[:5])

    def test_popular_terms_from_1000_samples_via_stems_and_normalization(self):
        abstract_from_US09315032_20160419 = (
            '''A system and method are disclosed for providing improved processing of video data. A multi-instance 
        encoding module receives combined video and audio input, which is then separated into a video and audio 
        source streams. The video source stream is pre-processed and corresponding video encoder instances are 
        initiated. The preprocessed video source stream is split into video data components, which are assigned to a 
        corresponding encoder instance. Encoding operations are performed by each video encoder instance to generate 
        video output components. The video output components are then assembled in a predetermined sequence to 
        generate an encoded video output stream. Concurrently, the audio source stream is encoded with an audio 
        encoder to generate an encoded audio output stream. The encoded video and audio output streams are combined 
        to generate a combined encoded output stream, which is provided as combined video and audio output. ''')

        df = pd.read_pickle(FilePaths.us_patents_random_1000_pickle_name)
        df2 = pd.DataFrame([['US09315032-20160419', abstract_from_US09315032_20160419, None, None]],
                           columns=['patent_id', 'abstract', 'classifications_cpc', 'publication_date'])
        df = df.append(df2)

        tfidf_engine = TFIDF(df, ngram_range=(2, 4), tokenizer=StemTokenizer(), normalize_doc_length=True)
        actual_popular_terms, ngrams_scores = tfidf_engine.detect_popular_ngrams_in_docs_set(docs_set=[1000])
        actual_scores = [score[0] for score in ngrams_scores]
        expected_popular_terms = ['audio encod', 'audio sourc stream', 'encod audio output stream']
        expected_scores = [0.17716887713942953, 0.08858443856971476, 0.08858443856971476, 0.08858443856971476, 0.08858443856971476]

        expected_in_results = expected_popular_terms[0] in actual_popular_terms and expected_popular_terms[
            1] in actual_popular_terms

        self.assertEquals(expected_in_results, True)
        self.assertEqual(expected_scores, actual_scores[:5])


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
