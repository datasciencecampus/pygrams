import os.path
import platform
import unittest

from scripts import FilePaths


class TestFilePaths(unittest.TestCase):

    def setUp(self):
        self.platformname = platform.system()

    def test_us_patents_random_100_pickle_name(self):
        self.assertTrue(os.path.isfile(os.path.join('..', FilePaths.us_patents_random_100_pickle_name)))

    def test_us_patents_random_1000_pickle_name(self):
        self.assertTrue(os.path.isfile(os.path.join('..', FilePaths.us_patents_random_1000_pickle_name)))

    def test_us_patents_random_10000_pickle_name(self):
        self.assertTrue(os.path.isfile(os.path.join('..', FilePaths.us_patents_random_10000_pickle_name)))

    def test_global_stopwords_filename(self):
        self.assertTrue(os.path.exists(FilePaths.global_stopwords_filename))

    def test_ngram_stopwords_filename(self):
        self.assertTrue(os.path.exists(FilePaths.ngram_stopwords_filename))

    def test_unigram_stopwords_filename(self):
        self.assertTrue(os.path.exists(FilePaths.unigram_stopwords_filename))


if __name__ == '__main__':
    unittest.main()
