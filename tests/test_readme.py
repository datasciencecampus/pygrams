import os
import sys
import unittest

import pytest

import pygrams


@pytest.mark.skipif(('TRAVIS' not in os.environ) or (sys.platform == 'win32'), reason="Only execute with Travis due to speed")
class TestReadme(unittest.TestCase):
    """
    Batch of tests to execute same commands as mentioned in the README.md, to ensure they work without crashing.
    Note that the tests need to be run at the main user folder as this shows the files exist - not to be run in the
    tests folder.
    """

    @classmethod
    def setUpClass(cls):
        os.chdir('..')

    @classmethod
    def tearDownClass(cls):
        os.chdir('tests')

    def test_no_arguments_and_use_cache(self):
        # clear cached result
        import shutil
        shutil.rmtree(os.path.join('cached', 'out-mdf-0.05'), ignore_errors=True)

        # should make cache
        pygrams.main([])

        # load cache
        pygrams.main(['-uc', 'out-mdf-0.05'])

    def test_10000_patents(self):
        pygrams.main(['-ds', 'USPTO-random-10000.pkl.bz2'])

    def test_mn_mx_uni_bi_trigrams(self):
        pygrams.main(['-mn', '1', '-mx', '3'])

    def test_mn_mx_unigrams(self):
        pygrams.main(['-mn', '1', '-mx', '1'])

    def test_mdf(self):
        pygrams.main(['-mdf', '0.05'])

    def test_pt(self):
        pygrams.main(['-pt', '0'])

    def test_prefilter_terms_10000(self):
        pygrams.main(['--prefilter_terms', '10000'])

    def test_date_from(self):
        pygrams.main(['-dh', 'publication_date', '-df', '2000/02/20'])

    def test_date_from_and_to(self):
        pygrams.main(['-dh', 'publication_date', '-df', '2000/03/01', '-dt', '2016/07/31'])

    # def test_filter(self):
    #     pygrams.main(['-fc', "['female','british']", '-fb', 'union'])

    def test_cpc(self):
        pygrams.main(['-cpc', 'Y02', '-ds', 'USPTO-random-10000.pkl.bz2'])

    def test_search_terms(self):
        pygrams.main(['-st', 'pharmacy', 'medicine', 'chemist'])

    def test_wordcloud(self):
        pygrams.main(['-o', 'wordcloud'])

    def test_multiplot(self):
        pygrams.main(['-o', 'multiplot', '-ts', '-dh', 'publication_date'])

    # def test_help(self):
    #     pygrams.main(['-h'])
