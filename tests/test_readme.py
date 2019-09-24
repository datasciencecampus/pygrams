import os
import unittest

import pytest

import pygrams


@pytest.mark.skipif('TRAVIS' not in os.environ, reason="Only execute with Travis due to speed")
class TestReadme(unittest.TestCase):
    """
    Batch of tests to execute same commands as mentioned in the README.md, to ensure they work without crashing
    """

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

    def test_mn_mx(self):
        pygrams.main(['-mn', '1', '-mx', '3'])
