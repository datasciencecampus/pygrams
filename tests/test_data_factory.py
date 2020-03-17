import unittest

import pandas as pd

from scripts import FilePaths
from scripts import data_factory as factory


class TestDataFactory(unittest.TestCase):

    def setUp(self):
        self.__df = pd.read_pickle(FilePaths.us_patents_random_100_pickle_name)
        self.__df = self.__df.reset_index()

    def test_reads_xls(self):
        df = factory.get('data/USPTO-random-100.xls')
        self.assertListEqual(list(self.__df['abstract']), list(df['abstract']))

    def test_reads_xlsx(self):
        df = factory.get('data/USPTO-random-100.xlsx')
        self.assertListEqual(list(self.__df['abstract']), list(df['abstract']))

    @unittest.skip('Unicode char fails under windows; see task #172 bug')
    def test_reads_csv(self):
        df = factory.get('data/USPTO-random-100.csv')
        self.assertListEqual(list(self.__df['abstract']), list(df['abstract']))

    def test_reads_pickles(self):
        df = factory.get('data/USPTO-random-100.pkl.bz2')
        self.assertEqual(len(df['abstract']), 100)


if __name__ == '__main__':
    unittest.main()
