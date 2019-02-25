import pandas as pd
import unittest
from scripts import FilePaths
from scripts.documents_filter import DocumentsFilter


class TestDocumentsFilter(unittest.TestCase):

    def setUp(self):
        df = pd.read_pickle(FilePaths.us_patents_random_100_pickle_name)
        self.__df = df.reset_index()

    def test_filter_cpc_Y02(self):
        doc_ids = DocumentsFilter(self.__df, None, 'union', 'Y02').doc_indices
        self.assertListEqual(list(doc_ids), [95])

    def test_filter_cpc_A61(self):
        doc_ids = DocumentsFilter(self.__df, None, 'union', 'A61').doc_indices
        self.assertListEqual(list(doc_ids), [67, 69, 72, 74, 11, 13, 17, 81, 85, 90, 94, 43, 50, 57, 60, 63])

    def test_filter_time(self):
        doc_ids = DocumentsFilter(self.__df, None, 'union', None, year_from='2010', month_from='06',
                                  dates_header='publication_date').doc_indices

        self.assertListEqual(list(doc_ids), [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                   51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                   76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

    def test_filter_cpc_A61_union_time(self):
        doc_ids = DocumentsFilter(self.__df, None, 'union', 'A61', year_from='2010', month_from='06',
                                  dates_header='publication_date').doc_indices
        self.assertListEqual(list(doc_ids),
                             [11, 13, 17, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                              48, 49, 50,
                              51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                              73, 74, 75,
                              76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                              98, 99])

    def test_filter_cpc_A61_intersection_time(self):
        doc_ids = DocumentsFilter(self.__df, None, 'intersection', 'A61', year_from='2010', month_from='06',
                                  dates_header='publication_date').doc_indices
        self.assertListEqual(list(doc_ids),
                             [67, 69, 72, 74, 43, 81, 50, 85, 57, 90, 60, 94, 63])


if __name__ == '__main__':
    unittest.main()