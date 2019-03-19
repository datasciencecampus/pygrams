import unittest

import pandas as pd
from pandas.io.excel import register_writer

from scripts.tfidf_wrapper import TFIDF
from scripts.text_processing import LemmaTokenizer
from scripts.utils.table_output import table_output
from tests.utils import ReferenceData


class FakeArgs(object):
    __slots__ = ['pick', 'time', 'focus']

@unittest.skip("Temporarily shut down module")
class TestTableOutput(unittest.TestCase):

    class FakeWriter(pd.ExcelWriter):
        engine = 'fakewriter'
        supported_extensions = ('.fake',)

        def __init__(self, path, engine=None, **kwargs):
            self.engine = engine
            self.path = path
            self.sheets = {}

        def write_cells(self, cells, sheet_name=None, startrow=0, startcol=0, freeze_panes=None):
            if sheet_name is None:
                raise ValueError('Must provide sheet_name')

            if sheet_name in self.sheets:
                raise ValueError(f'{sheet_name} already in self.sheets')

            cell_list = list(cells)
            max_row = 0
            max_col = 0
            for cell in cell_list:
                if cell.col > max_col:
                    max_col = cell.col

                if cell.row > max_row:
                    max_row = cell.row
            self.sheets[sheet_name] = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            for cell in cell_list:
                self.sheets[sheet_name][cell.row][cell.col] = cell.val

        def save(self):
            pass

    def test_table(self):
        max_n = 3
        min_n = 2

        ngram_multiplier = 4

        num_ngrams_report = 25
        num_ngrams_wordcloud = 25

        num_ngrams = max(num_ngrams_report, num_ngrams_wordcloud)

        tfidf_cold = TFIDF(ReferenceData.cold_df, tokenizer=LemmaTokenizer(), ngram_range=(min_n, max_n))
        tfidf_random = TFIDF(ReferenceData.random_df, tokenizer=LemmaTokenizer(), ngram_range=(min_n, max_n))

        citation_count_dict = {1: 10, 2: 3, 101: 2, 102: 0, 103: 5, 104: 4, 105: 10}

        args=FakeArgs()

        args.pick = 'sum'
        args.time = False
        args.focus = 'chi2'

        register_writer(TestTableOutput.FakeWriter)
        fake_writer = TestTableOutput.FakeWriter('spreadsheet.fake')

        table_output(tfidf_cold, tfidf_random, num_ngrams, args, ngram_multiplier, fake_writer, citation_count_dict)

        # Check sheet headings...
        self.assertListEqual(
            [None, 'Term', 'Score', 'Rank', 'Focus chi2 Score', 'Focus chi2 Rank', 'Diff Base to Focus Rank',
                              'Time Score', 'Time Rank', 'Diff Base to Time Rank', 'Citation Score', 'Citation Rank',
                              'Diff Base to Citation Rank'], fake_writer.sheets['Summary'][0])

        self.assertListEqual([None, 'Term', 'Score', 'Rank'], fake_writer.sheets['Base'][0])
        self.assertListEqual([None, 'Term', 'Focus chi2 Score', 'Focus chi2 Rank'], fake_writer.sheets['Focus'][0])
        self.assertListEqual([None, 'Term', 'Time Score', 'Time Rank'], fake_writer.sheets['Time'][0])
        self.assertListEqual([None, 'Term', 'Citation Score', 'Citation Rank'], fake_writer.sheets['Cite'][0])

        # Base sheet should match summary sheet
        for y in range(25):
            for x in range(4):
                self.assertEqual(fake_writer.sheets['Summary'][y + 1][x], fake_writer.sheets['Base'][y + 1][x])
