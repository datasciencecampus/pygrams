import unittest
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd

from scripts.utils.argschecker import ArgsChecker
from scripts.utils.pygrams_exception import PygramsException


class TestArgsChecker(unittest.TestCase):

    def setUpTest(self, mock_path):
        self.args = MagicMock()
        self.args.path = 'bogus path'
        self.args.doc_source = 'bogus source'
        self.args.min_ngrams = 1
        self.args.max_ngrams = 3
        self.args.num_ngrams_wordcloud = 20
        self.args.num_ngrams_report = 22
        self.args.wordcloud_title = 'default title'
        self.args.table_name = 'default table name'

        self.args_default = MagicMock()
        self.args_default.num_ngrams_report = 22
        self.args_default.num_ngrams_wordcloud = 20
        self.args_default.wordcloud_title = 'default title'
        self.args_default.table_name = 'default table name'

        mock_path.join = MagicMock()
        mock_path.join.return_value = ''
        mock_path.isfile = MagicMock()
        mock_path.isfile.return_value = True

    @mock.patch("scripts.utils.argschecker.path", create=True)
    def test_to_date_correct_format(self, mock_path):
        self.setUpTest(mock_path)
        self.args.date_to = '2019/03/10'
        args_checker = ArgsChecker(self.args, self.args_default)

        args_checker.checkargs()

    @mock.patch("scripts.utils.argschecker.path", create=True)
    def test_to_date_incorrect_format(self, mock_path):
        self.setUpTest(mock_path)
        bogus_date = '03/10'
        self.args.date_to = bogus_date
        args_checker = ArgsChecker(self.args, self.args_default)

        try:
            args_checker.checkargs()
            self.fail('Should have detected erroneous date format')

        except PygramsException as pe:
            self.assertEqual(f"date_to defined as '{bogus_date}' which is not in YYYY/MM/DD format",
                             pe.message, 'Messages do not match')

    @mock.patch("scripts.utils.argschecker.path", create=True)
    def test_from_date_correct_format(self, mock_path):
        self.setUpTest(mock_path)
        self.args.date_from = '2018/03/10'
        args_checker = ArgsChecker(self.args, self.args_default)

        args_checker.checkargs()

    @mock.patch("scripts.utils.argschecker.path", create=True)
    def test_from_date_incorrect_format(self, mock_path):
        self.setUpTest(mock_path)
        bogus_date = '03/11'
        self.args.date_from = bogus_date
        args_checker = ArgsChecker(self.args, self.args_default)

        try:
            args_checker.checkargs()
            self.fail('Should have detected erroneous date format')

        except PygramsException as pe:
            self.assertEqual(f"date_from defined as '{bogus_date}' which is not in YYYY/MM/DD format",
                             pe.message, 'Messages do not match')

    @mock.patch("scripts.utils.argschecker.path", create=True)
    def test_from_date_not_after_to_date(self, mock_path):
        self.setUpTest(mock_path)
        self.args.date_from = '2018/03/10'
        self.args.date_to = '2019/03/10'
        args_checker = ArgsChecker(self.args, self.args_default)

        args_checker.checkargs()

    @mock.patch("scripts.utils.argschecker.path", create=True)
    def test_from_date_after_to_date(self, mock_path):
        self.setUpTest(mock_path)
        self.args.date_from = '2019/03/11'
        self.args.date_to = '2019/03/10'
        args_checker = ArgsChecker(self.args, self.args_default)

        try:
            args_checker.checkargs()
            self.fail('Should have detected date_from > date_to')

        except PygramsException as pe:
            self.assertEqual(f"date_from '{self.args.date_from}' cannot be after date_to '{self.args.date_to}'",
                             pe.message, 'Messages do not match')

    @mock.patch("scripts.utils.argschecker.path", create=True)
    def test_get_docs_mask_dict(self, mock_path):
        expected_date_from = 201910
        expected_date_to = 201911
        self.setUpTest(mock_path)
        self.args.date_from = pd.Timestamp('2019/03/10')
        self.args.date_to = pd.Timestamp('2019/03/11')

        self.args.timeseries_date_from = pd.Timestamp('2019/03/10')
        self.args.timeseries_date_to = pd.Timestamp('2019/03/11')

        args_checker = ArgsChecker(self.args, self.args_default)

        args_dict = args_checker.get_docs_mask_dict()

        self.assertEqual(expected_date_from, args_dict['date']['from'])
        self.assertEqual(expected_date_to, args_dict['date']['to'])

    @mock.patch("scripts.utils.argschecker.path", create=True)
    def test_get_docs_mask_dict_date_from_to_None(self, mock_path):
        self.setUpTest(mock_path)
        self.args.date_from = None
        self.args.date_to = None

        self.args.timeseries_date_from = None
        self.args.timeseries_date_to = None

        args_checker = ArgsChecker(self.args, self.args_default)

        args_dict = args_checker.get_docs_mask_dict()

        self.assertIsNone(args_dict['date'])
