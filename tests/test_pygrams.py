import os
import unittest
from unittest import mock

import pygrams2
from scripts.utils.pygrams_exception import PygramsException


class TestPyGrams(unittest.TestCase):

    def test_args_json_not_requested(self):
        args = pygrams2.get_args([])
        self.assertFalse(args.json)

    def test_args_json_requested_short(self):
        args = pygrams2.get_args(['-j'])
        self.assertTrue(args.json)

    def test_args_json_requested_long(self):
        args = pygrams2.get_args(['--json'])
        self.assertTrue(args.json)

    def test_args_output_name_requested_long(self):
        args = pygrams2.get_args(['--outputs_name=my/test/name.txt'])
        self.assertEqual('my/test/name.txt', args.outputs_name)

    def test_args_document_source_requested_long(self):
        args = pygrams2.get_args(['--doc_source=my-test'])
        self.assertEqual('my-test', args.doc_source)

    @mock.patch("scripts.output_factory.json.dump", create=True)
    @mock.patch("scripts.output_factory.open", create=True)
    def test_json_configuration_encoding_sum_no_time_weighting(self, mock_open, mock_json_dump):
        patent_pickle_file_name = 'USPTO-random-100.pkl.bz2'
        patent_pickle_absolute_file_name = os.path.abspath(os.path.join('data', patent_pickle_file_name))
        output_file_name = 'test'
        report_file_name = os.path.join('outputs', 'reports', output_file_name + '.txt')
        json_file_name = os.path.join('outputs', 'reports', output_file_name + '.json')
        pygrams2.main(['-j', f'--outputs_name={output_file_name}', '-c', '-f=set', '-p=sum', '-cpc=Y12',
                       '-yf=1999', '-yt=2000', '-dh', 'publication_date', '-ds', patent_pickle_file_name])

        mock_open.assert_called_with(json_file_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_absolute_file_name,
                'tech_report': report_file_name
            },
            'month_year': {
                'from': '1999-01-01',
                'to': '2000-12-31'
            },
            'parameters': {
                'pick': 'sum',
                'time': False
            }
        }
        self.assertEqual(expected_json, actual_json)

    @mock.patch("scripts.output_factory.json.dump", create=True)
    @mock.patch("scripts.output_factory.open", create=True)
    def test_json_configuration_encoding_maximal_and_time_weighting(self, mock_open, mock_json_dump):
        patent_pickle_file_name = 'USPTO-random-100.pkl.bz2'
        patent_pickle_absolute_file_name = os.path.abspath(os.path.join('data', patent_pickle_file_name))
        output_file_name = 'test'
        report_file_name = os.path.join('outputs', 'reports', output_file_name + '.txt')
        json_file_name = os.path.join('outputs', 'reports', output_file_name + '.json')
        pygrams2.main(['-j', f'--outputs_name={output_file_name}', '-c', '-t', '-f=set', '-p=max', '-cpc=Y12',
                       '-yf=1998', '-yt=2001', '-dh', 'publication_date', '-ds', patent_pickle_file_name])

        mock_open.assert_called_with(json_file_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_absolute_file_name,
                'tech_report': report_file_name
            },
            'month_year': {
                'from': '1998-01-01',
                'to': '2001-12-31'
            },
            'parameters': {
                'pick': 'max',
                'time': True
            }
        }
        self.assertEqual(expected_json, actual_json)

    @mock.patch("scripts.terms_graph.json.dump", create=True)
    @mock.patch("scripts.terms_graph.open", create=True)
    def test_graph_creation(self, mock_open, mock_json_dump):
        fname = 'other'
        js_file_name = os.path.join('outputs', 'visuals', 'key-terms.js')
        json_file_name = os.path.join('outputs', 'reports', 'key-terms.json')
        graph_report_name = os.path.join('outputs', 'reports', fname + '_graph.txt')

        test_args = ['--doc_source', 'USPTO-random-100.pkl.bz2', '-o', 'graph', '--outputs_name', fname]
        pygrams2.main(test_args)

        mock_open.assert_any_call(json_file_name, 'w')
        mock_open.assert_any_call(js_file_name, 'w')
        mock_open.assert_any_call(graph_report_name, 'w')

        actual_json = mock_json_dump.call_args_list[0][0][0]
        self.assertIn('nodes', actual_json)
        self.assertIn('links', actual_json)

    @mock.patch("os.path.isfile", create=True)
    def test_reports_unsupported_df_format(self, mock_path_isfile):

        unknown_filename = 'unknown.format'

        def isfile_fake(file_name):
            if file_name == os.path.join('data', unknown_filename):
                return True
            else:
                return False

        mock_path_isfile.side_effect = isfile_fake
        test_args = ['--doc_source', unknown_filename]
        try:
            pygrams2.main(test_args)
            self.fail("should raise exception")
        except PygramsException as err:
            self.assertEqual('Unsupported file: ' + os.path.join('data', unknown_filename), err.message)


if __name__ == '__main__':
    unittest.main()
