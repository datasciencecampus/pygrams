import os
import unittest
from unittest import mock

import pygrams


class TestPyGrams(unittest.TestCase):

    def test_args_json_not_requested(self):
        args = pygrams.get_args([])
        self.assertFalse(args.json)

    def test_args_json_requested_short(self):
        args = pygrams.get_args(['-j'])
        self.assertTrue(args.json)

    def test_args_json_requested_long(self):
        args = pygrams.get_args(['--json'])
        self.assertTrue(args.json)

    def test_args_report_name_requested_long(self):
        args = pygrams.get_args(['--report_name=my/test/name.txt'])
        self.assertEqual('my/test/name.txt', args.report_name)

    def test_args_document_source_requested_long(self):
        args = pygrams.get_args(['--doc_source=my-test'])
        self.assertEqual('my-test', args.doc_source)

    @mock.patch("pygrams.json.dump", create=True)
    @mock.patch("pygrams.open", create=True)
    def test_json_configuration_encoding_minimal(self, mock_open, mock_json_dump):
        patent_pickle_file_name = 'test.pkl'
        patent_pickle_absolute_file_name = os.path.abspath('test.pkl')
        report_file_name = os.path.join(os.path.abspath(os.sep), 'dummy', 'test.txt')
        json_file_name = os.path.join(os.path.abspath(os.sep), 'dummy', 'test.json')
        args = pygrams.get_args(['-j', f'--report_name={report_file_name}'])

        pygrams.write_config_to_json(args, patent_pickle_file_name)

        self.assertTrue(args.json)
        mock_open.assert_called_with(json_file_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_absolute_file_name,
                'tech_report': report_file_name
            },
            'year': {
                'from': '01_2000',
                'to': '12_2019'
            },
            'parameters': {
                'cite': False,
                'cpc': '',
                'focus': None,
                'pick': 'sum',
                'time': False
            }
        }
        self.assertEqual(expected_json, actual_json)

    @mock.patch("pygrams.json.dump", create=True)
    @mock.patch("pygrams.open", create=True)
    def test_json_configuration_encoding_maximal(self, mock_open, mock_json_dump):
        patent_pickle_file_name = os.path.join('dummy', 'test.pkl')
        patent_pickle_absolute_file_name = os.path.abspath(patent_pickle_file_name)
        report_file_name = os.path.join(os.path.abspath(os.sep), 'dummy', 'test.txt')
        json_file_name = os.path.join(os.path.abspath(os.sep), 'dummy', 'test.json')
        args = pygrams.get_args(['-j', f'--report_name={report_file_name}', '-c', '-t', '-f=set', '-p=max', '-cpc=Y12',
                                 '-yf=1998', '-yt=2001'])

        pygrams.write_config_to_json(args, patent_pickle_file_name)

        self.assertTrue(args.json)
        mock_open.assert_called_with(json_file_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_absolute_file_name,
                'tech_report': report_file_name
            },
            'year': {
                'from': '01_1998',
                'to': '12_2001'
            },
            'parameters': {
                'cite': True,
                'cpc': 'Y12',
                'focus': 'set',
                'pick': 'max',
                'time': True
            }
        }
        self.assertEqual(expected_json, actual_json)

    @mock.patch("scripts.visualization.graphs.terms_graph.json.dump", create=True)
    @mock.patch("scripts.visualization.graphs.terms_graph.open", create=True)
    def test_fdg_creation(self, mock_open, mock_json_dump):
        fname = 'key-terms'
        js_file_name = os.path.join('outputs', 'visuals', fname + '.js')
        json_file_name = os.path.join('outputs', 'reports', fname + '.json')
        report_name = os.path.join('outputs', 'reports', 'report_tech.txt')
        graph_report_name = report_name[:len(report_name) - 4] + "_graph.txt"

        test_args = ['--doc_source', 'USPTO-random-1000.pkl.bz2', '-o', 'fdg', '--report_name', report_name]
        pygrams.main(test_args)

        mock_open.assert_any_call(json_file_name, 'w')
        mock_open.assert_any_call(js_file_name, 'w')
        mock_open.assert_any_call(graph_report_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        self.assertIn('nodes', actual_json)
        self.assertIn('links', actual_json)


if __name__ == '__main__':
    unittest.main()
